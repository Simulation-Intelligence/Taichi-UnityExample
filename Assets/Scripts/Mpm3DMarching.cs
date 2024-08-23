using System.Collections.Generic;
using UnityEngine;
using Taichi;
using UnityEngine.Rendering;
using System.Linq;
using System.Text.RegularExpressions;
using System.Diagnostics;
using Meta.WitAi.CallbackHandlers;
using UnityEngine.UIElements;
using System.IO;
using System.Text;
using System;
using UnityEngine.InputSystem;
using Oculus.Interaction;
using Oculus.Interaction.HandGrab;
using static SkeletonRenderer;
using GaussianSplatting.Runtime;
using UnityEngine.Experimental.Rendering;
using MarchingCubes;

public class Mpm3DMarching : MonoBehaviour
{
    public bool isInitialized = false;
    private Mesh _Mesh;
    private MeshFilter _MeshFilter;
    private MeshRenderer _MeshRenderer;

    public VolumeTextureUpdater volumeTextureUpdater;

    [Header("MpM Engine")]
    [SerializeField]
    private AotModuleAsset Mpm3DModule;
    private Kernel _Kernel_subsetep_reset_grid, _Kernel_substep_neohookean_p2g, _Kernel_substep_Kirchhoff_p2g,
    _Kernel_substep_calculate_signed_distance_field, _Kernel_substep_update_grid_v, _Kernel_substep_g2p,
     _Kernel_substep_apply_Von_Mises_plasticity, _Kernel_substep_apply_Drucker_Prager_plasticity, _Kernel_substep_p2g, _Kernel_substep_apply_plasticity,
     _Kernel_substep_apply_clamp_plasticity, _Kernel_substep_calculate_hand_sdf, _Kernel_substep_get_max_speed, _Kernel_substep_calculate_hand_hash, _Kernel_substep_adjust_particle, _Kernel_substep_calculate_hand_sdf_hash,
     _Kernel_init_dg, _Kernel_init_gaussian_data, _Kernel_substep_update_gaussian_data, _Kernel_scale_to_unit_cube, _Kernel_init_sphere, _Kernel_init_cylinder, _Kernel_init_torus,
     _Kernel_normalize_m, _Kernel_transform_and_merge, _Kernel_substep_fix_object, _Kernel_substep_p2g_multi,
        _Kernel_copy_array_1dim1, _Kernel_copy_array_1dim3, _Kernel_copy_array_1dim1I;

    public enum RenderType
    {
        PointMesh,
        Raymarching,
        GaussianSplat,
        MarchingCubes
    }
    public enum MaterialType
    {
        Customize,
        Clay,
        Dough,
        Elastic_Material
    }
    public enum InitShape
    {
        Cube,
        Sphere,
        Cylinder,
        Torus
    }
    public enum PlasticityType
    {
        Von_Mises,
        Drucker_Prager,
        Clamp,
        Elastic
    }
    public enum StressType
    {
        NeoHookean,
        Kirchhoff
    }
    public enum ObstacleType
    {
        Sphere,
        Hand
    }
    [Header("Material")]
    [SerializeField]
    public RenderType renderType = RenderType.GaussianSplat;

    private RenderType lastRenderType = RenderType.GaussianSplat;
    [SerializeField]
    private Material handMaterial;
    [SerializeField]
    private Material pointMaterial;
    [SerializeField]
    private Material raymarchingMaterial;
    [SerializeField]
    public MaterialType materialType = MaterialType.Customize;
    [SerializeField]
    public PlasticityType plasticityType = PlasticityType.Von_Mises;
    [SerializeField]
    public StressType stressType = StressType.NeoHookean;
    private Kernel _Kernel_init_particles;
    private NdArray<float> x, v, C, dg, grid_v, grid_m, sphere_pos, obstacle_velocities, sphere_velocities, sphere_radius, hand_sdf, marching_m;

    public NdArray<float> skeleton_segments, skeleton_velocities, obstacle_normals, skeleton_capsule_radius, max_v;
    public NdArray<float> E, SigY, nu, min_clamp, max_clamp, alpha, p_vol, p_mass;

    public NdArray<float> init_rotation, init_scale, init_sh, other_data, sh;
    private NdArray<int> segments_count_per_cell, hash_table, material, point_color;

    private float[] hand_skeleton_segments, hand_skeleton_segments_prev, hand_skeleton_velocities, _skeleton_capsule_radius, sphere_positions, _sphere_velocities, sphere_radii;

    private Bounds bounds;

    private ComputeGraph _Compute_Graph_g_init;
    private ComputeGraph _Compute_Graph_g_substep;

    [Header("Scene Settings")]
    [SerializeField]
    bool RunSimulation = true;
    private bool updated = false;
    private Grabbable _grabbable;
    [SerializeField]
    public GaussianSplatRenderManager splatManager;

    [SerializeField]
    MarchingCubeVisualizer[] marchingCubeVisualizers;

    public ComputeShader copyShader;

    private ComputeBuffer marching_m_computeBuffer;

    [SerializeField]
    public InitShape initShape = InitShape.Cube;

    [SerializeField]
    private Vector3 g = new(0, -9.8f, 0);

    private float gy;
    [SerializeField]
    public int n_grid = 32, bound = 3;

    [SerializeField]
    private float bounding_eps = 0.1f;
    [SerializeField]
    public float max_dt = 1e-4f, frame_time = 0.005f, particle_per_grid = 8, allowed_cfl = 0.5f, damping = 1f;
    public float cube_size = 0.2f, cylinder_height = 0.9f, cylinder_radius = 0.05f, torus_radius = 0.3f, torus_tube_radius = 0.05f;
    [SerializeField]
    public bool use_correct_cfl = false;

    [Header("Interaction Settings")]
    [SerializeField]
    private float hand_simulation_radius = 0.5f;
    private Vector3 boundary_min, boundary_max;

    // Fix the object in place
    [SerializeField]
    private bool is_fixed = false;
    private Vector3 fix_center = new Vector3(0.5f, 0.5f, 0.5f);
    private float fix_radius = 0.2f;
    // Use sticky boundary condition, 1 for sticky boundary, 0 for non-sticky boundary
    [SerializeField]
    private int use_sticky_boundary = 1;

    [Header("Obstacle")]
    [SerializeField]
    public ObstacleType obstacleType = ObstacleType.Sphere;
    [SerializeField]
    private Sphere[] sphere;

    [SerializeField]
    private OVRHand[] oculus_hands;
    [SerializeField]
    private OVRSkeleton[] oculus_skeletons;
    [SerializeField]
    private float[] preset_capsule_radius;
    private int skeleton_num_capsules = 24; // use default 24
    private int NParticles;
    private float dx, _p_vol, _p_mass, v_allowed;

    [Header("Scalars")]
    [SerializeField]
    public float _E = 1e4f;
    [SerializeField]
    public float _SigY = 1000, _nu = 0.3f, colide_factor = 0.5f, friction_k = 0.4f, p_rho = 1000, _min_clamp = 0.1f, _max_clamp = 0.1f, friction_angle = 30;

    private float[] E_host, SigY_host, nu_host, min_clamp_host, max_clamp_host, alpha_host, p_vol_host, p_mass_host;

    private int[] material_host, point_color_host; //upper 16bits: 3: # Drucker_Prager  1:  # Von_Mises 2:  # Clamp 0:  # Elastic  lower 16bits: 0:  # neohookean 1:  # kirchhoff

    private float mu, lambda, sin_phi, _alpha, max_density;

    private bool isRecording = false;
    private List<float[]> handPositions = new();

    private int handMotionIndex = 0;
    private InputAction spaceAction;

    private OVRSkeletonRenderer ovrRend;

    [Header("Hand Recording")]
    [SerializeField]
    private bool UseRecordDate = false;
    [SerializeField]
    private string filePath = "HandMotionData.txt";

    private bool RendererInitialized = false; // Used for recorded hand
    private List<CapsuleVisualization> _capsuleVisualizations = new();

    // Start is called before the first frame update
    void Start()
    {
        if (!isInitialized)
        {
            Initiate();
            isInitialized = true;
        }
    }
    public void Initiate()
    {

        var kernels = Mpm3DModule.GetAllKernels().ToDictionary(x => x.Name);
        if (kernels.Count > 0)
        {
            _Kernel_subsetep_reset_grid = kernels["substep_reset_grid"];
            _Kernel_substep_neohookean_p2g = kernels["substep_neohookean_p2g"];
            _Kernel_substep_Kirchhoff_p2g = kernels["substep_kirchhoff_p2g"];
            _Kernel_substep_calculate_signed_distance_field = kernels["substep_calculate_signed_distance_field"];
            _Kernel_substep_update_grid_v = kernels["substep_update_grid_v"];
            _Kernel_substep_g2p = kernels["substep_g2p"];
            _Kernel_substep_apply_Von_Mises_plasticity = kernels["substep_apply_Von_Mises_plasticity"];
            _Kernel_substep_apply_Drucker_Prager_plasticity = kernels["substep_apply_Drucker_Prager_plasticity"];
            _Kernel_substep_apply_clamp_plasticity = kernels["substep_apply_clamp_plasticity"];
            _Kernel_init_particles = kernels["init_particles"];
            _Kernel_init_dg = kernels["init_dg"];

            // Contact with hand
            _Kernel_substep_calculate_hand_sdf = kernels["substep_calculate_hand_sdf"];
            _Kernel_substep_get_max_speed = kernels["substep_get_max_speed"];
            _Kernel_substep_calculate_hand_hash = kernels["substep_calculate_hand_hash"];
            _Kernel_substep_calculate_hand_sdf_hash = kernels["substep_calculate_hand_sdf_hash"];
            _Kernel_substep_adjust_particle = kernels["substep_adjust_particle"];
            _Kernel_substep_p2g = kernels["substep_p2g"];
            _Kernel_substep_apply_plasticity = kernels["substep_apply_plasticity"];

            // Gaussian
            _Kernel_init_gaussian_data = kernels["init_gaussian_data"];
            _Kernel_substep_update_gaussian_data = kernels["substep_update_gaussian_data"];
            _Kernel_scale_to_unit_cube = kernels["scale_to_unit_cube"];

            _Kernel_normalize_m = kernels["normalize_m"];
            _Kernel_init_sphere = kernels["init_sphere"];
            _Kernel_init_cylinder = kernels["init_cylinder"];
            _Kernel_init_torus = kernels["init_torus"];
            _Kernel_transform_and_merge = kernels["transform_and_merge"];
            _Kernel_substep_fix_object = kernels["substep_fix_object"];

            _Kernel_substep_p2g_multi = kernels["substep_p2g_multi"];

            _Kernel_copy_array_1dim1 = kernels["copy_array_1dim1"];
            _Kernel_copy_array_1dim3 = kernels["copy_array_1dim3"];
            _Kernel_copy_array_1dim1I = kernels["copy_array_1dim1I"];
        }

        var cgraphs = Mpm3DModule.GetAllComputeGrpahs().ToDictionary(x => x.Name);
        if (cgraphs.Count > 0)
        {
            _Compute_Graph_g_init = cgraphs["init"];
            _Compute_Graph_g_substep = cgraphs["substep"];
        }

        // Sphere as a obstacle
        if (sphere == null || sphere.Length == 0)
        {
            sphere = new Sphere[] { GameObject.Find("SphereLeft").GetComponent<Sphere>(),
                                    GameObject.Find("SphereRight").GetComponent<Sphere>() };
        }
        sphere_pos = new NdArrayBuilder<float>().Shape(sphere.Length).ElemShape(3).HostWrite(true).Build();
        sphere_velocities = new NdArrayBuilder<float>().Shape(sphere.Length).ElemShape(3).HostWrite(true).Build();
        sphere_radius = new NdArrayBuilder<float>().Shape(sphere.Length).HostWrite(true).Build();
        max_v = new NdArrayBuilder<float>().Shape(1).HostRead(true).Build();

        sphere_positions = new float[3 * sphere.Length];
        _sphere_velocities = new float[3 * sphere.Length];
        sphere_radii = new float[sphere.Length];

        // Oculus hands
        if (oculus_hands == null || oculus_hands.Length == 0)
        {
            oculus_hands = new OVRHand[] { GameObject.Find("OVRCameraRig/TrackingSpace/LeftHandAnchor/LeftOVRHand").GetComponent<OVRHand>(),
                                           GameObject.Find("OVRCameraRig/TrackingSpace/RightHandAnchor/RightOVRHand").GetComponent<OVRHand>() };
        }
        if (oculus_skeletons == null || oculus_skeletons.Length == 0)
        {
            oculus_skeletons = new OVRSkeleton[] { GameObject.Find("OVRCameraRig/TrackingSpace/LeftHandAnchor/LeftOVRHand").GetComponent<OVRSkeleton>(),
                                                   GameObject.Find("OVRCameraRig/TrackingSpace/RightHandAnchor/RightOVRHand").GetComponent<OVRSkeleton>() };
        }
        UnityEngine.Debug.Log("Num of bones at start: " + oculus_skeletons[0].Bones.Count());
        skeleton_segments = new NdArrayBuilder<float>().Shape(skeleton_num_capsules * oculus_skeletons.Length, 2).ElemShape(3).HostWrite(true).Build(); // 24 skeleton segments, each segment has 6 floats
        skeleton_velocities = new NdArrayBuilder<float>().Shape(skeleton_num_capsules * oculus_skeletons.Length, 2).ElemShape(3).HostWrite(true).Build(); // 24 skeleton velocities, each velocity has 6 floats
        skeleton_capsule_radius = new NdArrayBuilder<float>().Shape(skeleton_num_capsules * oculus_skeletons.Length).HostWrite(true).Build(); // use a consistent radius for all capsules (at now)

        InitGrid();

        // Initialize hand skeleton segments
        hand_skeleton_segments = new float[skeleton_num_capsules * oculus_skeletons.Length * 6];
        hand_skeleton_segments_prev = new float[skeleton_num_capsules * oculus_skeletons.Length * 6];
        hand_skeleton_velocities = new float[skeleton_num_capsules * oculus_skeletons.Length * 6];
        _skeleton_capsule_radius = new float[skeleton_num_capsules * oculus_skeletons.Length];

        _MeshRenderer = GetComponent<MeshRenderer>();
        _MeshFilter = GetComponent<MeshFilter>();
        _grabbable = GetComponent<Grabbable>();

        dx = 1.0f / n_grid;
        mu = _E / (2 * (1 + _nu));
        lambda = _E * _nu / ((1 + _nu) * (1 - 2 * _nu));
        v_allowed = allowed_cfl * dx / max_dt;

        if (renderType == RenderType.GaussianSplat)
        {
            splatManager.init_gaussians();
            Init_gaussian();
        }
        else
        {
            Init_Particles();
        }
        // if (renderType == RenderType.PointMesh)
        // {
        Init_PointMesh();
        //}
        // if (renderType == RenderType.MarchingCubes)
        // {
        Init_MarchingCubes();
        // }



        Init_materials();
        Update_materials();




        if (renderType == RenderType.Raymarching)
        {
            _MeshRenderer.material = raymarchingMaterial;
            volumeTextureUpdater.width = n_grid;
            volumeTextureUpdater.height = n_grid;
            volumeTextureUpdater.depth = n_grid;
            volumeTextureUpdater.densityData = new float[n_grid * n_grid * n_grid];
            volumeTextureUpdater.volumeTex = new RenderTexture(n_grid, n_grid, 0, RenderTextureFormat.RFloat)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = n_grid,
                enableRandomWrite = true,
                wrapMode = TextureWrapMode.Clamp
            };
            volumeTextureUpdater.volumeTex.Create();
            volumeTextureUpdater.computeBuffer = new ComputeBuffer(n_grid * n_grid * n_grid, sizeof(float));

            volumeTextureUpdater.max_density = particle_per_grid * _p_mass;
            volumeTextureUpdater.targetMaterial = GetComponent<Renderer>().material;
        }


        // 24 line segments with 24 capsules in total
        preset_capsule_radius = SkeletonRenderer.preset_capsule_radius;
        for (int i = 0; i < skeleton_num_capsules * oculus_skeletons.Length; i++)
        {
            _skeleton_capsule_radius[i] = preset_capsule_radius[i % 24] / transform.lossyScale.x;
        }
        skeleton_capsule_radius.CopyFromArray(_skeleton_capsule_radius);

        // Use the recorded hand data for simulation tests in Unity
        if (UseRecordDate)
        {
            LoadHandMotionData();
        }
        spaceAction = new InputAction(binding: "<Keyboard>/space");
        spaceAction.performed += ctx => OnSpacePressed();
        spaceAction.Enable();
    }
    public void Init_PointMesh()
    {
        _Mesh = new Mesh();
        int[] indices = new int[NParticles];
        for (int i = 0; i < NParticles; ++i)
        {
            indices[i] = i;
        }
        Vector3[] vertices = new Vector3[NParticles];

        _Mesh.vertices = vertices;
        _Mesh.SetIndices(indices, MeshTopology.Points, 0);
        _Mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 114514f);
        _Mesh.name = "Mpm3D";
        _Mesh.MarkModified();
        _Mesh.UploadMeshData(false);
        _MeshFilter.mesh = _Mesh;
        _MeshRenderer.material = pointMaterial;
        bounds = new Bounds(_MeshFilter.transform.position + Vector3.one * 0.5f, Vector3.one);

    }
    void Init_Particles()
    {
        float volume = 0;
        // Determine the volume of the initial shape
        switch (initShape)
        {
            case InitShape.Cube:
                volume = cube_size * cube_size * cube_size;
                break;
            case InitShape.Sphere:
                volume = 4.0f / 3.0f * Mathf.PI * Mathf.Pow(cube_size / 2, 3);
                break;
            case InitShape.Cylinder:
                volume = Mathf.PI * Mathf.Pow(cylinder_radius, 2) * cylinder_height;
                break;
            case InitShape.Torus:
                volume = 2 * Mathf.PI * Mathf.PI * Mathf.Pow(torus_tube_radius, 2) * torus_radius;
                break;
        }
        // Determine the number of particles based on the grid size, particle density, and volume
        NParticles = (int)(n_grid * n_grid * n_grid * particle_per_grid * volume);
        UnityEngine.Debug.Log("Number of particles: " + NParticles);
        x = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3).Build();
        v = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3).Build();
        C = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3, 3).Build();
        dg = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3, 3).Build();

        // kernel initialization of different primitive shapes
        if (initShape == InitShape.Cube)
            if (_Compute_Graph_g_init != null)
            {
                _Compute_Graph_g_init.LaunchAsync(new Dictionary<string, object>
                {
                    { "x", x },
                    { "v", v }
                });
            }
            else
            {
                _Kernel_init_particles.LaunchAsync(x, v, dg, cube_size);
            }
        else if (initShape == InitShape.Sphere)
            _Kernel_init_sphere.LaunchAsync(x, dg, cube_size / 2);
        else if (initShape == InitShape.Cylinder)
            _Kernel_init_cylinder.LaunchAsync(x, dg, cylinder_height, cylinder_radius);
        else if (initShape == InitShape.Torus)
            _Kernel_init_torus.LaunchAsync(x, dg, torus_radius, torus_tube_radius);
    }

    public void Init_MarchingCubes()
    {
        _p_vol = dx * dx * dx / particle_per_grid;
        _p_mass = _p_vol * p_rho;
        max_density = particle_per_grid * _p_mass;

        marchingCubeVisualizers[0]._dimensions = new Vector3Int(n_grid, n_grid, n_grid);
        marchingCubeVisualizers[0]._gridScale = dx;

        marchingCubeVisualizers[0].Init();
    }

    public void Init_gaussian()
    {
        NParticles = splatManager.splatsNum;
        x = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3).HostWrite(true).Build();
        v = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3).Build();
        C = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3, 3).Build();
        dg = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3, 3).Build();

        //gaussian
        init_rotation = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(4).Build();
        init_scale = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3).Build();
        other_data = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(4).HostWrite(true).Build();
        init_sh = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(16, 3).HostWrite(true).Build();
        sh = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(16, 3).Build();
        x.CopyFromArray(splatManager.m_pos);
        other_data.CopyFromArray(splatManager.m_other);
        init_sh.CopyFromArray(splatManager.m_SH);

        if (_Compute_Graph_g_init != null)
        {
            _Compute_Graph_g_init.LaunchAsync(new Dictionary<string, object>
            {
                { "x", x },
                {"dg", dg},
                {"other_data", other_data},
                {"init_sh", init_sh},
                {"init_rotation", init_rotation},
                {"init_scale", init_scale},
                {"eps", bounding_eps},

            });
        }
        else
        {

            _Kernel_init_dg.LaunchAsync(dg);
            _Kernel_scale_to_unit_cube.LaunchAsync(x, other_data, bounding_eps);
            _Kernel_init_gaussian_data.LaunchAsync(init_rotation, init_scale, other_data);

        }
    }
    public void Init_materials()
    {
        _p_vol = dx * dx * dx / particle_per_grid;
        _p_mass = _p_vol * p_rho;
        max_density = particle_per_grid * _p_mass;
        sin_phi = Mathf.Sin(friction_angle * Mathf.Deg2Rad);
        _alpha = Mathf.Sqrt(2.0f / 3.0f) * 2 * sin_phi / (3 - sin_phi);

        E_host = new float[NParticles];
        SigY_host = new float[NParticles];
        nu_host = new float[NParticles];
        min_clamp_host = new float[NParticles];
        max_clamp_host = new float[NParticles];
        alpha_host = new float[NParticles];
        p_vol_host = new float[NParticles];
        p_mass_host = new float[NParticles];

        if (marchingCubeVisualizers.Length == 1)
            point_color_host = new int[NParticles];

        material_host = new int[NParticles];

        for (int i = 0; i < NParticles; i++)
        {
            E_host[i] = _E;
            SigY_host[i] = _SigY;
            nu_host[i] = _nu;
            min_clamp_host[i] = _min_clamp;
            max_clamp_host[i] = _max_clamp;
            alpha_host[i] = _alpha;
            p_vol_host[i] = _p_vol;
            p_mass_host[i] = _p_mass;
            material_host[i] = 0;
            if (marchingCubeVisualizers.Length == 1)
                point_color_host[i] = 0;
            switch (plasticityType)
            {
                case PlasticityType.Von_Mises:
                    material_host[i] |= 1 << 16;
                    break;
                case PlasticityType.Drucker_Prager:
                    material_host[i] |= 3 << 16;
                    break;
                case PlasticityType.Clamp:
                    material_host[i] = 2 << 16;
                    break;
                case PlasticityType.Elastic:
                    break;
            }
            switch (stressType)
            {
                case StressType.NeoHookean:
                    break;
                case StressType.Kirchhoff:
                    material_host[i] |= 1;
                    break;
            }
        }
    }
    public void Update_materials()
    {
        //materials
        E = new NdArrayBuilder<float>().Shape(NParticles).HostWrite(true).Build();
        SigY = new NdArrayBuilder<float>().Shape(NParticles).HostWrite(true).Build();
        nu = new NdArrayBuilder<float>().Shape(NParticles).HostWrite(true).Build();
        min_clamp = new NdArrayBuilder<float>().Shape(NParticles).HostWrite(true).Build();
        max_clamp = new NdArrayBuilder<float>().Shape(NParticles).HostWrite(true).Build();
        alpha = new NdArrayBuilder<float>().Shape(NParticles).HostWrite(true).Build();
        p_vol = new NdArrayBuilder<float>().Shape(NParticles).HostWrite(true).Build();
        p_mass = new NdArrayBuilder<float>().Shape(NParticles).HostWrite(true).Build();

        material = new NdArrayBuilder<int>().Shape(NParticles).HostWrite(true).Build();
        point_color = new NdArrayBuilder<int>().Shape(NParticles).HostWrite(true).Build();

        E.CopyFromArray(E_host);
        SigY.CopyFromArray(SigY_host);
        nu.CopyFromArray(nu_host);
        min_clamp.CopyFromArray(min_clamp_host);
        max_clamp.CopyFromArray(max_clamp_host);
        alpha.CopyFromArray(alpha_host);
        p_vol.CopyFromArray(p_vol_host);
        p_mass.CopyFromArray(p_mass_host);
        material.CopyFromArray(material_host);
        point_color.CopyFromArray(point_color_host);
    }

    public void CopyMaterials(Mpm3DMarching other)
    {
        other.E_host = new float[NParticles];
        other.SigY_host = new float[NParticles];
        other.nu_host = new float[NParticles];
        other.min_clamp_host = new float[NParticles];
        other.max_clamp_host = new float[NParticles];
        other.alpha_host = new float[NParticles];
        other.p_vol_host = new float[NParticles];
        other.p_mass_host = new float[NParticles];
        other.material_host = new int[NParticles];
        other.point_color_host = new int[NParticles];

        E_host.CopyTo(other.E_host, 0);
        SigY_host.CopyTo(other.SigY_host, 0);
        nu_host.CopyTo(other.nu_host, 0);
        min_clamp_host.CopyTo(other.min_clamp_host, 0);
        max_clamp_host.CopyTo(other.max_clamp_host, 0);
        alpha_host.CopyTo(other.alpha_host, 0);
        p_vol_host.CopyTo(other.p_vol_host, 0);
        p_mass_host.CopyTo(other.p_mass_host, 0);
        material_host.CopyTo(other.material_host, 0);
        point_color_host.CopyTo(other.point_color_host, 0);

    }

    // Update is called once per frame
    void Update()
    {
        if (!RunSimulation || _grabbable.SelectingPointsCount > 0)
        {
            if (!updated)
            {
                if (renderType == RenderType.GaussianSplat)
                {
                    other_data.CopyToNativeBufferAsync(splatManager.m_Render.m_GpuOtherData.GetNativeBufferPtr());
                    x.CopyToNativeBufferAsync(splatManager.m_Render.m_GpuPosData.GetNativeBufferPtr());
                }
                updated = true;
                Runtime.Submit();
            }
            return;
        }
        UpdateGravity();
        if (lastRenderType != renderType)
        {
            switch (lastRenderType)
            {
                case RenderType.PointMesh:
                    GetComponent<MeshRenderer>().enabled = false;
                    break;
                case RenderType.GaussianSplat:
                    GetComponent<GaussianSplatRenderer>().enabled = false;
                    break;
                case RenderType.MarchingCubes:
                    Transform[] allChildren = gameObject.GetComponentsInChildren<Transform>(true);
                    string childName = "MarchingCubeVisualizer";
                    foreach (Transform child in allChildren)
                    {
                        if (child.name == childName)
                        {
                            child.gameObject.SetActive(false);
                        }
                    }
                    break;
            }
            switch (renderType)
            {
                case RenderType.PointMesh:
                    GetComponent<MeshRenderer>().enabled = true;
                    break;
                case RenderType.GaussianSplat:
                    GetComponent<GaussianSplatRenderer>().enabled = true;
                    break;
                case RenderType.MarchingCubes:
                    Transform[] allChildren = gameObject.GetComponentsInChildren<Transform>(true);
                    string childName = "MarchingCubeVisualizer";
                    foreach (Transform child in allChildren)
                    {
                        if (child.name == childName)
                        {
                            child.gameObject.SetActive(true);
                        }
                    }
                    break;
            }
            lastRenderType = renderType;
        }
        if (_Compute_Graph_g_substep != null)
        {
            UpdateHandSDF();
            _Compute_Graph_g_substep.LaunchAsync(new Dictionary<string, object>
            {
                {"v", v},
                { "grid_m",grid_m},
                { "x",x},
                { "C",C},
                { "grid_v",grid_v},
                {"init_sh",init_sh},
                {"sh",sh},
                {"other_data",other_data},
                { "dg",dg},
                {"init_scale",init_scale},
                {"init_rotation",init_rotation},
                { "mu_0",mu},
                { "lambda_0",lambda},
                { "p_vol",_p_vol},
                { "p_mass",_p_mass},
                { "dx",dx},
                { "dt",max_dt},
                { "n_grid",n_grid},
                {"gx",g.x},
                {"gy",g.y},
                {"gz",g.z},
                { "k",colide_factor},
                { "damping",damping},
                { "friction_k",friction_k},
                { "v_allowed",v_allowed},
                { "min_clamp",_min_clamp},
                { "max_clamp",_max_clamp},
                {"hand_sdf",hand_sdf},
                {"skeleton_segments",skeleton_segments},
                {"skeleton_velocities",skeleton_velocities},
                {"skeleton_capsule_radius",skeleton_capsule_radius},
                {"obstacle_normals",obstacle_normals},
                {"obstacle_velocities",obstacle_velocities},
                {"bound",bound},
                {"min_x",boundary_min[0]},
                {"max_x",boundary_max[0]},
                {"min_y",boundary_min[1]},
                {"max_y",boundary_max[1]},
                {"min_z",boundary_min[2]},
                {"max_z",boundary_max[2]},
            });
        }
        else
        {
            // Simulation
            float dt = max_dt, time_left = frame_time;
            switch (obstacleType)
            {
                case ObstacleType.Sphere:
                    UpdateSphereSDF();
                    if (Intersectwith(sphere))
                    {
                        _Kernel_substep_calculate_signed_distance_field.LaunchAsync(sphere_pos, hand_sdf, obstacle_normals, sphere_radius, dx, dt, boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
                    }
                    break;
                case ObstacleType.Hand:
                    if (UseRecordDate)
                    {
                        UpdateHandSDFFromRecordedData(handMotionIndex++);
                        RenderRecordedHandSkeletonCapsule(handMotionIndex - 1);
                    }
                    else
                    {
                        UpdateHandSDF();
                    }
                    if (IntersectwithHand(oculus_hands))
                    {
                        if (transform.lossyScale.x > 1.0f)
                        {
                            // The object is scaled up
                            _Kernel_substep_calculate_hand_hash.LaunchAsync(skeleton_segments, skeleton_capsule_radius, n_grid, hash_table, segments_count_per_cell);
                            _Kernel_substep_calculate_hand_sdf_hash.LaunchAsync(skeleton_segments, skeleton_velocities, hand_sdf, obstacle_normals, obstacle_velocities, skeleton_capsule_radius, dx, hash_table, segments_count_per_cell,
                            boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
                        }
                        else
                        {
                            // The object is scaled down
                            _Kernel_substep_calculate_hand_sdf.LaunchAsync(skeleton_segments, skeleton_velocities, hand_sdf, obstacle_normals, obstacle_velocities, skeleton_capsule_radius, dx,
                            boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
                        }
                    }
                    break;
            }
            while (time_left > 0)
            {
                time_left -= dt;

                _Kernel_subsetep_reset_grid.LaunchAsync(grid_v, grid_m, marching_m, boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
                //_Kernel_substep_p2g.LaunchAsync(x, v, C, dg, grid_v, grid_m, E, nu, material, p_vol, p_mass, dx, dt, boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
                _Kernel_substep_p2g_multi.LaunchAsync(x, v, C, dg, grid_v, grid_m, point_color, marching_m, E, nu, material, p_vol, p_mass, dx, dt, boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
                _Kernel_substep_update_grid_v.LaunchAsync(grid_v, grid_m, hand_sdf, obstacle_normals, obstacle_velocities, g.x, g.y, g.z, colide_factor, damping, friction_k, v_allowed, dt, n_grid, dx, bound, use_sticky_boundary, boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);

                // If fix the object in place during the modeling process
                if (is_fixed)
                    _Kernel_substep_fix_object.LaunchAsync(grid_v, fix_center.x, fix_center.y, fix_center.z, fix_radius);

                _Kernel_substep_g2p.LaunchAsync(x, v, C, grid_v, dx, dt, boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
                _Kernel_substep_apply_plasticity.LaunchAsync(dg, x, E, nu, material, SigY, alpha, min_clamp, max_clamp, boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);

                if (use_correct_cfl)
                {
                    v_allowed = float.MaxValue;
                    // Taichi Allocate memory, hostwrite are not considered
                    _Kernel_substep_get_max_speed.LaunchAsync(v, x, max_v, boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
                    float[] max_speed = new float[1];
                    max_v.CopyToArray(max_speed);
                    dt = Mathf.Min(max_dt, dx * allowed_cfl / max_speed[0]);
                    dt = Mathf.Min(dt, time_left);
                }
            }
            if (transform.lossyScale.x > 1.0f)
            {
                _Kernel_substep_adjust_particle.LaunchAsync(x, v, hash_table, segments_count_per_cell, skeleton_capsule_radius, skeleton_velocities, skeleton_segments, boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
            }
        }

        if (renderType == RenderType.PointMesh)
        {
            x.CopyToNativeBufferAsync(_Mesh.GetNativeVertexBufferPtr(0));
        }
        else if (renderType == RenderType.Raymarching)
        {
            grid_m.CopyToNativeBufferAsync(volumeTextureUpdater.computeBuffer.GetNativeBufferPtr());
        }
        else if (renderType == RenderType.GaussianSplat)
        {
            _Kernel_substep_update_gaussian_data.LaunchAsync(init_rotation, init_scale, dg, other_data, init_sh, sh, x,
                                                             boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
            other_data.CopyToNativeBufferAsync(splatManager.m_Render.m_GpuOtherData.GetNativeBufferPtr());
            sh.CopyToNativeBufferAsync(splatManager.m_Render.m_GpuSHData.GetNativeBufferPtr());
            x.CopyToNativeBufferAsync(splatManager.m_Render.m_GpuPosData.GetNativeBufferPtr());
        }
        else if (renderType == RenderType.MarchingCubes)
        {
            _Kernel_normalize_m.LaunchAsync(marching_m, max_density);
            marching_m.CopyToNativeBufferAsync(marching_m_computeBuffer.GetNativeBufferPtr());
            int kernelId = copyShader.FindKernel("CopySubBuffer");
            copyShader.SetBuffer(kernelId, "sourceBuffer", marching_m_computeBuffer);
            int num = marchingCubeVisualizers[0]._voxelBuffer.count;
            for (int i = 0; i < marchingCubeVisualizers.Length; i++)
            {
                copyShader.SetInt("sourceOffset", i * num);
                copyShader.SetBuffer(kernelId, "destinationBuffer", marchingCubeVisualizers[i]._voxelBuffer);
                int threadGroupSize = 1024; // 1024 threads per group
                int threadGroups = Mathf.CeilToInt((float)num / threadGroupSize);
                copyShader.Dispatch(kernelId, threadGroups, 1, 1);
                marchingCubeVisualizers[i].shouldUpdate = true;
            }
        }
        Runtime.Submit();
    }

    public void MergeAndUpdate(Mpm3DMarching other)
    {
        if (other.renderType != renderType)
        {
            UnityEngine.Debug.LogError("Cannot merge different render types.");
            return;
        }
        if (renderType == RenderType.GaussianSplat)
        {
            MergeGaussianRenders(other.splatManager.m_Render);
        }
        else
        {
            MergeParticles(other);
            if (renderType == RenderType.MarchingCubes)
            {
                MergeMarchingCubes(other);
            }
        }
        MergeMaterials(other);
        other.gameObject.SetActive(false);
    }
    private void MergeGaussianRenders(GaussianSplatRenderer otherRender)
    {
        if (otherRender == null)
        {
            UnityEngine.Debug.LogError("Other render is null.");
            return;
        }
        var render = splatManager.m_Render;
        int totalSplats = render.splatCount + otherRender.splatCount;
        if (totalSplats > GaussianSplatAsset.kMaxSplats)
        {
            UnityEngine.Debug.LogWarning("Cannot merge, too many splats.");
            return;
        }

        int copyDstOffset = render.splatCount;
        render.EditSetSplatCount(totalSplats);
        otherRender.EditCopySplatsInto(render, 0, copyDstOffset, otherRender.splatCount);
        splatManager.init_gaussians();
        Init_gaussian();
    }
    private void MergeMarchingCubes(Mpm3DMarching other)
    {
        Transform[] allChildren = other.gameObject.GetComponentsInChildren<Transform>(true);
        string childName = "MarchingCubeVisualizer";
        for (int i = 0; i < other.NParticles; i++)
        {
            other.point_color_host[i] += marchingCubeVisualizers.Length;
        }
        foreach (Transform child in allChildren)
        {
            if (child.name == childName)
            {
                if (child.TryGetComponent<MarchingCubeVisualizer>(out var m))
                {
                    m._dimensions = new Vector3Int(n_grid, n_grid, n_grid);
                    m._gridScale = dx;
                    m.Init();
                    marchingCubeVisualizers = marchingCubeVisualizers.Concat(new MarchingCubeVisualizer[] { m }).ToArray();
                    child.SetParent(transform, false);
                }
            }
        }
        marching_m = new NdArrayBuilder<float>().Shape(marchingCubeVisualizers.Length, n_grid, n_grid, n_grid).Build();
        marching_m_computeBuffer = new ComputeBuffer(n_grid * n_grid * n_grid * marchingCubeVisualizers.Length, sizeof(float));
    }

    private void MergeParticles(Mpm3DMarching other)
    {
        int totalParticles = NParticles + other.NParticles;
        Matrix4x4 transform1 = transform.localToWorldMatrix;
        Matrix4x4 transform2 = other.transform.localToWorldMatrix;
        NdArray<float> new_x = new NdArrayBuilder<float>().Shape(totalParticles).ElemShape(3).Build();
        NdArray<float> t1 = new NdArrayBuilder<float>().Shape(4, 4).HostWrite(true).Build();
        NdArray<float> t2 = new NdArrayBuilder<float>().Shape(4, 4).HostWrite(true).Build();
        t1.CopyFromArray(MatrixtoFloatArray(transform1));
        t2.CopyFromArray(MatrixtoFloatArray(transform2));
        _Kernel_transform_and_merge.LaunchAsync(new_x, x, other.x, t1, t2);

        NParticles = totalParticles;

        NdArray<float> _other_data = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(4).Build();
        _Kernel_scale_to_unit_cube.LaunchAsync(new_x, _other_data, bounding_eps);

        x = new_x;
        v = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3).Build();
        C = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3, 3).Build();
        dg = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3, 3).Build();

        _Kernel_init_dg.LaunchAsync(dg);
    }
    private void MergeMaterials(Mpm3DMarching other)
    {
        E_host = E_host.Concat(other.E_host).ToArray();
        SigY_host = SigY_host.Concat(other.SigY_host).ToArray();
        nu_host = nu_host.Concat(other.nu_host).ToArray();
        min_clamp_host = min_clamp_host.Concat(other.min_clamp_host).ToArray();
        max_clamp_host = max_clamp_host.Concat(other.max_clamp_host).ToArray();
        alpha_host = alpha_host.Concat(other.alpha_host).ToArray();
        p_vol_host = p_vol_host.Concat(other.p_vol_host).ToArray();
        p_mass_host = p_mass_host.Concat(other.p_mass_host).ToArray();
        material_host = material_host.Concat(other.material_host).ToArray();
        point_color_host = point_color_host.Concat(other.point_color_host).ToArray();

        E = new NdArrayBuilder<float>().Shape(NParticles).HostWrite(true).Build();
        SigY = new NdArrayBuilder<float>().Shape(NParticles).HostWrite(true).Build();
        nu = new NdArrayBuilder<float>().Shape(NParticles).HostWrite(true).Build();
        min_clamp = new NdArrayBuilder<float>().Shape(NParticles).HostWrite(true).Build();
        max_clamp = new NdArrayBuilder<float>().Shape(NParticles).HostWrite(true).Build();
        alpha = new NdArrayBuilder<float>().Shape(NParticles).HostWrite(true).Build();
        p_vol = new NdArrayBuilder<float>().Shape(NParticles).HostWrite(true).Build();
        p_mass = new NdArrayBuilder<float>().Shape(NParticles).HostWrite(true).Build();
        material = new NdArrayBuilder<int>().Shape(NParticles).HostWrite(true).Build();
        point_color = new NdArrayBuilder<int>().Shape(NParticles).HostWrite(true).Build();

        Update_materials();
    }
    public void MergeGrabbable(GameObject object2)
    {
        object2.transform.SetParent(gameObject.transform);
    }
    public void FixObject(Vector3 center, float radius)
    {
        is_fixed = true;
        fix_center = center;
        fix_radius = radius;
    }
    public void SetFixed(bool fixed_)
    {
        is_fixed = fixed_;
    }
    public bool GetIsFixed()
    {
        return is_fixed;
    }
    public void SetStickyBoundary(bool sticky)
    {
        use_sticky_boundary = (sticky ? 1 : 0);
    }
    public bool GetIsStickyBoundary()
    {
        return use_sticky_boundary == 1;
    }
    public void SetHandsimulationRadius(float radius)
    {
        hand_simulation_radius = radius;
    }
    public float GetHandsimulationRadius()
    {
        return hand_simulation_radius;
    }
    public void InitGrid()
    {
        grid_v = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).ElemShape(3).Build();
        grid_m = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).Build();
        hand_sdf = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).Build();
        obstacle_velocities = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).ElemShape(3).Build();
        obstacle_normals = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).ElemShape(3).Build();
        segments_count_per_cell = new NdArrayBuilder<int>().Shape(n_grid, n_grid, n_grid).Build();
        hash_table = new NdArrayBuilder<int>().Shape(n_grid, n_grid, n_grid, skeleton_num_capsules * oculus_skeletons.Length).Build();

        marching_m = new NdArrayBuilder<float>().Shape(marchingCubeVisualizers.Length, n_grid, n_grid, n_grid).Build();

        marching_m_computeBuffer = new ComputeBuffer(n_grid * n_grid * n_grid * marchingCubeVisualizers.Length, sizeof(float));
    }
    public void DiposeGrid()
    {
        grid_v.Dispose();
        grid_m.Dispose();
        hand_sdf.Dispose();
        obstacle_velocities.Dispose();
        obstacle_normals.Dispose();
        segments_count_per_cell.Dispose();
        hash_table.Dispose();
    }
    public void SetGridSize(int n)
    {
        particle_per_grid = particle_per_grid * (n_grid * n_grid * n_grid) / (n * n * n);
        max_density = particle_per_grid * _p_mass;
        n_grid = n;
        dx = 1.0f / n_grid;

        InitGrid();

        if (renderType == RenderType.MarchingCubes)
        {
            for (int i = 0; i < marchingCubeVisualizers.Length; i++)
            {
                marchingCubeVisualizers[i]._dimensions = new Vector3Int(n_grid, n_grid, n_grid);
                marchingCubeVisualizers[i]._gridScale = dx;
                marchingCubeVisualizers[i].Init();
            }
        }
    }
    public int GetGridSize()
    {
        return n_grid;
    }
    public void IncreaseGridSize(int num)
    {
        if (n_grid + num >= 150)
        {
            UnityEngine.Debug.LogWarning("Cannot increase grid size anymore.");
            SetGridSize(150);
            return;
        }
        SetGridSize(n_grid + num);
    }
    public void DecreaseGridSize(int num)
    {
        if (n_grid - num <= 50)
        {
            UnityEngine.Debug.LogWarning("Cannot decrease grid size anymore.");
            SetGridSize(50);
            return;
        }
        SetGridSize(n_grid - num);
    }
    public void AdjustTextureColor(Color rgba)
    {
        if (renderType == RenderType.GaussianSplat)
        {
            AdjustGaussianTextureColor(rgba);
        }
        else if (renderType == RenderType.MarchingCubes)
        {
            AdjustMarchingCubeTextureColor(rgba);
        }
    }
    public void AdjustTextureColorRed(float r)
    {
        if (renderType == RenderType.GaussianSplat)
        {
            AdjustGaussianTextureColorRed(r);
        }
        else if (renderType == RenderType.MarchingCubes)
        {
            AdjustMarchingCubeTextureColorRed(r);
        }
    }
    void AdjustGaussianTextureColor(Color rgba)
    {
        var colorData = splatManager.m_color;
        var asset = splatManager.m_Render.m_Asset;
        // Adjust the color in the NativeArray
        for (int i = 0; i < colorData.Length; i += 4)
        {
            colorData[i] *= rgba.r;     // Red channel
            colorData[i + 1] *= rgba.g; // Green channel
            colorData[i + 2] *= rgba.b; // Blue channel
            colorData[i + 3] *= rgba.a; // Alpha channel

        }
        // Set the modified color data back to the texture
        var (texWidth, texHeight) = GaussianSplatAsset.CalcTextureSize(asset.splatCount);
        var texFormat = GaussianSplatAsset.ColorFormatToGraphics(asset.colorFormat);
        var tex = new Texture2D(texWidth, texHeight, texFormat, TextureCreationFlags.DontInitializePixels | TextureCreationFlags.IgnoreMipmapLimit | TextureCreationFlags.DontUploadUponCreate) { name = "GaussianColorData" };
        tex.SetPixelData(colorData, 0);
        tex.Apply(false, true);
        splatManager.m_Render.m_GpuColorData = tex;

    }
    void AdjustGaussianTextureColorRed(float r)
    {
        var colorData = splatManager.m_color;
        var asset = splatManager.m_Render.m_Asset;
        // Adjust the color in the NativeArray
        for (int i = 0; i < colorData.Length; i += 4)
        {
            colorData[i] *= r;     // Red channel

        }
        // Set the modified color data back to the texture
        var (texWidth, texHeight) = GaussianSplatAsset.CalcTextureSize(asset.splatCount);
        var texFormat = GaussianSplatAsset.ColorFormatToGraphics(asset.colorFormat);
        var tex = new Texture2D(texWidth, texHeight, texFormat, TextureCreationFlags.DontInitializePixels | TextureCreationFlags.IgnoreMipmapLimit | TextureCreationFlags.DontUploadUponCreate) { name = "GaussianColorData" };
        tex.SetPixelData(colorData, 0);
        tex.Apply(false, true);
        splatManager.m_Render.m_GpuColorData = tex;
    }
    void AdjustMarchingCubeTextureColor(Color rgba, int index = 0)
    {
        marchingCubeVisualizers[index].GetComponent<MeshRenderer>().material.color = rgba;
    }
    void AdjustMarchingCubeTextureColorRed(float r, int index = 0)
    {
        marchingCubeVisualizers[index].GetComponent<MeshRenderer>().material.color = new Color(r, 0, 0, 1);
    }
    public void CopyObjectTo(Mpm3DMarching other)
    {
        if (renderType == RenderType.GaussianSplat)
        {
            // CopyFromGaussianSplat(other);
        }
        else
        {
            UnityEngine.Debug.Log("COPY Not implemented yet.");

            other.Initiate();

            other.NParticles = NParticles;

            other.x = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3).HostWrite(true).Build();
            other.v = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3).Build();
            other.C = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3, 3).Build();
            other.dg = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3, 3).Build();

            _Kernel_copy_array_1dim3.LaunchAsync(x, other.x);
            _Kernel_init_dg.LaunchAsync(other.dg);

            CopyMaterials(other);

            other.Update_materials();

            other.SetGridSize(other.n_grid);
        }
    }
    public void Reset()
    {
        if (renderType == RenderType.GaussianSplat)
        {
            Init_gaussian();
        }
        else
        {
            Init_Particles();
            Init_MarchingCubes();
            Init_materials();
            Update_materials();
        }
    }
    public void SetGravity(float y)
    {
        gy = y;
    }
    private void UpdateGravity()
    {
        g = transform.InverseTransformDirection(new Vector3(0, gy, 0));
    }
    public float GetGravity()
    {
        return gy;
    }

    void UpdateSphereSDF()
    {
        for (int i = 0; i < sphere.Length; i++)
        {
            Vector3 curpos = sphere[i].Position;
            Vector3 velocity = sphere[i].Velocity;
            sphere_positions[i * 3] = curpos.x - transform.position.x;
            sphere_positions[i * 3 + 1] = curpos.y - transform.position.y;
            sphere_positions[i * 3 + 2] = curpos.z - transform.position.z;
            _sphere_velocities[i * 3 + 1] = velocity.y;
            _sphere_velocities[i * 3 + 2] = velocity.z;
            sphere_radii[i] = sphere[i].Radius;
        }
        sphere_pos.CopyFromArray(sphere_positions);
        sphere_velocities.CopyFromArray(_sphere_velocities);
        sphere_radius.CopyFromArray(sphere_radii);
    }

    void UpdateHandSDF()
    {
        Vector3 Center = new();
        for (int i = 0; i < oculus_skeletons.Length; i++)
        {
            if (oculus_hands[i].IsTracked && oculus_hands[i].HandConfidence == OVRHand.TrackingConfidence.High)
            {
                int numBones = oculus_skeletons[i].Bones.Count();
                //UnityEngine.Debug.Log("Num of Bones while tracking: " + numBones);
                if (numBones > 0)
                {
                    // Use the wrist position as the hand position
                    Center += oculus_skeletons[i].Bones[0].Transform.position;

                    int init = i * skeleton_num_capsules * 6;
                    for (int j = 0; j < numBones; j++)
                    {
                        var bone = oculus_skeletons[i].Bones[j];
                        Vector3 start = bone.Transform.position;
                        Vector3 end = bone.Transform.parent.position;
                        if (isRecording)
                        {
                            handPositions.Add(new float[] { start.x, start.y, start.z, end.x, end.y, end.z });
                        }

                        // World to local coordinate conversion and calculate the velocity of the segment
                        UpdateHandSkeletonSegment(init + j * 6, start, end, frame_time);

                        // Get the radius of each capsule
                        _skeleton_capsule_radius[i * skeleton_num_capsules + j] = preset_capsule_radius[j] / transform.lossyScale.x;
                    }

                    // Copy the hand skeleton segments and velocities to the compute buffer
                    skeleton_segments.CopyFromArray(hand_skeleton_segments);
                    skeleton_velocities.CopyFromArray(hand_skeleton_velocities);
                    skeleton_capsule_radius.CopyFromArray(_skeleton_capsule_radius);
                }
            }
        }

        // Update the simulation box domain around the two hands based on the position of them
        Center /= oculus_skeletons.Length;
        boundary_min = transform.InverseTransformPoint(Center) - Vector3.one * hand_simulation_radius / transform.lossyScale.x;
        boundary_max = transform.InverseTransformPoint(Center) + Vector3.one * hand_simulation_radius / transform.lossyScale.x;
    }

    void SaveHandMotionData()
    {
        using StreamWriter writer = new(filePath);
        foreach (var position in handPositions)
        {
            writer.WriteLine(string.Join(",", position));
        }
    }

    void LoadHandMotionData()
    {
        handPositions.Clear();
        using StreamReader reader = new(filePath);

        string line;
        while ((line = reader.ReadLine()) != null)
        {
            float[] position = Array.ConvertAll(line.Split(','), float.Parse);
            handPositions.Add(position);
        }
    }

    void UpdateHandSDFFromRecordedData(int index)
    {
        index %= handPositions.Count / (skeleton_num_capsules * oculus_skeletons.Length);
        if (handPositions.Count == 0)
        {
            UnityEngine.Debug.LogWarning("No recorded hand positions available.");
            return;
        }

        // Iterate through the recorded hand positions and apply them to the hand skeleton segments
        for (int i = 0; i < oculus_skeletons.Length; i++)
        {
            int init = i * skeleton_num_capsules * 6;

            for (int j = 0; j < skeleton_num_capsules; j++)
            {
                int idx = index * skeleton_num_capsules * oculus_skeletons.Length + i * skeleton_num_capsules + j;
                if (idx >= handPositions.Count)
                {
                    return;
                }
                float[] position = handPositions[idx];
                UpdateHandSkeletonSegment(init + j * 6, new Vector3(position[0], position[1], position[2]), new Vector3(position[3], position[4], position[5]), frame_time);
            }
        }

        skeleton_segments.CopyFromArray(hand_skeleton_segments);
        skeleton_velocities.CopyFromArray(hand_skeleton_velocities);
    }

    private void RenderRecordedHandSkeletonCapsule(int index)
    {
        // initialization
        if (!RendererInitialized)
        {
            for (int i = 0; i < oculus_skeletons.Length; i++)
            {
                for (int j = 0; j < skeleton_num_capsules; j++)
                {
                    var capsuleVis = new CapsuleVisualization(preset_capsule_radius[j], handMaterial);
                    _capsuleVisualizations.Add(capsuleVis);
                }
            }
            RendererInitialized = true;
        }

        if (RendererInitialized)
        {
            index %= handPositions.Count / (skeleton_num_capsules * oculus_skeletons.Length);
            for (int i = 0; i < oculus_skeletons.Length; i++)
            {
                int init = i * skeleton_num_capsules * 6;

                for (int j = 0; j < skeleton_num_capsules; j++)
                {
                    int idx = index * skeleton_num_capsules * oculus_skeletons.Length + i * skeleton_num_capsules + j;
                    if (idx >= handPositions.Count)
                    {
                        return;
                    }
                    float[] position = handPositions[idx];
                    _capsuleVisualizations[i * skeleton_num_capsules + j].Update(new Vector3(position[0], position[1], position[2]), new Vector3(position[3], position[4], position[5]));
                }
            }
        }
    }

    private void UpdateHandSkeletonSegment(int init, Vector3 segment_start, Vector3 segment_end, float frameTime)
    {
        // Convert the segment start and segment end points to local coordinates relative to this transform
        Vector3 TransformedStart = transform.InverseTransformPoint(segment_start);
        Vector3 TransformedEnd = transform.InverseTransformPoint(segment_end);

        hand_skeleton_segments[init] = TransformedStart.x;
        hand_skeleton_segments[init + 1] = TransformedStart.y;
        hand_skeleton_segments[init + 2] = TransformedStart.z;
        hand_skeleton_segments[init + 3] = TransformedEnd.x;
        hand_skeleton_segments[init + 4] = TransformedEnd.y;
        hand_skeleton_segments[init + 5] = TransformedEnd.z;

        // Calculate the velocity of the segment
        hand_skeleton_velocities[init] = (TransformedStart.x - hand_skeleton_segments_prev[init]) / frameTime;
        hand_skeleton_velocities[init + 1] = (TransformedStart.y - hand_skeleton_segments_prev[init + 1]) / frameTime;
        hand_skeleton_velocities[init + 2] = (TransformedStart.z - hand_skeleton_segments_prev[init + 2]) / frameTime;
        hand_skeleton_velocities[init + 3] = (TransformedEnd.x - hand_skeleton_segments_prev[init + 3]) / frameTime;
        hand_skeleton_velocities[init + 4] = (TransformedEnd.y - hand_skeleton_segments_prev[init + 4]) / frameTime;
        hand_skeleton_velocities[init + 5] = (TransformedEnd.z - hand_skeleton_segments_prev[init + 5]) / frameTime;

        // Update the previous frame positions
        hand_skeleton_segments_prev[init] = TransformedStart.x;
        hand_skeleton_segments_prev[init + 1] = TransformedStart.y;
        hand_skeleton_segments_prev[init + 2] = TransformedStart.z;
        hand_skeleton_segments_prev[init + 3] = TransformedEnd.x;
        hand_skeleton_segments_prev[init + 4] = TransformedEnd.y;
        hand_skeleton_segments_prev[init + 5] = TransformedEnd.z;
    }

    bool Intersectwith(Sphere[] o)
    {
        for (int i = 0; i < o.Length; i++)
        {
            Bounds b = new(o[i].transform.position, o[i].transform.lossyScale);

            if (b == null)
            {
                continue;
            }
            if (bounds.Intersects(b))
            {
                return true;
            }
        }
        return false;
    }

    bool IntersectwithHand(OVRHand[] hands)
    {
        return true;
    }

    void OnSpacePressed()
    {
        isRecording = !isRecording;
        //print("Recording: " + isRecording);
        if (!isRecording)
        {
            SaveHandMotionData();
        }
        else
        {
            handPositions.Clear();
        }
    }
    unsafe void PrintNativeTextureData(IntPtr ptr, int size)
    {

        byte* data = (byte*)ptr.ToPointer();
        StringBuilder sb = new();
        for (int i = 0; i < size; i++)
        {
            sb.AppendFormat("Byte {0}: {1:X2} ", i, data[i]);
        }
        UnityEngine.Debug.Log(sb.ToString());
    }
    private float[] MatrixtoFloatArray(Matrix4x4 matrix)
    {
        float[] array = new float[16];
        array[0] = matrix.m00;
        array[1] = matrix.m01;
        array[2] = matrix.m02;
        array[3] = matrix.m03;
        array[4] = matrix.m10;
        array[5] = matrix.m11;
        array[6] = matrix.m12;
        array[7] = matrix.m13;
        array[8] = matrix.m20;
        array[9] = matrix.m21;
        array[10] = matrix.m22;
        array[11] = matrix.m23;
        array[12] = matrix.m30;
        array[13] = matrix.m31;
        array[14] = matrix.m32;
        array[15] = matrix.m33;
        return array;
    }
}