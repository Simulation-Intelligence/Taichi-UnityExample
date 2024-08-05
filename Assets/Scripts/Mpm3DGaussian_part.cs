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
using static SkeletonRenderer;


public class Mpm3DGaussian_part : MonoBehaviour
{
    private Mesh _Mesh;
    private MeshFilter _MeshFilter;
    private MeshRenderer _MeshRenderer;

    public VolumeTextureUpdater volumeTextureUpdater;

    [Header("MpM Engine")]
    [SerializeField]
    private AotModuleAsset Mpm3DModule;
    private Kernel _Kernel_subsetep_reset_grid, _Kernel_substep_neohookean_p2g, _Kernel_substep_Kirchhoff_p2g,
    _Kernel_substep_calculate_signed_distance_field, _Kernel_substep_update_grid_v, _Kernel_substep_g2p,
     _Kernel_substep_apply_Von_Mises_plasticity, _Kernel_substep_apply_Drucker_Prager_plasticity,
     _Kernel_substep_apply_clamp_plasticity, _Kernel_substep_calculate_hand_sdf, _Kernel_substep_get_max_speed,
     _Kernel_init_dg, _Kernel_init_gaussian_data, _Kernel_substep_update_gaussian_data, _Kernel_scale_to_unit_cube;

    public enum RenderType
    {
        PointMesh,
        Raymarching,
        GaussianSplat
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
    private RenderType renderType = RenderType.PointMesh;
    [SerializeField]
    private Material handMaterial;
    [SerializeField]
    private Material pointMaterial;
    [SerializeField]
    private Material raymarchingMaterial;
    [SerializeField]
    private PlasticityType plasticityType = PlasticityType.Von_Mises;
    [SerializeField]
    private StressType stressType = StressType.NeoHookean;
    private Kernel _Kernel_init_particles;
    private NdArray<float> x, v, C, dg, grid_v, grid_m, obstacle_pos, obstacle_velocity, obstacle_radius, sdf;

    public NdArray<float> skeleton_segments, skeleton_velocities, hand_sdf, obstacle_norms, skeleton_capsule_radius, max_v;

    public NdArray<float> init_rotation, init_scale, init_sh, other_data, sh;

    private float[] hand_skeleton_segments, hand_skeleton_segments_prev, hand_skeleton_velocities, _skeleton_capsule_radius, sphere_positions, sphere_velocities, sphere_radii;

    private Bounds bounds;

    private ComputeGraph _Compute_Graph_g_init;
    private ComputeGraph _Compute_Graph_g_update;

    [Header("Scene Settings")]
    [SerializeField]
    bool RunSimulation = true;
    private bool updated = false;
    [SerializeField]
    GaussianSplatRenderManager splatManager;

    [SerializeField]
    private Vector3 g = new(0, -9.8f, 0);
    [SerializeField]
    private int n_grid = 32, bound = 3;
    [SerializeField]
    private float bounding_eps = 0.1f;
    [SerializeField]
    private float max_dt = 1e-4f, frame_time = 0.005f, cube_size = 0.2f, particle_per_grid = 8, allowed_cfl = 0.5f, damping = 1f;
    [SerializeField]
    bool use_correct_cfl = false;

    [SerializeField]
    private float hand_simulation_radius = 0.5f;

    private Vector3 boundary_min, boundary_max;


    [Header("Obstacle")]
    [SerializeField]
    private ObstacleType obstacleType = ObstacleType.Sphere;
    [SerializeField]
    private Sphere[] sphere;

    [SerializeField]
    private OVRHand[] oculus_hands;
    [SerializeField]
    private OVRSkeleton[] oculus_skeletons;
    [SerializeField]
    private float Skeleton_capsule_radius = 0.01f;
    private float[] preset_capsule_radius;
    private int skeleton_num_capsules = 24; // use default 24
    private int NParticles;
    private float dx, p_vol, p_mass, v_allowed;

    [Header("Scalars")]
    [SerializeField]
    private float E = 1e4f;
    [SerializeField]
    private float SigY = 1000, nu = 0.3f, colide_factor = 0.5f, friction_k = 0.4f, p_rho = 1000, min_clamp = 0.1f, max_clamp = 0.1f, friction_angle = 30;

    private float mu, lambda, sin_phi, alpha;

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

    private bool RendererInitialized = false;
    private List<CapsuleVisualization> _capsuleVisualizations = new();

    // Start is called before the first frame update
    void Start()
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

            // new added
            _Kernel_substep_calculate_hand_sdf = kernels["substep_calculate_hand_sdf"];
            _Kernel_substep_get_max_speed = kernels["substep_get_max_speed"];

            //gaussian
            _Kernel_init_gaussian_data = kernels["init_gaussian_data"];
            _Kernel_substep_update_gaussian_data = kernels["substep_update_gaussian_data"];
            _Kernel_scale_to_unit_cube = kernels["scale_to_unit_cube"];
        }

        var cgraphs = Mpm3DModule.GetAllComputeGrpahs().ToDictionary(x => x.Name);
        if (cgraphs.Count > 0)
        {
            _Compute_Graph_g_init = cgraphs["init"];
            _Compute_Graph_g_update = cgraphs["update"];
        }

        //initialize
        if (renderType == RenderType.GaussianSplat)
        {
            NParticles = splatManager.splatsNum;
        }
        else
        {
            NParticles = (int)(n_grid * n_grid * n_grid * cube_size * cube_size * cube_size * particle_per_grid);
        }
        dx = 1.0f / n_grid;
        p_vol = dx * dx * dx / particle_per_grid;
        p_mass = p_vol * p_rho;
        mu = E / (2 * (1 + nu));
        lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
        v_allowed = allowed_cfl * dx / max_dt;
        sin_phi = Mathf.Sin(friction_angle * Mathf.Deg2Rad);
        alpha = Mathf.Sqrt(2.0f / 3.0f) * 2 * sin_phi / (3 - sin_phi);


        _MeshRenderer = GetComponent<MeshRenderer>();
        _MeshFilter = GetComponent<MeshFilter>();

        if (renderType == RenderType.Raymarching)
        {
            _MeshRenderer.material = raymarchingMaterial;
            volumeTextureUpdater.width = n_grid;
            volumeTextureUpdater.height = n_grid;
            volumeTextureUpdater.depth = n_grid;
            volumeTextureUpdater.densityData = new float[n_grid * n_grid * n_grid];
            volumeTextureUpdater.volumeTex = new RenderTexture(n_grid, n_grid, 0, RenderTextureFormat.RFloat)
            {
                dimension = UnityEngine.Rendering.TextureDimension.Tex3D,
                volumeDepth = n_grid,
                enableRandomWrite = true,
                wrapMode = TextureWrapMode.Clamp
            };
            volumeTextureUpdater.volumeTex.Create();
            volumeTextureUpdater.computeBuffer = new ComputeBuffer(n_grid * n_grid * n_grid, sizeof(float));

            volumeTextureUpdater.max_density = particle_per_grid * p_mass;
            volumeTextureUpdater.targetMaterial = GetComponent<Renderer>().material;
        }


        // Taichi Allocate memory, hostwrite are not considered
        x = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3).HostWrite(true).Build();
        v = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3).Build();
        C = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3, 3).Build();
        dg = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3, 3).Build();
        grid_v = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).ElemShape(3).Build();
        grid_m = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).Build();
        sdf = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).Build();
        obstacle_pos = new NdArrayBuilder<float>().Shape(sphere.Length).ElemShape(3).HostWrite(true).Build();
        obstacle_velocity = new NdArrayBuilder<float>().Shape(sphere.Length).ElemShape(3).HostWrite(true).Build();
        obstacle_radius = new NdArrayBuilder<float>().Shape(sphere.Length).HostWrite(true).Build();
        max_v = new NdArrayBuilder<float>().Shape(1).HostRead(true).Build();

        sphere_positions = new float[3 * sphere.Length];
        sphere_velocities = new float[3 * sphere.Length];
        sphere_radii = new float[sphere.Length];

        // new added
        UnityEngine.Debug.Log("Num of bones at start: " + oculus_skeletons[0].Bones.Count());
        skeleton_segments = new NdArrayBuilder<float>().Shape(skeleton_num_capsules * oculus_skeletons.Length, 2).ElemShape(3).HostWrite(true).Build(); // 24 skeleton segments, each segment has 6 floats
        skeleton_velocities = new NdArrayBuilder<float>().Shape(skeleton_num_capsules * oculus_skeletons.Length, 2).ElemShape(3).HostWrite(true).Build(); // 24 skeleton velocities, each velocity has 6 floats
        hand_sdf = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).Build();
        skeleton_capsule_radius = new NdArrayBuilder<float>().Shape(skeleton_num_capsules * oculus_skeletons.Length).HostWrite(true).Build(); // use a consistent radius for all capsules (at now)
        obstacle_norms = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).ElemShape(3).Build();

        //gaussian
        init_rotation = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(4).Build();
        init_scale = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3).Build();
        other_data = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(4).HostWrite(true).Build();
        init_sh = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(16, 3).HostWrite(true).Build();
        sh = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(16, 3).Build();

        //skeleton_num_capsules = oculus_skeletons[0].Bones.Count();
        hand_skeleton_segments = new float[skeleton_num_capsules * oculus_skeletons.Length * 6];
        hand_skeleton_segments_prev = new float[skeleton_num_capsules * oculus_skeletons.Length * 6];
        hand_skeleton_velocities = new float[skeleton_num_capsules * oculus_skeletons.Length * 6];
        _skeleton_capsule_radius = new float[skeleton_num_capsules * oculus_skeletons.Length];
        
        // for (int i = 0; i < skeleton_num_capsules * oculus_skeletons.Length; i++)
        // {
        //     _skeleton_capsule_radius[i] = Skeleton_capsule_radius / scale.x;
        // }
        // skeleton_capsule_radius.CopyFromArray(_skeleton_capsule_radius);

        // 24 line segments with 24 capsules in total
        preset_capsule_radius = new float[] { 0,
                                              0,
                                              0,
                                              0.015382f,
                                              0.013382f,
                                              0.01028295f,
                                              0.01822828f,
                                              0.01029526f,
                                              0.008038102f,
                                              0.02323196f,
                                              0.01117394f,
                                              0.008030958f,
                                              0.01608828f,
                                              0.009922137f,
                                              0.007611672f,
                                              0.01823196f,
                                              0.015f,
                                              0.008483353f,
                                              0.006764194f,
                                              0.009768805f,
                                              0.007636196f,
                                              0.007629411f,
                                              0.007231089f,
                                              0.006425985f };
        for (int i = 0; i < skeleton_num_capsules * oculus_skeletons.Length; i++)
        {
            _skeleton_capsule_radius[i] = preset_capsule_radius[i % 24] / transform.localScale.x;
            //_skeleton_capsule_radius[i] = preset_capsule_radius / transform.localScale.x;
        }
        skeleton_capsule_radius.CopyFromArray(_skeleton_capsule_radius);
        
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
            x.CopyFromArray(splatManager.m_pos);
            other_data.CopyFromArray(splatManager.m_other);
            init_sh.CopyFromArray(splatManager.m_SH);
            _Kernel_init_dg.LaunchAsync(dg);
            _Kernel_scale_to_unit_cube.LaunchAsync(x, other_data, bounding_eps);
            _Kernel_init_gaussian_data.LaunchAsync(init_rotation, init_scale, other_data);

        }

        _Mesh = new Mesh();
        int[] indices = new int[NParticles];
        for (int i = 0; i < NParticles; ++i)
        {
            indices[i] = i;
        }
        Vector3[] vertices = new Vector3[NParticles];

        var index = indices.ToArray();
        _Mesh.vertices = vertices;
        _Mesh.SetIndices(indices, MeshTopology.Points, 0);
        _Mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 114514f);
        _Mesh.name = "Mpm3D";
        _Mesh.MarkModified();
        _Mesh.UploadMeshData(false);
        _MeshFilter.mesh = _Mesh;
        if (renderType == RenderType.PointMesh)
        {
            _MeshRenderer.material = pointMaterial;
            bounds = new Bounds(_MeshFilter.transform.position + Vector3.one * 0.5f, Vector3.one);
        }

        if (UseRecordDate)
        {
            LoadHandMotionData();
        }
        spaceAction = new InputAction(binding: "<Keyboard>/space");
        spaceAction.performed += ctx => OnSpacePressed();
        spaceAction.Enable();
    }

    // Update is called once per frame
    void Update()
    {
        if (!RunSimulation)
        {
            if (!updated)
            {
                other_data.CopyToNativeBufferAsync(splatManager.m_Render.m_GpuOtherData.GetNativeBufferPtr());
                x.CopyToNativeBufferAsync(splatManager.m_Render.m_GpuPosData.GetNativeBufferPtr());
                updated = true;
                Runtime.Submit();
            }
            return;
        }
        if (_Compute_Graph_g_update != null)
        {
            _Compute_Graph_g_update.LaunchAsync(new Dictionary<string, object>
            {
                {"v", v},
                { "grid_m",grid_m},
                { "x",x},
                { "C",C},
                { "grid_v",grid_v},
                { "g_x",g.x},
                { "g_y",g.y},
                { "g_z",g.z},
            });
        }
        else
        {
            //kernel update
            float dt = max_dt, time_left = frame_time;
            switch (obstacleType)
            {
                case ObstacleType.Sphere:
                    UpdateSphereSDF();
                    if (Intersectwith(sphere))
                    {
                        _Kernel_substep_calculate_signed_distance_field.LaunchAsync(obstacle_pos, sdf, obstacle_norms, obstacle_radius, dx, dt, boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
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
                        _Kernel_substep_calculate_hand_sdf.LaunchAsync(skeleton_segments, skeleton_velocities, sdf, obstacle_norms, obstacle_velocity, skeleton_capsule_radius, dx,
                        boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
                    }
                    break;
            }
            while (time_left > 0)
            {
                time_left -= dt;
                _Kernel_subsetep_reset_grid.LaunchAsync(grid_v, grid_m);
                switch (stressType)
                {
                    case StressType.NeoHookean:
                        _Kernel_substep_neohookean_p2g.LaunchAsync(x, v, C, dg, grid_v, grid_m, mu, lambda, p_vol, p_mass, dx, dt,
                         boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
                        break;
                    case StressType.Kirchhoff:
                        _Kernel_substep_Kirchhoff_p2g.LaunchAsync(x, v, C, dg, grid_v, grid_m, mu, lambda, p_vol, p_mass, dx, dt,
                        boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
                        break;
                }

                _Kernel_substep_update_grid_v.LaunchAsync(grid_v, grid_m, sdf, obstacle_norms, obstacle_velocity, g.x, g.y, g.z, colide_factor, damping, friction_k, v_allowed, dt, n_grid, dx, bound,
                boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
                _Kernel_substep_g2p.LaunchAsync(x, v, C, grid_v, dx, dt, boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
                switch (plasticityType)
                {
                    case PlasticityType.Von_Mises:
                        _Kernel_substep_apply_Von_Mises_plasticity.LaunchAsync(dg, x, mu, SigY, boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
                        break;
                    case PlasticityType.Drucker_Prager:
                        _Kernel_substep_apply_Drucker_Prager_plasticity.LaunchAsync(dg, x, lambda, mu, alpha, boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
                        break;
                    case PlasticityType.Clamp:
                        _Kernel_substep_apply_clamp_plasticity.LaunchAsync(dg, x, min_clamp, max_clamp, boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
                        break;
                    case PlasticityType.Elastic:
                        break;
                }
                if (use_correct_cfl)
                {
                    v_allowed = float.MaxValue;
                    //Taichi Allocate memory,hostwrite are not considered
                    _Kernel_substep_get_max_speed.LaunchAsync(v, x, max_v, boundary_min[0], boundary_max[0], boundary_min[1], boundary_max[1], boundary_min[2], boundary_max[2]);
                    float[] max_speed = new float[1];
                    max_v.CopyToArray(max_speed);
                    dt = Mathf.Min(max_dt, dx * allowed_cfl / max_speed[0]);
                    dt = Mathf.Min(dt, time_left);
                }
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
            _Kernel_substep_update_gaussian_data.LaunchAsync(init_rotation, init_scale, dg, other_data, init_sh, sh);
            other_data.CopyToNativeBufferAsync(splatManager.m_Render.m_GpuOtherData.GetNativeBufferPtr());
            sh.CopyToNativeBufferAsync(splatManager.m_Render.m_GpuSHData.GetNativeBufferPtr());
            x.CopyToNativeBufferAsync(splatManager.m_Render.m_GpuPosData.GetNativeBufferPtr());
        }
        Runtime.Submit();
    }

    public void Reset()
    {
        if (_Compute_Graph_g_init != null)
        {
            _Compute_Graph_g_init.LaunchAsync(new Dictionary<string, object>
            {
                { "x", x },
                { "v", v },
            });
        }
        else
        {
            //kernel initialize
            _Kernel_init_particles.LaunchAsync(x, v);
        }
    }

    public void SetGravity(float y)
    {
        g.y = y;
    }
    void UpdateSphereSDF()
    {
        for (int i = 0; i < sphere.Length; i++)
        {
            Vector3 curpos = sphere[i].Position;
            Vector3 velocity = sphere[i].Velocity;
            sphere_positions[i * 3] = curpos.x - _MeshFilter.transform.position.x;
            sphere_positions[i * 3 + 1] = curpos.y - _MeshFilter.transform.position.y;
            sphere_positions[i * 3 + 2] = curpos.z - _MeshFilter.transform.position.z;
            sphere_velocities[i * 3 + 1] = velocity.y;
            sphere_velocities[i * 3 + 2] = velocity.z;
            sphere_radii[i] = sphere[i].Radius;
        }
        obstacle_pos.CopyFromArray(sphere_positions);
        obstacle_velocity.CopyFromArray(sphere_velocities);
        obstacle_radius.CopyFromArray(sphere_radii);
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
                        UpdateHandSkeletonSegment(init + j * 6, start, end, frame_time);
                        _skeleton_capsule_radius[i * skeleton_num_capsules + j] = Skeleton_capsule_radius / transform.localScale.x;
                    }
                    skeleton_segments.CopyFromArray(hand_skeleton_segments);
                    skeleton_velocities.CopyFromArray(hand_skeleton_velocities);
                    skeleton_capsule_radius.CopyFromArray(_skeleton_capsule_radius);
                }
            }
        }
        Center /= oculus_skeletons.Length;
        boundary_min = Center - Vector3.one * hand_simulation_radius;
        boundary_max = Center + Vector3.one * hand_simulation_radius;
        boundary_min = transform.InverseTransformPoint(boundary_min);
        boundary_max = transform.InverseTransformPoint(boundary_max);
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

    private void UpdateHandSkeletonSegment(int init, Vector3 start, Vector3 end, float frameTime)
    {
        Vector3 TransformedStart = transform.InverseTransformPoint(start);
        Vector3 TransformedEnd = transform.InverseTransformPoint(end);
        hand_skeleton_segments[init] = TransformedStart.x;
        hand_skeleton_segments[init + 1] = TransformedStart.y;
        hand_skeleton_segments[init + 2] = TransformedStart.z;
        hand_skeleton_segments[init + 3] = TransformedEnd.x;
        hand_skeleton_segments[init + 4] = TransformedEnd.y;
        hand_skeleton_segments[init + 5] = TransformedEnd.z;

        hand_skeleton_velocities[init] = (TransformedStart.x - hand_skeleton_segments_prev[init]) / frameTime;
        hand_skeleton_velocities[init + 1] = (TransformedStart.y - hand_skeleton_segments_prev[init + 1]) / frameTime;
        hand_skeleton_velocities[init + 2] = (TransformedStart.z - hand_skeleton_segments_prev[init + 2]) / frameTime;
        hand_skeleton_velocities[init + 3] = (TransformedEnd.x - hand_skeleton_segments_prev[init + 3]) / frameTime;
        hand_skeleton_velocities[init + 4] = (TransformedEnd.y - hand_skeleton_segments_prev[init + 4]) / frameTime;
        hand_skeleton_velocities[init + 5] = (TransformedEnd.z - hand_skeleton_segments_prev[init + 5]) / frameTime;

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
            Bounds b = new(o[i].transform.position, o[i].transform.localScale);

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
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < size; i++)
        {
            sb.AppendFormat("Byte {0}: {1:X2} ", i, data[i]);
        }
        UnityEngine.Debug.Log(sb.ToString());
    }
}