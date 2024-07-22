using System.Collections.Generic;
using UnityEngine;
using Taichi;
using UnityEngine.Rendering;
using System.Linq;
using System.Text.RegularExpressions;
using System.Diagnostics;
using Meta.WitAi.CallbackHandlers;
using UnityEngine.UIElements;

public class Mpm3DSolidSDF : MonoBehaviour
{
    private Mesh _Mesh;
    private MeshFilter _MeshFilter;

    [Header("MpM Engine")]
    [SerializeField]
    private AotModuleAsset Mpm3DModule;
    private Kernel _Kernel_subsetep_reset_grid, _Kernel_substep_neohookean_p2g, _Kernel_substep_Kirchhoff_p2g,
    _Kernel_substep_calculate_signed_distance_field, _Kernel_substep_update_grid_v, _Kernel_substep_g2p,
     _Kernel_substep_apply_Von_Mises_plasticity, _Kernel_substep_apply_Drucker_Prager_plasticity,
     _Kernel_substep_apply_clamp_plasticity, _Kernel_substep_calculate_hand_sdf, _Kernel_substep_get_max_speed;


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
    private PlasticityType plasticityType = PlasticityType.Von_Mises;
    [SerializeField]
    private StressType stressType = StressType.NeoHookean;
    private Kernel _Kernel_init_particles;
    private NdArray<float> x, v, C, dg, grid_v, grid_m, obstacle_pos, obstacle_velocity, obstacle_radius, sdf;

    public NdArray<float> skeleton_segments, skeleton_velocities, hand_sdf, obstacle_norms, skeleton_capsule_radius, max_v;

    private float[] hand_skeleton_segments, hand_skeleton_segments_prev, hand_skeleton_velocities, _skeleton_capsule_radius, sphere_positions, sphere_velocities, sphere_radii;

    private Bounds bounds;

    private ComputeGraph _Compute_Graph_g_init;
    private ComputeGraph _Compute_Graph_g_update;

    [Header("Scene Settings")]
    [SerializeField]
    private Vector3 g = new(0, -9.8f, 0);
    [SerializeField]
    private int n_grid = 32;
    [SerializeField]
    private float max_dt = 1e-4f, frame_time = 0.005f, cube_size = 0.2f, particle_per_grid = 8, allowed_cfl = 0.5f, damping = 1f;
    [SerializeField]
    bool use_correct_cfl = false;



    private Vector3 scale;
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
    private int skeleton_num_capsules = 24; // use default 24
    private int NParticles;
    private float dx, p_vol, p_mass, v_allowed;

    [Header("Scalars")]
    [SerializeField]
    private float E = 1e4f;
    [SerializeField]
    private float SigY = 1000, nu = 0.3f, colide_factor = 0.5f, friction_k = 0.4f, p_rho = 1000, min_clamp = 0.1f, max_clamp = 0.1f, friction_angle = 30;

    private float mu, lambda, sin_phi, alpha;

    // Start is called before the first frame update
    void Start()
    {
        var kernels1 = Mpm3DModule.GetAllKernels().ToDictionary(x => x.Name);
        if (kernels1.Count > 0)
        {
            _Kernel_subsetep_reset_grid = kernels1["substep_reset_grid"];
            _Kernel_substep_neohookean_p2g = kernels1["substep_neohookean_p2g"];
            _Kernel_substep_Kirchhoff_p2g = kernels1["substep_kirchhoff_p2g"];
            _Kernel_substep_calculate_signed_distance_field = kernels1["substep_calculate_signed_distance_field"];
            _Kernel_substep_update_grid_v = kernels1["substep_update_grid_v"];
            _Kernel_substep_g2p = kernels1["substep_g2p"];
            _Kernel_substep_apply_Von_Mises_plasticity = kernels1["substep_apply_Von_Mises_plasticity"];
            _Kernel_substep_apply_Drucker_Prager_plasticity = kernels1["substep_apply_Drucker_Prager_plasticity"];
            _Kernel_substep_apply_clamp_plasticity = kernels1["substep_apply_clamp_plasticity"];
            _Kernel_init_particles = kernels1["init_particles"];

            // new added
            _Kernel_substep_calculate_hand_sdf = kernels1["substep_calculate_hand_sdf"];
            _Kernel_substep_get_max_speed = kernels1["substep_get_max_speed"];
        }

        var cgraphs = Mpm3DModule.GetAllComputeGrpahs().ToDictionary(x => x.Name);
        if (cgraphs.Count > 0)
        {
            _Compute_Graph_g_init = cgraphs["init"];
            _Compute_Graph_g_update = cgraphs["update"];
        }

        //initialize
        NParticles = (int)(n_grid * n_grid * n_grid * cube_size * cube_size * cube_size * particle_per_grid);
        dx = 1.0f / n_grid;
        scale = transform.localScale;
        p_vol = dx * dx * dx / particle_per_grid;
        p_mass = p_vol * p_rho * scale.x * scale.y * scale.z;
        mu = E / (2 * (1 + nu));
        lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
        v_allowed = allowed_cfl * dx / max_dt;
        sin_phi = Mathf.Sin(friction_angle * Mathf.Deg2Rad);
        alpha = Mathf.Sqrt(2.0f / 3.0f) * 2 * sin_phi / (3 - sin_phi);

        //Taichi Allocate memory,hostwrite are not considered
        x = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3).Build();
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

        //skeleton_num_capsules = oculus_skeletons[0].Bones.Count();
        hand_skeleton_segments = new float[skeleton_num_capsules * oculus_skeletons.Length * 6];
        hand_skeleton_segments_prev = new float[skeleton_num_capsules * oculus_skeletons.Length * 6];
        hand_skeleton_velocities = new float[skeleton_num_capsules * oculus_skeletons.Length * 6];
        _skeleton_capsule_radius = new float[skeleton_num_capsules * oculus_skeletons.Length];
        for (int i = 0; i < skeleton_num_capsules * oculus_skeletons.Length; i++)
        {
            _skeleton_capsule_radius[i] = Skeleton_capsule_radius / scale.x;
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
            //kernel initialize
            _Kernel_init_particles.LaunchAsync(x, v, dg, cube_size);
        }

        _MeshFilter = GetComponent<MeshFilter>();
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

        bounds = new Bounds(_MeshFilter.transform.position + Vector3.one * 0.5f, Vector3.one);
    }

    // Update is called once per frame
    void Update()
    {
        if (_Compute_Graph_g_update != null)
        {
            _Compute_Graph_g_update.LaunchAsync(new Dictionary<string, object>
            {
                {"v", v},
                {"grid_m",grid_m},
                {"x",x},
                {"C",C},
                {"grid_v",grid_v},
                {"g_x",g.x},
                {"g_y",g.y},
                {"g_z",g.z},
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
                        _Kernel_substep_calculate_signed_distance_field.LaunchAsync(obstacle_pos, sdf, obstacle_norms, obstacle_radius, dx, dt);
                    }
                    break;
                case ObstacleType.Hand:
                    UpdateHandSDF();
                    if (IntersectwithHand(oculus_hands))
                    {
                        _Kernel_substep_calculate_hand_sdf.LaunchAsync(skeleton_segments, skeleton_velocities, sdf, obstacle_norms, obstacle_velocity, skeleton_capsule_radius, dx);
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
                        _Kernel_substep_neohookean_p2g.LaunchAsync(x, v, C, dg, grid_v, grid_m, mu, lambda, p_vol, p_mass, dx, dt);
                        break;
                    case StressType.Kirchhoff:
                        _Kernel_substep_Kirchhoff_p2g.LaunchAsync(x, v, C, dg, grid_v, grid_m, mu, lambda, p_vol, p_mass, dx, dt);
                        break;
                }

                _Kernel_substep_update_grid_v.LaunchAsync(grid_v, grid_m, sdf, obstacle_norms, obstacle_velocity, g.x / scale.x, g.y / scale.y, g.z / scale.z, colide_factor, damping, friction_k, v_allowed, dt, n_grid);
                _Kernel_substep_g2p.LaunchAsync(x, v, C, grid_v, dx, dt);
                switch (plasticityType)
                {
                    case PlasticityType.Von_Mises:
                        _Kernel_substep_apply_Von_Mises_plasticity.LaunchAsync(dg, mu, SigY);
                        break;
                    case PlasticityType.Drucker_Prager:
                        _Kernel_substep_apply_Drucker_Prager_plasticity.LaunchAsync(dg, lambda, mu, alpha);
                        break;
                    case PlasticityType.Clamp:
                        _Kernel_substep_apply_clamp_plasticity.LaunchAsync(dg, min_clamp, max_clamp);
                        break;
                    case PlasticityType.Elastic:
                        break;
                }
                if (use_correct_cfl)
                {
                    v_allowed = float.MaxValue;
                    _Kernel_substep_get_max_speed.LaunchAsync(v, max_v);
                    float[] max_speed = new float[1];
                    max_v.CopyToArray(max_speed);
                    dt = Mathf.Min(max_dt, dx * allowed_cfl / max_speed[0]);
                    dt = Mathf.Min(dt, time_left);
                }
            }
        }
        x.CopyToNativeBufferAsync(_Mesh.GetNativeVertexBufferPtr(0));
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
            sphere_positions[i * 3] = (curpos.x - _MeshFilter.transform.position.x) / scale.x;
            sphere_positions[i * 3 + 1] = (curpos.y - _MeshFilter.transform.position.y) / scale.y;
            sphere_positions[i * 3 + 2] = (curpos.z - _MeshFilter.transform.position.z) / scale.z;
            sphere_velocities[i * 3] = velocity.x;
            sphere_velocities[i * 3 + 1] = velocity.y;
            sphere_velocities[i * 3 + 2] = velocity.z;
            sphere_radii[i] = sphere[i].Radius / scale.x;
        }
        obstacle_pos.CopyFromArray(sphere_positions);
        obstacle_velocity.CopyFromArray(sphere_velocities);
        obstacle_radius.CopyFromArray(sphere_radii);
    }

    void UpdateHandSDF()
    {
        for (int i = 0; i < oculus_skeletons.Length; i++)
        {
            if (oculus_hands[i].IsTracked && oculus_hands[i].HandConfidence == OVRHand.TrackingConfidence.High)
            {
                int numBones = oculus_skeletons[i].Bones.Count();
                //UnityEngine.Debug.Log("Num of Bones while tracking: " + numBones);
                if (numBones > 0)
                {
                    int init = i * skeleton_num_capsules * 6;
                    for (int j = 0; j < numBones; j++)
                    {
                        var bone = oculus_skeletons[i].Bones[j];
                        Vector3 start = bone.Transform.position;
                        Vector3 end = bone.Transform.parent.position;
                        hand_skeleton_segments[init + j * 6] = (start.x - _MeshFilter.transform.position.x) / scale.x;
                        hand_skeleton_segments[init + j * 6 + 1] = (start.y - _MeshFilter.transform.position.y) / scale.y;
                        hand_skeleton_segments[init + j * 6 + 2] = (start.z - _MeshFilter.transform.position.z) / scale.z;
                        hand_skeleton_segments[init + j * 6 + 3] = (end.x - _MeshFilter.transform.position.x) / scale.x;
                        hand_skeleton_segments[init + j * 6 + 4] = (end.y - _MeshFilter.transform.position.y) / scale.y;
                        hand_skeleton_segments[init + j * 6 + 5] = (end.z - _MeshFilter.transform.position.z) / scale.z;

                        hand_skeleton_velocities[init + j * 6] = (start.x - hand_skeleton_segments_prev[init + j * 6]) / frame_time;
                        hand_skeleton_velocities[init + j * 6 + 1] = (start.y - hand_skeleton_segments_prev[init + j * 6 + 1]) / frame_time;
                        hand_skeleton_velocities[init + j * 6 + 2] = (start.z - hand_skeleton_segments_prev[init + j * 6 + 2]) / frame_time;
                        hand_skeleton_velocities[init + j * 6 + 3] = (end.x - hand_skeleton_segments_prev[init + j * 6 + 3]) / frame_time;
                        hand_skeleton_velocities[init + j * 6 + 4] = (end.y - hand_skeleton_segments_prev[init + j * 6 + 4]) / frame_time;
                        hand_skeleton_velocities[init + j * 6 + 5] = (end.z - hand_skeleton_segments_prev[init + j * 6 + 5]) / frame_time;

                        hand_skeleton_segments_prev[init + j * 6] = start.x;
                        hand_skeleton_segments_prev[init + j * 6 + 1] = start.y;
                        hand_skeleton_segments_prev[init + j * 6 + 2] = start.z;
                        hand_skeleton_segments_prev[init + j * 6 + 3] = end.x;
                        hand_skeleton_segments_prev[init + j * 6 + 4] = end.y;
                        hand_skeleton_segments_prev[init + j * 6 + 5] = end.z;
                    }
                    skeleton_segments.CopyFromArray(hand_skeleton_segments);
                    skeleton_velocities.CopyFromArray(hand_skeleton_velocities);
                }
            }
        }
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
}