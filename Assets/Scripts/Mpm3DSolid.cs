using System.Collections.Generic;
using UnityEngine;
using Taichi;
using UnityEngine.Rendering;
using System.Linq;
using System.Text.RegularExpressions;
using System.Diagnostics;

public class Mpm3DSolid : MonoBehaviour
{
    private Mesh _Mesh;
    private MeshFilter _MeshFilter;

    [Header("MpM Engine")]
    [SerializeField]
    private AotModuleAsset Mpm3DModule;
    private Kernel _Kernel_subsetep_reset_grid;
    private Kernel _Kernel_substep_neohookean_p2g;
    private Kernel _Kernel_substep_Kirchhoff_p2g;
    private Kernel _Kernel_substep_calculate_signed_distance_field;
    private Kernel _Kernel_substep_update_grid_v;
    private Kernel _Kernel_substep_g2p;
    private Kernel _Kernel_substep_apply_Von_Mises_plasticity;
    private Kernel _Kernel_substep_apply_Drucker_Prager_plasticity;
    private Kernel _Kernel_substep_apply_clamp_plasticity;


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
    [Header("Material")]
    [SerializeField]
    private PlasticityType plasticityType = PlasticityType.Von_Mises;
    [SerializeField]
    private StressType stressType = StressType.NeoHookean;
    private Kernel _Kernel_init_particles;
    private NdArray<float> x;
    public NdArray<float> v;
    public NdArray<float> C;
    public NdArray<float> dg;
    public NdArray<float> grid_v;
    public NdArray<float> grid_m;
    public NdArray<float> obstacle_pos;
    public NdArray<float> obstacle_velocity;
    public NdArray<float> obstacle_radius;
    public NdArray<float> sdf;
    public NdArray<float> grid_obstacle_vel;

    private float[] sphere_positions;

    private float[] sphere_velocities;

    private float[] sphere_radii;

    private Bounds bounds;

    private ComputeGraph _Compute_Graph_g_init;
    private ComputeGraph _Compute_Graph_g_update;

    [Header("Scene Settings")]
    [SerializeField]
    private Vector3 g = new(0, -9.8f, 0);
    [SerializeField]
    private float n_grid = 64, dt = 1e-4f, cube_size = 0.2f, particle_per_grid = 8, allowed_cfl = 0.5f;
    [Header("Obstacle")]
    [SerializeField]
    private Sphere[] sphere;


    private int NParticles => (int)(n_grid * n_grid * n_grid * cube_size * cube_size * cube_size * particle_per_grid);
    private float dx => 1 / n_grid;

    private float p_vol => dx * dx * dx;

    private float p_mass => p_vol * p_rho / particle_per_grid;

    private float v_allowed => allowed_cfl * dx / dt;
    [Header("Scalars")]
    [SerializeField]
    private float E = 1e4f;
    [SerializeField]
    private float SigY = 1000, nu = 0.3f, colide_factor = 0.5f, p_rho = 1000, min_clamp = 0.1f, max_clamp = 0.1f, friction_angle = 30;

    private float mu => E / (2 * (1 + nu));
    private float lambda => E * nu / ((1 + nu) * (1 - 2 * nu));

    private float sin_phi => Mathf.Sin(friction_angle * Mathf.Deg2Rad);

    private float alpha => Mathf.Sqrt(2.0f / 3.0f) * 2 * sin_phi / (3 - sin_phi);

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
        }

        var cgraphs = Mpm3DModule.GetAllComputeGrpahs().ToDictionary(x => x.Name);
        if (cgraphs.Count > 0)
        {
            _Compute_Graph_g_init = cgraphs["init"];
            _Compute_Graph_g_update = cgraphs["update"];
        }
        int n_grid = 64;

        //Taichi Allocate memory,hostwrite are not considered
        x = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3).Build();
        v = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3).Build();
        C = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3, 3).Build();
        dg = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3, 3).Build();
        grid_v = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).ElemShape(3).Build();
        grid_m = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).Build();
        sdf = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).Build();
        grid_obstacle_vel = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).ElemShape(3).Build();
        obstacle_pos = new NdArrayBuilder<float>().Shape(sphere.Length).ElemShape(3).HostWrite(true).Build();
        obstacle_velocity = new NdArrayBuilder<float>().Shape(sphere.Length).ElemShape(3).HostWrite(true).Build();
        obstacle_radius = new NdArrayBuilder<float>().Shape(sphere.Length).HostWrite(true).Build();
        sphere_positions = new float[3 * sphere.Length];
        sphere_velocities = new float[3 * sphere.Length];
        sphere_radii = new float[sphere.Length];
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
        UpdateObstacle();
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
            const int NUM_SUBSTEPS = 50;
            for (int i = 0; i < NUM_SUBSTEPS; i++)
            {
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
                if (Intersectwith(sphere))
                {
                    _Kernel_substep_calculate_signed_distance_field.LaunchAsync(obstacle_pos, obstacle_velocity, sdf, grid_obstacle_vel, obstacle_radius, colide_factor, dx, dt);
                }
                _Kernel_substep_update_grid_v.LaunchAsync(grid_v, grid_m, sdf, grid_obstacle_vel, g.x, g.y, g.z, v_allowed, dt);
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
    void UpdateObstacle()
    {
        for (int i = 0; i < sphere.Length; i++)
        {
            Vector3 curpos = sphere[i].Position;
            Vector3 velocity = sphere[i].Velocity;
            sphere_positions[i * 3] = curpos.x - _MeshFilter.transform.position.x;
            sphere_positions[i * 3 + 1] = curpos.y - _MeshFilter.transform.position.y;
            sphere_positions[i * 3 + 2] = curpos.z - _MeshFilter.transform.position.z;
            sphere_velocities[i * 3] = velocity.x;
            sphere_velocities[i * 3 + 1] = velocity.y;
            sphere_velocities[i * 3 + 2] = velocity.z;
            sphere_radii[i] = sphere[i].Radius;
        }
        obstacle_pos.CopyFromArray(sphere_positions);
        obstacle_velocity.CopyFromArray(sphere_velocities);
        obstacle_radius.CopyFromArray(sphere_radii);
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
}