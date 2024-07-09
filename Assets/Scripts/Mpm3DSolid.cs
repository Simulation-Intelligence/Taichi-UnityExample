using System.Collections.Generic;
using UnityEngine;
using Taichi;
using UnityEngine.Rendering;
using System.Linq;

public class Mpm3DSolid : MonoBehaviour
{
    private Mesh _Mesh;
    private MeshFilter _MeshFilter;

    public AotModuleAsset Mpm3DModule;
    private Kernel _Kernel_subsetep_reset_grid;
    private Kernel _Kernel_substep_p2g;
    private Kernel _Kernel_substep_calculate_signed_distance_field;
    private Kernel _Kernel_substep_update_grid_v;
    private Kernel _Kernel_substep_g2p;
    private Kernel _Kernel_init_particles;
    public NdArray<float> x;
    public NdArray<float> v;
    public NdArray<float> C;
    public NdArray<float> J;
    public NdArray<float> dg;
    public NdArray<float> grid_v;
    public NdArray<float> grid_m;
    public NdArray<float> obstacle_pos;
    public NdArray<float> obstacle_velocity;
    public float obstacle_radius = 0.2f;
    public NdArray<float> sdf;
    public NdArray<float> grid_obstacle_vel;

    private float[] sphere_positions;

    private float[] sphere_velocities;

    private Bounds bounds;

    private ComputeGraph _Compute_Graph_g_init;
    private ComputeGraph _Compute_Graph_g_update;

    public float g_x = 0.0f;
    public float g_y = -9.8f;
    public float g_z = 0.0f;

    public Sphere sphere;
    int NParticles = 65536;

    // Start is called before the first frame update
    void Start()
    {
        var kernels = Mpm3DModule.GetAllKernels().ToDictionary(x => x.Name);
        if (kernels.Count > 0)
        {
            _Kernel_subsetep_reset_grid = kernels["substep_reset_grid"];
            _Kernel_substep_p2g = kernels["substep_p2g"];
            _Kernel_substep_calculate_signed_distance_field = kernels["substep_calculate_signed_distance_field"];
            _Kernel_substep_update_grid_v = kernels["substep_update_grid_v"];
            _Kernel_substep_g2p = kernels["substep_g2p"];
            _Kernel_init_particles = kernels["init_particles"];
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
        J = new NdArray<float>(NParticles);
        dg = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3, 3).Build();
        grid_v = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).ElemShape(3).Build();
        grid_m = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).Build();
        sdf = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).Build();
        grid_obstacle_vel = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).ElemShape(3).Build();
        obstacle_pos = new NdArrayBuilder<float>().Shape(1).ElemShape(3).HostWrite(true).Build();
        obstacle_velocity = new NdArrayBuilder<float>().Shape(1).ElemShape(3).HostWrite(true).Build();
        sphere_positions = new float[3];
        sphere_velocities = new float[3];
        if (_Compute_Graph_g_init != null)
        {
            _Compute_Graph_g_init.LaunchAsync(new Dictionary<string, object>
            {
                { "x", x },
                { "v", v },
                { "J", J },
            });
        }
        else
        {
            //kernel initialize
            _Kernel_init_particles.LaunchAsync(x, v, J, dg);
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

        bounds = new Bounds(_MeshFilter.transform.position + Vector3.one * 0.25f, Vector3.one * 0.5f);
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
                {"J",J},
                {"grid_v",grid_v},
                {"g_x",g_x},
                {"g_y",g_y},
                {"g_z",g_z},
            });
        }
        else
        {
            //kernel update
            const int NUM_SUBSTEPS = 50;
            for (int i = 0; i < NUM_SUBSTEPS; i++)
            {
                _Kernel_subsetep_reset_grid.LaunchAsync(grid_v, grid_m);
                _Kernel_substep_p2g.LaunchAsync(x, v, C, J, dg, grid_v, grid_m);
                if (intersectwith(sphere.gameObject))
                {
                    _Kernel_substep_calculate_signed_distance_field.LaunchAsync(obstacle_pos, obstacle_velocity, sdf, grid_obstacle_vel, obstacle_radius);
                }
                _Kernel_substep_update_grid_v.LaunchAsync(grid_v, grid_m, sdf, grid_obstacle_vel, g_x, g_y, g_z);
                _Kernel_substep_g2p.LaunchAsync(x, v, C, J, grid_v);
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
                { "J", J },
            });
        }
        else
        {
            //kernel initialize
            _Kernel_init_particles.LaunchAsync(x, v, J);
        }
    }

    public void SetGravity(float y)
    {
        g_y = y;
    }
    void UpdateObstacle()
    {
        Vector3 pos = sphere.Position;
        Vector3 vel = sphere.Velocity;
        sphere_positions[0] = pos.x - _MeshFilter.transform.position.x;
        sphere_positions[1] = pos.y - _MeshFilter.transform.position.y;
        sphere_positions[2] = pos.z - _MeshFilter.transform.position.z;
        sphere_velocities[0] = vel.x;
        sphere_velocities[1] = vel.y;
        sphere_velocities[2] = vel.z;
        obstacle_pos.CopyFromArray(sphere_positions);
        obstacle_velocity.CopyFromArray(sphere_velocities);
        obstacle_radius = sphere.Radius;
    }
    bool intersectwith(GameObject o)
    {
        Bounds b = new(o.transform.position, o.transform.localScale);

        if (b == null)
        {
            return false;
        }
        return bounds.Intersects(b);
    }
}