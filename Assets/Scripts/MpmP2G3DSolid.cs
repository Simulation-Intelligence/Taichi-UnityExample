using System.Collections.Generic;
using UnityEngine;
using Taichi;
using UnityEngine.Rendering;
using System.Linq;

public class MpmP2G3DSolid : MonoBehaviour
{
    private Mesh _Mesh;
    private MeshFilter _MeshFilter;

    public AotModuleAsset Mpm3DModule;
    private Kernel _Kernel_subsetep_reset_grid;
    private Kernel _Kernel_substep_p2g;
    private Kernel _Kernel_substep_obstacle_p2g;
    private Kernel _Kernel_substep_update_grid_v_;
    private Kernel _Kernel_substep_g2p;
    private Kernel _Kernel_substep_apply_plasticity;
    private Kernel _Kernel_init_particles;
    public NdArray<float> x;
    public NdArray<float> v;
    public NdArray<float> C;
    public NdArray<float> dg;
    public NdArray<float> grid_v;
    public NdArray<float> grid_m;
    public NdArray<float> obstacle_pos;
    public NdArray<float> obstacle_velocity;
    private Bounds bounds;

    private ComputeGraph _Compute_Graph_g_init;
    private ComputeGraph _Compute_Graph_g_update;

    public float g_x = 0.0f;
    public float g_y = -9.8f;
    public float g_z = 0.0f;

    public MeshVertexInfo meshVertexInfo;
    public float obstacleMass;

    public int NParticles = 524288;

    public bool use_plasticity = false;

    // Start is called before the first frame update
    void Start()
    {
        var kernels = Mpm3DModule.GetAllKernels().ToDictionary(x => x.Name);
        if (kernels.Count > 0)
        {
            _Kernel_subsetep_reset_grid = kernels["substep_reset_grid"];
            _Kernel_substep_p2g = kernels["substep_p2g"];
            _Kernel_substep_obstacle_p2g = kernels["substep_obstacle_p2g"];
            _Kernel_substep_update_grid_v_ = kernels["substep_update_grid_v_"];
            _Kernel_substep_g2p = kernels["substep_g2p"];
            if (use_plasticity)
                _Kernel_substep_apply_plasticity = kernels["substep_apply_plasticity"];
            _Kernel_init_particles = kernels["init_particles"];
        }
        var cgraphs = Mpm3DModule.GetAllComputeGrpahs().ToDictionary(x => x.Name);
        if (cgraphs.Count > 0)
        {
            _Compute_Graph_g_init = cgraphs["init"];
            _Compute_Graph_g_update = cgraphs["update"];
        }
        int n_grid = 64;

        int vertexCount = meshVertexInfo.combinedVertices.Length / 3;
        //Taichi Allocate memory,hostwrite are not considered
        x = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3).Build();
        v = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3).Build();
        C = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3, 3).Build();
        dg = new NdArrayBuilder<float>().Shape(NParticles).ElemShape(3, 3).Build();
        grid_v = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).ElemShape(3).Build();
        grid_m = new NdArrayBuilder<float>().Shape(n_grid, n_grid, n_grid).Build();
        obstacle_pos = new NdArrayBuilder<float>().Shape(vertexCount).ElemShape(3).HostWrite(true).Build();
        obstacle_velocity = new NdArrayBuilder<float>().Shape(vertexCount).ElemShape(3).HostWrite(true).Build();

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
            _Kernel_init_particles.LaunchAsync(x, v, dg);
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
                _Kernel_substep_p2g.LaunchAsync(x, v, C, dg, grid_v, grid_m);
                _Kernel_substep_obstacle_p2g.LaunchAsync(obstacle_pos, obstacle_velocity, grid_v, grid_m, obstacleMass);
                _Kernel_substep_update_grid_v_.LaunchAsync(grid_v, grid_m, g_x, g_y, g_z);
                _Kernel_substep_g2p.LaunchAsync(x, v, C, grid_v);
                if (_Kernel_substep_apply_plasticity != null && use_plasticity)
                {
                    _Kernel_substep_apply_plasticity.LaunchAsync(x, v, C, dg);
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
        g_y = y;
    }
    void UpdateObstacle()
    {
        obstacle_pos.CopyFromArray(meshVertexInfo.combinedVertices);
        obstacle_velocity.CopyFromArray(meshVertexInfo.combinedVelocities);
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