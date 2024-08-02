using UnityEngine;
using UnityEngine.Rendering;
using GaussianSplatting.Runtime;
public class GaussianSplatRenderManager : MonoBehaviour
{
    // Member variable of GaussianSplatRender
    public GaussianSplatRenderer m_Render;

    // Array to store position data
    public float[] m_pos;

    public float[] m_other;

    public float[] m_color;

    public float[] m_SH;

    public float eps = 0.1f;

    public Vector3 min, max;
    bool setted = false;



    // Member to store the number of splats
    public int splatsNum { get; private set; }

    // Awake is called when the script instance is being loaded
    void Start()
    {
        splatsNum = m_Render.splatCount;
        m_pos = new float[splatsNum * 3];
        m_other = new float[splatsNum * 4];
        m_SH = new float[splatsNum * 16 * 3];
        GetPos();
        GetOther();
        GetShs();
        ScaleToUnitCube();

    }
    void Update()
    {
        // if (!setted)
        // {
        //     m_Render.m_GpuOtherData.SetData(m_other);
        //     setted = true;
        // }
    }
    // Function to copy data from m_GpuPosData to m_pos and update splatsNum
    public void GetPos()
    {
        if (m_Render == null)
        {
            Debug.LogError("GaussianSplatRenderSystem instance is not initialized.");
            return;
        }

        var gpuPosData = m_Render.m_GpuPosData;
        if (gpuPosData == null || gpuPosData.count == 0)
        {
            Debug.LogWarning("No position data available.");
            return;
        }
        gpuPosData.GetData(m_pos);
    }
    public void GetOther()
    {
        if (m_Render == null)
        {
            Debug.LogError("GaussianSplatRenderSystem instance is not initialized.");
            return;
        }

        var gpuOtherData = m_Render.m_GpuOtherData;
        if (gpuOtherData == null || gpuOtherData.count == 0)
        {
            Debug.LogWarning("No scale data available.");
            return;
        }
        gpuOtherData.GetData(m_other);
    }
    // public void GetColor()
    // {
    //     if (m_Render == null)
    //     {
    //         Debug.LogError("GaussianSplatRenderSystem instance is not initialized.");
    //         return;
    //     }

    //     var gpuColorData = m_Render.m_GpuColorData;
    //     if (gpuColorData == null || gpuColorData.count == 0)
    //     {
    //         Debug.LogWarning("No color data available.");
    //         return;
    //     }
    //     gpuColorData.GetData(m_color);
    // }

    public void GetShs()
    {
        if (m_Render == null)
        {
            Debug.LogError("GaussianSplatRenderSystem instance is not initialized.");
            return;
        }

        var gpuShsData = m_Render.m_GpuSHData;
        if (gpuShsData == null || gpuShsData.count == 0)
        {
            Debug.LogWarning("No shs data available.");
            return;
        }
        gpuShsData.GetData(m_SH);
    }

    public void ScaleToUnitCube()
    {
        // Find the bounding box of the splats
        min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        max = new Vector3(float.MinValue, float.MinValue, float.MinValue);

        for (int i = 0; i < splatsNum; i++)
        {
            Vector3 pos = new(m_pos[i * 3], m_pos[i * 3 + 1], m_pos[i * 3 + 2]);
            min = Vector3.Min(min, pos);
            max = Vector3.Max(max, pos);
        }

        // Center and size of the bounding box
        Vector3 center = (min + max) / 2f;
        Vector3 size = max - min;

        // Scale factor based on the largest dimension
        float scaleFactor = (1f - 2 * eps) / Mathf.Max(size.x, size.y, size.z);

        Vector3 newCenter = new(0.5f, 0.5f, 0.5f);

        for (int i = 0; i < splatsNum; i++)
        {
            // Scale positions and translate them to the new center
            m_pos[i * 3] = (m_pos[i * 3] - center.x) * scaleFactor + newCenter.x;
            m_pos[i * 3 + 1] = (m_pos[i * 3 + 1] - center.y) * scaleFactor + newCenter.y;
            m_pos[i * 3 + 2] = (m_pos[i * 3 + 2] - center.z) * scaleFactor + newCenter.z;

            // Scale scales (already exponentiated)
            m_other[i * 4 + 1] *= scaleFactor;
            m_other[i * 4 + 2] *= scaleFactor;
            m_other[i * 4 + 3] *= scaleFactor;
        }

        // Update min and max after scaling
        min = (min - center) * scaleFactor + newCenter;
        max = (max - center) * scaleFactor + newCenter;
    }
    public void Dispose()
    {
        // Manually release resources here
        m_pos = null;
        m_other = null;
        m_color = null;
        m_SH = null;
    }
    void OnDestroy()
    {
        Dispose();
    }
    // Additional methods for managing GaussianSplatRenderSystem can be added here
}
