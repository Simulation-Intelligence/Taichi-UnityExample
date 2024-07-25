using UnityEngine;
using System;
public class VolumeTextureUpdater : MonoBehaviour
{
    public int width = 32;  // X 轴尺寸
    public int height = 32; // Y 轴尺寸
    public int depth = 32;  // Z 轴尺寸
    public GameObject targetObject; // 目标 GameObject
    public string texturePropertyName = "_volumeTex"; // 材质中使用的纹理属性名

    public float[] densityData; // 三维数组存储密度数据

    public float max_density = 100;
    private Texture3D volumeTex;   // Texture3D 对象
    private Material targetMaterial; // 目标材质

    void Start()
    {
        // 初始化密度数据和 Texture3D
        densityData = new float[width * height * depth];
        volumeTex = new Texture3D(width, height, depth, TextureFormat.RFloat, false);

        // 获取目标对象的材质
        targetMaterial = targetObject.GetComponent<Renderer>().material;
        if (targetMaterial == null)
        {
            Debug.LogError("Target object does not have a material.");
            return;
        }

        // 初始化或更新密度数据，这里只是一个示例，可以根据实际需求更新
        UpdateDensityData();

        // 设置初始 Texture3D
        UpdateVolumeTexture();

    }

    void Update()
    {
        // 在每一帧或需要时更新密度数据
        //UpdateDensityData();

        // 更新 Texture3D 的数据
        UpdateVolumeTexture();
    }

    void UpdateDensityData()
    {
        // 定义球体的半径
        float radius = 0.5f * Mathf.Min(width, height, depth);
        // 定义球体的中心
        Vector3 center = new Vector3(width / 2.0f, height / 2.0f, depth / 2.0f);

        for (int z = 0; z < depth; z++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // 计算当前点到中心的距离
                    Vector3 pos = new Vector3(x, y, z);
                    float distance = Vector3.Distance(pos, center);

                    // 如果点在球体内，设置密度为 1，否则为 0
                    densityData[x + y * width + z * width * height] = (distance <= radius) ? 1.0f : 0.0f;
                }
            }
        }
    }

    void UpdateVolumeTexture()
    {
        Color[] colors = new Color[width * height * depth];

        for (int z = 0; z < depth; z++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int index = x + y * width + z * width * height;
                    float density = Math.Min(1.0f, densityData[index] / max_density);
                    if (density > 0.0f)
                        colors[index] = new Color(density, 0, 0, 0);
                    else
                        colors[index] = new Color(0, 0, 0, 0);
                }
            }
        }

        volumeTex.SetPixels(colors);
        volumeTex.Apply();

        // 将 Texture3D 设置到材质的指定属性
        targetMaterial.SetTexture(texturePropertyName, volumeTex);
    }
}
