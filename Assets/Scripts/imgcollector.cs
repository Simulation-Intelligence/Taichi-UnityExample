using UnityEngine;
using System.IO;

public class NeRFDataGenerator : MonoBehaviour
{
    public GameObject model; // 球心的模型
    public int numberOfSamples = 100; // 样本数量
    public float sphereRadius = 10f; // 球体半径
    public Camera captureCamera; // 用于拍摄的相机
    public string savePath = "Assets/NeRFImages"; // 保存路径

    void Start()
    {
        GenerateNeRFData();
    }

    void GenerateNeRFData()
    {
        if (!Directory.Exists(savePath))
        {
            Directory.CreateDirectory(savePath);
        }

        for (int i = 0; i < numberOfSamples; i++)
        {
            Vector3 randomPosition = Random.onUnitSphere * sphereRadius;
            captureCamera.transform.position = randomPosition;
            captureCamera.transform.LookAt(model.transform);

            RenderTexture renderTexture = new RenderTexture(1024, 1024, 24);
            captureCamera.targetTexture = renderTexture;
            Texture2D screenShot = new Texture2D(1024, 1024, TextureFormat.RGB24, false);

            captureCamera.Render();
            RenderTexture.active = renderTexture;
            screenShot.ReadPixels(new Rect(0, 0, 1024, 1024), 0, 0);
            captureCamera.targetTexture = null;
            RenderTexture.active = null;
            Destroy(renderTexture);

            byte[] bytes = screenShot.EncodeToJPG();
            string filename = Path.Combine(savePath, "image" + i.ToString("D3") + ".jpg");
            File.WriteAllBytes(filename, bytes);

            Debug.Log("Saved: " + filename);
        }
    }
}
