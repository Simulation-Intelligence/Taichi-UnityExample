using UnityEngine;
using System.IO;

public class CameraVideoCapture : MonoBehaviour
{
    public GameObject objectToRotate; // 需要旋转的物体
    public Camera captureCamera; // 用于拍摄的相机
    public Vector3 rotationAxis = Vector3.up; // 物体旋转的轴（默认是Y轴）

    public float rotationSpeed = 10f; // 旋转速度
    public int Resolution = 1024; // 图像分辨率
    public string savePath = "Assets/VideoFrames"; // 保存路径
    public bool createSubfolders = false; // 是否创建子文件夹，按物体名称生成

    private int frameCount = 0; // 帧计数器

    void Start()
    {
        // 创建保存路径
        if (!Directory.Exists(savePath))
        {
            Directory.CreateDirectory(savePath);
        }

        // 如果创建子文件夹，则按物体名称创建
        if (createSubfolders)
        {
            string subfolder = savePath + "/images_" + objectToRotate.name;
            if (!Directory.Exists(subfolder))
            {
                Directory.CreateDirectory(subfolder);
            }
            savePath = subfolder;
        }
    }

    void Update()
    {
        // 让物体沿某一轴旋转（可以修改为任意轴）
        objectToRotate.transform.Rotate(rotationAxis * rotationSpeed * Time.deltaTime); // 默认绕Y轴旋转

        // 每一帧生成一个图片
        CaptureFrame();
    }

    void CaptureFrame()
    {
        // 生成随机位置
        // Vector3 randomPosition = Random.onUnitSphere * 10f + objectToRotate.transform.position;
        // captureCamera.transform.position = randomPosition;
        captureCamera.transform.LookAt(objectToRotate.transform); // 相机始终指向物体

        // 创建RenderTexture并设置为相机目标纹理
        RenderTexture renderTexture = new RenderTexture(Resolution, Resolution, 24);
        captureCamera.targetTexture = renderTexture;

        // 创建一个临时的Texture2D来保存捕获的图像
        Texture2D screenShot = new Texture2D(Resolution, Resolution, TextureFormat.RGB24, false);

        // 渲染相机视图并读取像素
        captureCamera.Render();
        RenderTexture.active = renderTexture;
        screenShot.ReadPixels(new Rect(0, 0, Resolution, Resolution), 0, 0);
        captureCamera.targetTexture = null; // 清除目标纹理
        RenderTexture.active = null; // 清除活动纹理

        // 清理RenderTexture
        Destroy(renderTexture);

        // 编码为PNG并保存到指定路径
        byte[] bytes = screenShot.EncodeToPNG();
        string filename = Path.Combine(savePath, "frame_" + frameCount.ToString("D3") + ".png");
        File.WriteAllBytes(filename, bytes);

        // 输出保存信息
        Debug.Log("Saved: " + filename);

        // 更新帧计数器
        frameCount++;
    }
}
