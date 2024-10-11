using UnityEngine;

public class AverageFrameDelay : MonoBehaviour
{
    private float totalFrameTime = 0f;
    private int frameCount = 0;
    private float updateInterval = 1.0f; // 计算平均帧延迟的时间窗口，1秒
    private float nextUpdate = 0.0f;

    void Update()
    {
        // 累加每一帧的时间
        totalFrameTime += Time.deltaTime;
        frameCount++;

        // 如果经过了预设的时间窗口（比如1秒），进行统计
        if (Time.time >= nextUpdate)
        {
            // 计算平均帧延迟（单位：秒）
            float averageFrameTime = totalFrameTime / frameCount;

            // 也可以转换为帧率（FPS）
            float averageFPS = 1.0f / averageFrameTime;

            // 输出平均帧延迟和FPS
            Debug.Log($"平均帧延迟: {averageFrameTime * 1000.0f} 毫秒");
            Debug.Log($"平均FPS: {averageFPS}");

            // 重置计数器
            totalFrameTime = 0f;
            frameCount = 0;
            nextUpdate = Time.time + updateInterval;
        }
    }
}
