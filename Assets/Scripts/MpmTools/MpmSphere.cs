
using UnityEngine;


//示例工具
public class MpmSphere : MpmTool
{
    public float radius = 0.1f;
    void Awake()
    {
        // 在 Awake 中设置特定的 numCapsules 和 init_capsules 值
        numCapsules = 1;

        init_capsules = new Capsule[numCapsules];

        // 初始化每个 Capsule 的位置和半径
        init_capsules[0].start = new Vector3(0, 0, 0);
        init_capsules[0].end = new Vector3(0, 0, 0);
        init_capsules[0].radius = radius;

    }
}

