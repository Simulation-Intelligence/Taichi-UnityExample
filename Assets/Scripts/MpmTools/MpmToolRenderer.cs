using UnityEngine;

[RequireComponent(typeof(MpmTool))]
public class ToolRender : MonoBehaviour
{
    public MpmTool mpmTool; // 引用MpmTool组件
    public Material material; // 用于渲染的材质

    private GameObject[] capsuleObjects; // 用于存储胶囊体的组合（每个包含两个球和一个圆柱）

    void Start()
    {
        // 获取 MpmTool 组件
        if (mpmTool == null)
        {
            mpmTool = GetComponent<MpmTool>();
        }

        // 初始化胶囊体组合数组
        capsuleObjects = new GameObject[mpmTool.numCapsules];

        // 创建胶囊体组合
        for (int i = 0; i < mpmTool.numCapsules; i++)
        {
            GameObject capsule = new GameObject("Capsule_" + i);
            capsule.transform.SetParent(transform); // 将胶囊体设置为当前GameObject的子对象

            // 创建球体
            GameObject sphere1 = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere1.transform.SetParent(capsule.transform);
            sphere1.GetComponent<Renderer>().material = material; // 设置材质

            GameObject sphere2 = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere2.transform.SetParent(capsule.transform);
            sphere2.GetComponent<Renderer>().material = material; // 设置材质

            // 创建圆柱体
            GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            cylinder.transform.SetParent(capsule.transform);
            cylinder.GetComponent<Renderer>().material = material; // 设置材质

            // 保存这个组合
            capsuleObjects[i] = capsule;
        }
    }

    void Update()
    {
        if (mpmTool == null || capsuleObjects == null)
        {
            return;
        }

        // 更新每个胶囊体的位置和大小
        for (int i = 0; i < mpmTool.numCapsules; i++)
        {
            MpmTool.Capsule capsule = mpmTool.capsules[i];
            UpdateCapsuleObject(capsuleObjects[i], capsule.start, capsule.end, capsule.radius);
        }
    }

    // 更新胶囊体组合对象的函数
    void UpdateCapsuleObject(GameObject capsuleObject, Vector3 start, Vector3 end, float radius)
    {
        // 计算中点和方向
        Vector3 center = (start + end) / 2.0f;
        Vector3 direction = (end - start).normalized;
        float height = (end - start).magnitude;

        // 更新球体位置和大小
        Transform sphere1 = capsuleObject.transform.GetChild(0);
        Transform sphere2 = capsuleObject.transform.GetChild(1);
        sphere1.position = start;
        sphere2.position = end;
        sphere1.localScale = sphere2.localScale = Vector3.one * radius * 2;

        // 更新圆柱体位置、旋转和大小
        Transform cylinder = capsuleObject.transform.GetChild(2);
        cylinder.position = center;
        cylinder.up = direction; // 设置方向
        cylinder.localScale = new Vector3(radius * 2, height / 2, radius * 2); // 注意：Unity中的圆柱高度是沿y轴的
    }
}
