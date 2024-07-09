using UnityEngine;

[System.Serializable]
public class Sphere
{
    public GameObject sphereObject;
    private Vector3 lastpos = Vector3.zero;

    public Sphere(GameObject sphere)
    {
        sphereObject = sphere;
    }

    public Vector3 Position
    {
        get { return sphereObject.transform.position; }
        set { sphereObject.transform.position = value; }
    }

    public Vector3 Velocity
    {
        get
        {
            Vector3 curpos = sphereObject.transform.position;//当前点
            Vector3 _speed = (curpos - lastpos) / Time.deltaTime;//与上一个点做计算除去当前帧花的时间。
            lastpos = curpos;//把当前点保存下一次用
            return _speed;

        }
    }

    public float Radius
    {
        get { return sphereObject.transform.localScale.x / 2; }
        set { sphereObject.transform.localScale = Vector3.one * value * 2; }
    }
}
