using UnityEngine;

public class MatTool : MonoBehaviour
{
    public struct Primitive
    {
        public Vector3 sphere1;
        public float radii1;
        public Vector3 sphere2;
        public float radii2;
        public Vector3 sphere3;
        public float radii3;
    }
    [HideInInspector]
    public int numPrimitives;
    public Primitive[] init_primitives;
    public Primitive[] primitives;
    
    void Start()
    {
        primitives = new Primitive[numPrimitives];
        for (int i = 0; i < numPrimitives; i++)
        {
            primitives[i] = init_primitives[i];
        }
    }
    void Update()
    {
        UpdatePrimitives();
    }
    // Virutal method to be overriden by child classes
    protected virtual void UpdatePrimitives()
    {
        TransformFixedPrimitives();
    }
    void TransformFixedPrimitives()
    {
        for (int i = 0; i < numPrimitives; i++)
        {
            // Update primitive position using the oculus Hand Joint Component
            primitives[i].sphere1 = transform.TransformPoint(init_primitives[i].sphere1);
            primitives[i].sphere2 = transform.TransformPoint(init_primitives[i].sphere2);
            primitives[i].sphere3 = transform.TransformPoint(init_primitives[i].sphere3);
        }
    }
}