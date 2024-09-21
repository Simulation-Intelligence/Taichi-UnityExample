using UnityEngine;

public class MatToolCone : MatTool
{
    public Vector3 sphere1 = new Vector3(0, 0, 0);
    public float radii1 = 0.1f;
    public Vector3 sphere2 = new Vector3(0.5f, 0, 0.8f);
    public float radii2 = 0.1f;
    // Set radii3 to 0.0f to treat it as a cone
    public Vector3 sphere3 = new Vector3(0, 0, 0);
    public float radii3 = 0.0f;
    
    void Awake()
    {
        numPrimitives = 1;
        init_primitives = new Primitive[numPrimitives];
        
        init_primitives[0].sphere1 = sphere1;
        init_primitives[0].radii1 = radii1;
        init_primitives[0].sphere2 = sphere2;
        init_primitives[0].radii2 = radii2;
        init_primitives[0].sphere3 = sphere3;
        init_primitives[0].radii3 = radii3;
    }
}