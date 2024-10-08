using UnityEngine;

[RequireComponent(typeof(MatTool))]
public class MatToolRenderer: MonoBehaviour
{
    public MatTool matTool; // Reference to the MatTool component
    public Material material; // Material used for rendering

    // Array to store the primitive assemblies (each consisting of three spheres)
    private GameObject[] primitiveObjects;
    
    void Start()
    {
        // Inherit material
        if (material == null)
        {
            material = GetComponent<MeshRenderer>().material;
        }

        // Get the MatTool component
        if (matTool == null)
        {
            matTool = GetComponent<MatTool>();
        }

        // Initialize the primitive assembly array
        primitiveObjects = new GameObject[matTool.numPrimitives];
        
        // Create child objects for each primitive
        for (int i = 0; i < matTool.numPrimitives; i++)
        {
            GameObject primitive = new GameObject("Primitive_" + i);
            primitive.transform.SetParent(transform); // Set child gameObject
            primitive.transform.localScale = Vector3.one;
            
            // Create three spheres
            GameObject sphere1 = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere1.transform.SetParent(primitive.transform);
            sphere1.GetComponent<Renderer>().material = material; 

            GameObject sphere2 = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere2.transform.SetParent(primitive.transform);
            sphere2.GetComponent<Renderer>().material = material; 

            GameObject sphere3 = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere3.transform.SetParent(primitive.transform);
            sphere3.GetComponent<Renderer>().material = material;

            primitive.transform.GetChild(0).localPosition = matTool.init_primitives[i].sphere1;
            primitive.transform.GetChild(1).localPosition = matTool.init_primitives[i].sphere2;
            primitive.transform.GetChild(2).localPosition = matTool.init_primitives[i].sphere3;
            primitive.transform.GetChild(0).localScale = Vector3.one * matTool.init_primitives[i].radii1 * 2;
            primitive.transform.GetChild(1).localScale = Vector3.one * matTool.init_primitives[i].radii2 * 2;
            primitive.transform.GetChild(2).localScale = Vector3.one * matTool.init_primitives[i].radii3 * 2;

            primitiveObjects[i] = primitive;
        }
    }
    
    void Update()
    {
        if (matTool == null || primitiveObjects == null)
        {
            return;
        }
        // Update each primitive's position and size
        for (int i = 0; i < matTool.numPrimitives; i++)
        {
            MatTool.Primitive primitive = matTool.primitives[i];
            UpdatePrimitiveObject(primitiveObjects[i], primitive.sphere1, primitive.sphere2, primitive.sphere3, primitive.radii1, primitive.radii2, primitive.radii3);
        }
    }

    void UpdatePrimitiveObject(GameObject primitiveObject, Vector3 sphere1, Vector3 sphere2, Vector3 sphere3, float radii1, float radii2, float radii3)
    {
        // Update the position and size of the spheres
        primitiveObject.transform.GetChild(0).position = sphere1;
        primitiveObject.transform.GetChild(1).position = sphere2;
        primitiveObject.transform.GetChild(2).position = sphere3;
    }
}