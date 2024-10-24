using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PinchGesture : MonoBehaviour
{
    public enum FingerType { Thumb, Index, Middle, Ring, Pinky }
    public enum HandType
    {
        LeftHand,
        RightHand
    }

    public HandType handType = HandType.RightHand;
    private OVRHand hand;
    private OVRSkeleton handSkeleton;
    public FingerType firstFinger = FingerType.Thumb;
    public FingerType secondFinger = FingerType.Middle;

    [HideInInspector] public bool isPinching = false;
    [HideInInspector] public Vector3 initialPinchPosition;
    [HideInInspector] public Vector3 pinchMovement;
    [HideInInspector] public Vector3 lastPinchPosition;
    [HideInInspector] public Vector3 pinchSpeed;

    public float pinchThreshold = 0.02f;
    public float pinchRadius = 0.05f; // Sphere radius
    private GameObject pinchSphere;
    void Start()
    {
        if (handType == HandType.LeftHand)
        {
            hand = GameObject.Find("OVRCameraRig/TrackingSpace/LeftHandAnchor/LeftOVRHand").GetComponent<OVRHand>();
            handSkeleton = GameObject.Find("OVRCameraRig/TrackingSpace/LeftHandAnchor/LeftOVRHand").GetComponent<OVRSkeleton>();
        }
        else if (handType == HandType.RightHand)
        {
            hand = GameObject.Find("OVRCameraRig/TrackingSpace/RightHandAnchor/RightOVRHand").GetComponent<OVRHand>();
            handSkeleton = GameObject.Find("OVRCameraRig/TrackingSpace/RightHandAnchor/RightOVRHand").GetComponent<OVRSkeleton>();
        }
    }

    void Update()
    {
        if (hand.IsTracked && handSkeleton != null)
        {
            DetectPinch(handSkeleton);
        }
    }

    void DetectPinch(OVRSkeleton handSkeleton)
    {
        Transform firstFingerTip = GetFingerTransform(handSkeleton, firstFinger);
        Transform secondFingerTip = GetFingerTransform(handSkeleton, secondFinger);
        if (firstFingerTip == null || secondFingerTip == null) return;

        float distance = Vector3.Distance(firstFingerTip.position, secondFingerTip.position);

        if (distance < pinchThreshold && !isPinching)
        {
            isPinching = true;
            initialPinchPosition = (firstFingerTip.position + secondFingerTip.position) / 2;
            lastPinchPosition = initialPinchPosition;
            CreateOrUpdateSphere(initialPinchPosition);
            Debug.Log("Pinch started at position: " + initialPinchPosition);
        }
        else if (distance >= pinchThreshold && isPinching)
        {
            isPinching = false;
            DestroySphere();
            pinchSpeed = Vector3.zero;
            Debug.Log("Pinch ended");
        }

        if (isPinching)
        {
            Vector3 currentPinchPosition = (firstFingerTip.position + secondFingerTip.position) / 2;
            pinchMovement = currentPinchPosition - initialPinchPosition;
            pinchSpeed = (currentPinchPosition - lastPinchPosition) / Time.deltaTime;
            lastPinchPosition = currentPinchPosition;

            CreateOrUpdateSphere(currentPinchPosition);

            Debug.Log("Pinch movement: " + pinchMovement + ", Speed: " + pinchSpeed);
        }
    }

    void CreateOrUpdateSphere(Vector3 position)
    {
        if (pinchSphere == null)
        {
            pinchSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            pinchSphere.transform.localScale = Vector3.one * (2 * pinchRadius);

            // Create a transparent material
            Material transparentMaterial = new Material(Shader.Find("Standard"));
            transparentMaterial.color = new Color(0, 1, 0, 0.2f); // Semi-transparent green
            transparentMaterial.SetFloat("_Mode", 3); // Enable transparency mode
            transparentMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            transparentMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            transparentMaterial.SetInt("_ZWrite", 0);
            transparentMaterial.DisableKeyword("_ALPHATEST_ON");
            transparentMaterial.EnableKeyword("_ALPHABLEND_ON");
            transparentMaterial.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            transparentMaterial.renderQueue = 3000;

            pinchSphere.GetComponent<Renderer>().material = transparentMaterial;
        }
        pinchSphere.transform.position = position;
    }

    void DestroySphere()
    {
        if (pinchSphere != null)
        {
            Destroy(pinchSphere);
            pinchSphere = null;
        }
    }

    Transform GetFingerTransform(OVRSkeleton handSkeleton, FingerType fingerType)
    {
        foreach (var bone in handSkeleton.Bones)
        {
            if (bone.Id == GetBoneId(fingerType)) return bone.Transform;
        }
        return null;
    }

    OVRSkeleton.BoneId GetBoneId(FingerType fingerType)
    {
        return fingerType switch
        {
            FingerType.Thumb => OVRSkeleton.BoneId.Hand_ThumbTip,
            FingerType.Index => OVRSkeleton.BoneId.Hand_IndexTip,
            FingerType.Middle => OVRSkeleton.BoneId.Hand_MiddleTip,
            FingerType.Ring => OVRSkeleton.BoneId.Hand_RingTip,
            FingerType.Pinky => OVRSkeleton.BoneId.Hand_PinkyTip,
            _ => OVRSkeleton.BoneId.Invalid
        };
    }
}
