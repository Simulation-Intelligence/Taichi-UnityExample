using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PinchGesture : MonoBehaviour
{
    public enum HandType { LeftHand, RightHand }

    public HandType handType = HandType.RightHand;
    private OVRHand hand;
    private OVRSkeleton oculus_skeleton;

    // 手势检测的指尖选择，直接使用 BoneId
    public OVRSkeleton.BoneId firstPinchFinger = OVRSkeleton.BoneId.Hand_ThumbTip;
    public OVRSkeleton.BoneId secondPinchFinger = OVRSkeleton.BoneId.Hand_MiddleTip;

    // 旋转手势的指尖选择
    public OVRSkeleton.BoneId firstRotateFinger = OVRSkeleton.BoneId.Hand_IndexTip;
    public OVRSkeleton.BoneId secondRotateFinger = OVRSkeleton.BoneId.Hand_ThumbTip;

    // 旋转手势的旋转轴计算所需的关节
    public OVRSkeleton.BoneId firstJointBone = OVRSkeleton.BoneId.Hand_Middle1;
    public OVRSkeleton.BoneId secondJointBone = OVRSkeleton.BoneId.Hand_Ring1;

    [SerializeField]
    private SmoothHand smoothHand;
    private List<Transform> _handJointsData;
    public bool UseSmoothHand = true;

    [HideInInspector] public bool isPinching = false;
    [HideInInspector] public bool isRotating = false;

    [HideInInspector] public Vector3 initialPinchPosition;
    [HideInInspector] public Vector3 pinchMovement;
    [HideInInspector] public Vector3 lastPinchPosition;
    [HideInInspector] public Vector3 pinchSpeed;

    [HideInInspector] public Vector3 initialRotatePosition;
    [HideInInspector] public Vector3 rotationAxis;
    [HideInInspector] public float rotationSpeed;

    [HideInInspector] private float previousAngle1 = 0f;
    [HideInInspector] private float previousAngle2 = 0f;

    [HideInInspector] public Vector3 initialDirectionJoint1;
    [HideInInspector] public Vector3 initialDirectionJoint2;

    public float pinchThreshold = 0.02f;
    public float rotationThreshold = 0.03f;
    public float pinchRadius = 0.05f;

    // Visualize the selection area while pinch translation and rotation
    private GameObject pinchSphere;
    private GameObject rotationSphere;
    public bool RenderPinchSphere = true;
    public bool RenderRotationSphere = true;

    void Start()
    {
        if (handType == HandType.LeftHand)
        {
            hand = GameObject.Find("OVRCameraRig/TrackingSpace/LeftHandAnchor/LeftOVRHand").GetComponent<OVRHand>();
            oculus_skeleton = GameObject.Find("OVRCameraRig/TrackingSpace/LeftHandAnchor/LeftOVRHand").GetComponent<OVRSkeleton>();
            _handJointsData = smoothHand.SmoothLeftHandJoints;
        }
        else
        {
            hand = GameObject.Find("OVRCameraRig/TrackingSpace/RightHandAnchor/RightOVRHand").GetComponent<OVRHand>();
            oculus_skeleton = GameObject.Find("OVRCameraRig/TrackingSpace/RightHandAnchor/RightOVRHand").GetComponent<OVRSkeleton>();
            _handJointsData = smoothHand.SmoothRightHandJoints;
        }
    }

    void Update()
    {
        if (hand.IsTracked && oculus_skeleton != null)
        {
            DetectPinch();
            DetectRotation();
        }
    }

    void DetectPinch()
    {
        Transform firstFingerTip = GetBoneTransform(firstPinchFinger);
        Transform secondFingerTip = GetBoneTransform(secondPinchFinger);
        if (firstFingerTip == null || secondFingerTip == null) return;

        float distance = Vector3.Distance(firstFingerTip.position, secondFingerTip.position);

        if (distance < pinchThreshold && !isPinching)
        {
            isPinching = true;
            initialPinchPosition = (firstFingerTip.position + secondFingerTip.position) / 2;
            lastPinchPosition = initialPinchPosition;
            CreateOrUpdatePinchSphere(initialPinchPosition);
        }
        else if (distance >= pinchThreshold && isPinching)
        {
            isPinching = false;
            DestroyPinchSphere();
            pinchSpeed = Vector3.zero;
        }

        if (isPinching)
        {
            Vector3 currentPinchPosition = (firstFingerTip.position + secondFingerTip.position) / 2;
            pinchMovement = currentPinchPosition - initialPinchPosition;
            pinchSpeed = (currentPinchPosition - lastPinchPosition) / Time.deltaTime;
            lastPinchPosition = currentPinchPosition;

            CreateOrUpdatePinchSphere(currentPinchPosition);
        }
    }

    void DetectRotation()
    {
        // 获取旋转手势的两个指尖
        Transform firstRotateTip = GetBoneTransform(firstRotateFinger);
        Transform secondRotateTip = GetBoneTransform(secondRotateFinger);
        if (firstRotateTip == null || secondRotateTip == null) return;

        // 检查指尖之间的距离
        float distance = Vector3.Distance(firstRotateTip.position, secondRotateTip.position);

        if (distance < rotationThreshold && !isRotating)
        {
            // 开始检测旋转
            isRotating = true;
            initialRotatePosition = (firstRotateTip.position + secondRotateTip.position) / 2;

            // 计算旋转轴（基于两个额外关节的中点）
            Transform joint1 = GetBoneTransform(firstJointBone);
            Transform joint2 = GetBoneTransform(secondJointBone);
            if (joint1 != null && joint2 != null)
            {
                Vector3 jointMidpoint = (joint1.position + joint2.position) / 2;
                rotationAxis = (initialRotatePosition - jointMidpoint).normalized;

                // 记录初始方向
                initialDirectionJoint1 = joint1.position - jointMidpoint;
                initialDirectionJoint2 = joint2.position - jointMidpoint;

                // 初始化上一帧角度
                previousAngle1 = 0f;
                previousAngle2 = 0f;
            }
            CreateOrUpdateRotationSphere(initialRotatePosition);
        }
        else if (distance >= rotationThreshold && isRotating)
        {
            // 停止旋转检测
            isRotating = false;
            rotationSpeed = 0;
            previousAngle1 = 0f;
            previousAngle2 = 0f;
            DestroyRotationSphere();
        }

        // 计算瞬时角速度
        if (isRotating)
        {
            Transform joint1 = GetBoneTransform(firstJointBone);
            Transform joint2 = GetBoneTransform(secondJointBone);

            if (joint1 != null && joint2 != null)
            {
                Vector3 jointMidpoint = (joint1.position + joint2.position) / 2;

                // 当前方向向量
                Vector3 currentDirectionJoint1 = joint1.position - jointMidpoint;
                Vector3 currentDirectionJoint2 = joint2.position - jointMidpoint;

                // 计算当前角度
                float currentAngle1 = Vector3.SignedAngle(initialDirectionJoint1, currentDirectionJoint1, rotationAxis);
                float currentAngle2 = Vector3.SignedAngle(initialDirectionJoint2, currentDirectionJoint2, rotationAxis);

                // 计算瞬时角速度
                float angularVelocity1 = (currentAngle1 - previousAngle1) / Time.deltaTime;
                float angularVelocity2 = (currentAngle2 - previousAngle2) / Time.deltaTime;

                // 更新上一帧角度
                previousAngle1 = currentAngle1;
                previousAngle2 = currentAngle2;

                // 计算平均瞬时角速度
                rotationSpeed = (angularVelocity1 + angularVelocity2) / 2.0f / 360.0f;

                Debug.Log($"Rotation Speed: {rotationSpeed} degrees/second");
            }
            CreateOrUpdateRotationSphere(initialRotatePosition);
        }
    }

    Transform GetBoneTransform(OVRSkeleton.BoneId boneId)
    {
        if (!UseSmoothHand)
        {
            foreach (var bone in oculus_skeleton.Bones)
            {
                if (bone.Id == boneId) return bone.Transform;
            }
        }
        else
        {
            for (int i = 0; i < oculus_skeleton.Bones.Count; i++)
            {
                OVRBone bone = oculus_skeleton.Bones[i];
                if (bone.Id == boneId)
                {
                    return _handJointsData[i];
                }
            }
        }
        return null;
    }

    void CreateOrUpdatePinchSphere(Vector3 position)
    {
        if (pinchSphere == null && RenderPinchSphere)
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

        if (pinchSphere != null)
            pinchSphere.transform.position = position;
    }

    void DestroyPinchSphere()
    {
        if (pinchSphere != null)
        {
            Destroy(pinchSphere);
            pinchSphere = null;
        }
    }

    void CreateOrUpdateRotationSphere(Vector3 position)
    {
        if (rotationSphere == null && RenderRotationSphere)
        {
            rotationSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            rotationSphere.transform.localScale = Vector3.one * (2 * pinchRadius);

            // Create a transparent material
            Material transparentMaterial = new Material(Shader.Find("Standard"));
            transparentMaterial.color = new Color(0, 0, 1, 0.2f); // Semi-transparent blue
            transparentMaterial.SetFloat("_Mode", 3); // Enable transparency mode
            transparentMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            transparentMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            transparentMaterial.SetInt("_ZWrite", 0);
            transparentMaterial.DisableKeyword("_ALPHATEST_ON");
            transparentMaterial.EnableKeyword("_ALPHABLEND_ON");
            transparentMaterial.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            transparentMaterial.renderQueue = 3000;

            rotationSphere.GetComponent<Renderer>().material = transparentMaterial;
        }
        if (rotationSphere != null)
            rotationSphere.transform.position = position;
    }

    void DestroyRotationSphere()
    {
        if (rotationSphere != null)
        {
            Destroy(rotationSphere);
            rotationSphere = null;
        }
    }
}
