using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PinchGesture : MonoBehaviour
{
    public enum FingerType
    {
        Thumb,
        Index,
        Middle,
        Ring,
        Pinky
    }

    public OVRHand hand;
    public OVRSkeleton handSkeleton;

    // 可以在编辑器中选择手指
    public FingerType firstFinger = FingerType.Thumb;
    public FingerType secondFinger = FingerType.Middle;

    [HideInInspector] public bool isPinching = false;
    [HideInInspector] public Vector3 initialPinchPosition;
    [HideInInspector] public Vector3 pinchMovement;

    [HideInInspector] public Vector3 lastPinchPosition; // 上一次捏合位置
    [HideInInspector] public float pinchSpeed = 0.0f; // 捏合移动的速度

    // 捏合检测的阈值
    public float pinchThreshold = 0.02f;

    void Update()
    {
        // 检测右手的捏合
        if (hand.IsTracked && handSkeleton != null)
        {
            DetectPinch(handSkeleton);
        }
    }

    void DetectPinch(OVRSkeleton handSkeleton)
    {
        // 获取用户选择的两根手指
        Transform firstFingerTip = GetFingerTransform(handSkeleton, firstFinger);
        Transform secondFingerTip = GetFingerTransform(handSkeleton, secondFinger);

        // 计算两指间的距离
        float distance = Vector3.Distance(firstFingerTip.position, secondFingerTip.position);

        // 判断是否捏合
        if (distance < pinchThreshold && !isPinching)
        {
            // 开始捏合
            isPinching = true;
            initialPinchPosition = (firstFingerTip.position + secondFingerTip.position) / 2; // 获取捏合的初始位置
            lastPinchPosition = initialPinchPosition; // 初始化捏合位置
            Debug.Log("Pinch started at position: " + initialPinchPosition);
        }
        else if (distance >= pinchThreshold && isPinching)
        {
            // 结束捏合
            isPinching = false;
            pinchSpeed = 0.0f; // 捏合结束时速度归零
            Debug.Log("Pinch ended");
        }

        // 如果正在捏合，获取当前的捏合位置和移动方向、速度
        if (isPinching)
        {
            Vector3 currentPinchPosition = (firstFingerTip.position + secondFingerTip.position) / 2;
            pinchMovement = currentPinchPosition - initialPinchPosition; // 计算移动的方向

            // 计算速度
            float distanceMoved = Vector3.Distance(currentPinchPosition, lastPinchPosition);
            float deltaTime = Time.deltaTime; // 获取每帧时间
            pinchSpeed = distanceMoved / deltaTime; // 移动的速度

            // 更新上一次位置
            lastPinchPosition = currentPinchPosition;

            Debug.Log("Pinch movement direction: " + pinchMovement);
            Debug.Log("Pinch movement speed: " + pinchSpeed);
        }
    }

    // 根据枚举获取指定手指的Transform
    Transform GetFingerTransform(OVRSkeleton handSkeleton, FingerType fingerType)
    {
        switch (fingerType)
        {
            case FingerType.Thumb:
                return handSkeleton.Bones[(int)OVRPlugin.BoneId.Hand_ThumbTip].Transform;
            case FingerType.Index:
                return handSkeleton.Bones[(int)OVRPlugin.BoneId.Hand_IndexTip].Transform;
            case FingerType.Middle:
                return handSkeleton.Bones[(int)OVRPlugin.BoneId.Hand_MiddleTip].Transform;
            case FingerType.Ring:
                return handSkeleton.Bones[(int)OVRPlugin.BoneId.Hand_RingTip].Transform;
            case FingerType.Pinky:
                return handSkeleton.Bones[(int)OVRPlugin.BoneId.Hand_PinkyTip].Transform;
            default:
                return null;
        }
    }
}
