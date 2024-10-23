using System;
using UnityEngine;
using Oculus.Interaction;
using Oculus.Interaction.Input;

public class MatToolCapsule : MatTool
{
    private Vector3 sphere1 = new Vector3(-0.6f, 0, 0);
    private float radii1 = 0.05f;
    private Vector3 sphere2 = new Vector3(0.6f, 0, 0);
    private float radii2 = 0.05f;
    // Note: Set sphere3 and radii3 to 0.0f to treat it as a cone in the system
    private Vector3 sphere3 = new Vector3(0, 0, 0);
    private float radii3 = 0.0f;

    private HandJoint handJoint;
    public enum HandType
    {
        LeftHand,
        RightHand
    }
    public HandType handType;
    [SerializeField]
    private HandJointId _handJointId;
    private OVRHand oculus_hand;
    private OVRSkeleton oculus_skeleton;

    void Awake()
    {
        numPrimitives = 1;
        init_primitives = new Primitive[numPrimitives];

        // Initialize positions and radius
        init_primitives[0].sphere1 = sphere1;
        init_primitives[0].radii1 = radii1;
        init_primitives[0].sphere2 = sphere2;
        init_primitives[0].radii2 = radii2;
        init_primitives[0].sphere3 = sphere3;
        init_primitives[0].radii3 = radii3;

        // Oculus hands
        if (handType == HandType.LeftHand)
        {
            oculus_hand = GameObject.Find("OVRCameraRig/TrackingSpace/LeftHandAnchor/LeftOVRHand").GetComponent<OVRHand>();
            oculus_skeleton = GameObject.Find("OVRCameraRig/TrackingSpace/LeftHandAnchor/LeftOVRHand").GetComponent<OVRSkeleton>();
        }
        else if (handType == HandType.RightHand)
        {
            oculus_hand = GameObject.Find("OVRCameraRig/TrackingSpace/RightHandAnchor/RightOVRHand").GetComponent<OVRHand>();
            oculus_skeleton = GameObject.Find("OVRCameraRig/TrackingSpace/RightHandAnchor/RightOVRHand").GetComponent<OVRSkeleton>();
        }
    }

    protected override void UpdatePrimitives()
    {
        // Update Gameobject Transform
        if (oculus_hand.IsTracked)
        {
            var jointId = _handJointId.ToString().Replace("Hand", "Hand_");
            foreach (var bone in oculus_skeleton.Bones)
            {

                if (bone.Id == (OVRSkeleton.BoneId)Enum.Parse(typeof(OVRSkeleton.BoneId), jointId))
                {
                    // Rotate 90 degrees around the x-axis to align with the hand joint
                    transform.position = bone.Transform.position;
                    transform.rotation = bone.Transform.rotation * _rotationOffset;

                    for (int i = 0; i < numPrimitives; i++)
                    {
                        // Update primitive position using the oculus Hand Joint Component
                        UpdatePrimitive(ref primitives[i], init_primitives[i], transform);
                    }
                    break;
                }
            }
        }
    }
}