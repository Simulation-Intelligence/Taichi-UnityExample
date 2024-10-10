using System;
using System.Linq;
using UnityEngine;
using System.Collections.Generic;
using Oculus.Interaction;
using Oculus.Interaction.Input;

public class MatScissors : MatTool
{
    public enum HandType
    {
        LeftHand,
        RightHand
    }
    public HandType handType;
    [SerializeField]
    private HandJointId _handJointId1, _handJointId2;
    private OVRHand oculus_hand;
    private OVRSkeleton oculus_skeleton;

    private Vector3 sphere11 = new Vector3(0, 0, 0);
    private float radii11 = 0.04f;
    private Vector3 sphere12 = new Vector3(0, 0, 1.0f);
    private float radii12 = 0.04f;
    // Note: Set sphere3 and radii3 to 0.0f to treat it as a cone in the system
    private Vector3 sphere13 = new Vector3(0, 0, 0);
    private float radii13 = 0.0f;

    private Vector3 sphere21 = new Vector3(0, 0, 0);
    private float radii21 = 0.04f;
    private Vector3 sphere22 = new Vector3(0, 0, 1.0f);
    private float radii22 = 0.04f;
    // Note: Set sphere3 and radii3 to 0.0f to treat it as a cone in the system
    private Vector3 sphere23 = new Vector3(0, 0, 0);
    private float radii23 = 0.0f;

    // Map between primitive index and joint index
    void Awake()
    {
        numPrimitives = 2;
        init_primitives = new Primitive[numPrimitives];

        // Initialize positions and radius
        init_primitives[0].sphere1 = sphere11;
        init_primitives[0].radii1 = radii11;
        init_primitives[0].sphere2 = sphere12;
        init_primitives[0].radii2 = radii12;
        init_primitives[0].sphere3 = sphere13;
        init_primitives[0].radii3 = radii13;

        init_primitives[1].sphere1 = sphere21;
        init_primitives[1].radii1 = radii21;
        init_primitives[1].sphere2 = sphere22;
        init_primitives[1].radii2 = radii22;
        init_primitives[1].sphere3 = sphere23;
        init_primitives[1].radii3 = radii23;
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
        // Update mat hand without hand tracking
        if (oculus_hand.IsTracked)
        {
            foreach (var bone in oculus_skeleton.Bones)
            {
                var jointId1 = _handJointId1.ToString().Replace("Hand", "Hand_");
                if (bone.Id == (OVRSkeleton.BoneId)Enum.Parse(typeof(OVRSkeleton.BoneId), jointId1))
                {
                    // Rotate 90 degrees around the x-axis to align with the hand joint
                    transform.position = bone.Transform.position;
                    transform.rotation = bone.Transform.rotation * _rotationOffset;

                    for (int i = 0; i < numPrimitives; i++)
                    {
                        // Update primitive position using the oculus Hand Joint Component
                        primitives[i].sphere1 = transform.TransformPoint(init_primitives[i].sphere1);
                        primitives[i].sphere2 = transform.TransformPoint(init_primitives[i].sphere2);
                        primitives[i].sphere3 = transform.TransformPoint(init_primitives[i].sphere3);

                        // Multiply by localScale to get the correct radius
                        primitives[i].radii1 = init_primitives[i].radii1 * transform.localScale.x;
                        primitives[i].radii2 = init_primitives[i].radii2 * transform.localScale.x;
                        primitives[i].radii3 = init_primitives[i].radii3 * transform.localScale.x;
                    }
                    break;
                }
            }
        }
    }
}