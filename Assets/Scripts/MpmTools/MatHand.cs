using System;
using UnityEngine;
using System.Collections.Generic;
using Oculus.Interaction;
using Oculus.Interaction.Input;

public class MatHand : MatTool
{
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
        // Initialization with hand_mat.json
        LoadPrimitivesFromJson("Prefabs/Tools/hand_mat");

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

        // Oculus hands
        // if (handType == HandType.LeftHand)
        // {
        //     if (oculus_hand == null | oculus_skeleton == null)
        //     {
        //         oculus_hand = GameObject.Find("OVRCameraRig/TrackingSpace/LeftHandAnchor/LeftOVRHand").GetComponent<OVRHand>();
        //         oculus_skeleton = GameObject.Find("OVRCameraRig/TrackingSpace/LeftHandAnchor/LeftOVRHand").GetComponent<OVRSkeleton>();
        //     }
        // }
        // else if (handType == HandType.RightHand)
        // {
        //     if (oculus_hand == null | oculus_skeleton == null)
        //     {
        //         oculus_hand = GameObject.Find("OVRCameraRig/TrackingSpace/RightHandAnchor/RightOVRHand").GetComponent<OVRHand>();
        //         oculus_skeleton = GameObject.Find("OVRCameraRig/TrackingSpace/RightHandAnchor/RightOVRHand").GetComponent<OVRSkeleton>();
        //     }
        // }
    }

    protected override void UpdatePrimitives()
    {
        // Update Gameobject Transform
        if (oculus_hand.IsTracked)
        {
            foreach (var bone in oculus_skeleton.Bones)
            {
                var jointId = _handJointId.ToString().Replace("Hand", "Hand_");
                if (bone.Id == (OVRSkeleton.BoneId)Enum.Parse(typeof(OVRSkeleton.BoneId), jointId))
                {
                    transform.position = bone.Transform.position;
                    transform.rotation = bone.Transform.rotation;
                    
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
    
    // protected override void UpdatePrimitives()
    // {
    //     if (oculus_hand.IsTracked && oculus_hand.HandConfidence == OVRHand.TrackingConfidence.High)
    //     {
    //         int numBones = oculus_skeleton.Bones.Count;
    //         if (numBones > 0)
    //         {
    //             for (int j = 0; j < numBones; j++)
    //             {
    //                 OVRBone bone = oculus_skeleton.Bones[j];
                    
    //             }
    //         }
    //     }
    // }
}