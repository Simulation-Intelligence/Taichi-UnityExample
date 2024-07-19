using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SkeletonRenderer : MonoBehaviour
{
    // visualize capsules
    public class CapsuleVisualization
    {
        private GameObject CapsuleGO;
        private OVRBoneCapsule BoneCapsule;
        private Vector3 capsuleScale;
        private MeshRenderer Renderer;
        private Material RenderMaterial;

        private float capsult_radius = 0.007f;
        
        public CapsuleVisualization(Transform begin, Transform end)
        {
            Vector3 BoneBegin = begin.position;
            Vector3 BoneEnd = end.parent.position;

            GameObject capsule = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            capsule.transform.localScale = new Vector3(capsult_radius * 2, Vector3.Distance(BoneBegin, BoneEnd) / 2, capsult_radius * 2);
            capsule.transform.position = (BoneBegin + BoneEnd) / 2;
            Vector3 direction = (BoneEnd - BoneBegin).normalized;
            capsule.transform.rotation = Quaternion.LookRotation(direction) * Quaternion.Euler(90, 0, 0);
        }
        
        public void Update()
        {
        }
    }

    // visualize skeletion line segments
    public class BoneVisualization
    {
        private GameObject BoneGO;
        private Transform BoneBegin;
        private Transform BoneEnd;
        private LineRenderer Line;
        private Material RenderMaterial;
        private const float LINE_RENDERER_WIDTH = 0.002f;
        
        public BoneVisualization(GameObject rootGO, Material renderMat, Transform begin, Transform end)
        {
            RenderMaterial = renderMat;

            BoneBegin = begin;
            BoneEnd = end;

            BoneGO = new GameObject(begin.name);
            BoneGO.transform.SetParent(rootGO.transform, false);

            Line = BoneGO.AddComponent<LineRenderer>();
            Line.sharedMaterial = RenderMaterial;
            Line.useWorldSpace = true;
            Line.positionCount = 2;

            Line.SetPosition(0, BoneBegin.position);
            Line.SetPosition(1, BoneEnd.position);

            Line.startWidth = LINE_RENDERER_WIDTH;
            Line.endWidth = LINE_RENDERER_WIDTH;
        }

        public void Update(float scale, bool shouldRender)
        {
            Line.SetPosition(0, BoneBegin.position);
            Line.SetPosition(1, BoneEnd.position);

            Line.startWidth = LINE_RENDERER_WIDTH;
            Line.endWidth = LINE_RENDERER_WIDTH;

            Line.enabled = shouldRender;
            Line.sharedMaterial = RenderMaterial;
        }
    }
}