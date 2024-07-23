using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

public class SkeletonRenderer
{
    private List<SegmentVisualization> _segmentVisualizations;
    private List<CapsuleVisualization> _capsuleVisualizations;

    public float Skeleton_capsule_radius = 0.01f;

    private void Start()
    {
        _segmentVisualizations = new List<SegmentVisualization>();
        _capsuleVisualizations = new List<CapsuleVisualization>();
    }

    public void Update()
    {
        for (int i = 0; i < _capsuleVisualizations.Count; i++)
        {
            //_capsuleVisualizations[i].Update(begin, end);
        }
    }

    // Visualize hand skeleton by capsules
    public class CapsuleVisualization
    {
        private GameObject CapsuleGO;
        private Vector3 BoneBegin;
        private Vector3 BoneEnd;
        
        private Vector3 capsuleScale;
        private MeshRenderer Renderer;
        private Material RenderMaterial;

        private float capsult_radius;
        
        public CapsuleVisualization(float radius)
        {
            capsult_radius = radius;
            CapsuleGO = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        }
        
        public void Update(Vector3 BoneBegin, Vector3 BoneEnd)
        {
            CapsuleGO.transform.localScale = new Vector3(capsult_radius * 2, Vector3.Distance(BoneBegin, BoneEnd) / 2, capsult_radius * 2);
            CapsuleGO.transform.position = (BoneBegin + BoneEnd) / 2;
            Vector3 direction = (BoneEnd - BoneBegin).normalized;
            CapsuleGO.transform.rotation = Quaternion.LookRotation(direction) * Quaternion.Euler(90, 0, 0);
        }
    }

    // Visualize hand skeletion by line segments
    public class SegmentVisualization
    {
        private GameObject BoneGO;
        private Transform BoneBegin;
        private Transform BoneEnd;
        private LineRenderer Line;
        private Material RenderMaterial;
        private const float LINE_RENDERER_WIDTH = 0.002f;
        
        public SegmentVisualization(GameObject rootGO, Material renderMat, Transform begin, Transform end)
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

            Line.sharedMaterial = RenderMaterial;
        }
    }
}

