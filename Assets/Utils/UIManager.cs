using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using TMPro;

class UIManager : MonoBehaviour
{
    public GameObject UI_canvas;
    public Transform UI_anchor;
    private Vector3 canvas_anchor_offset;
    [SerializeField]
    private Camera sceneCamera;
    [SerializeField]
    private OVRHand[] oculus_hands;
    [SerializeField]
    private OVRSkeleton[] oculus_skeletons;
    [SerializeField]
    private Button[] buttons;
    [SerializeField]
    private Toggle[] toggles;
    [SerializeField]
    private TMP_Dropdown[] dropdowns;
    [SerializeField]
    private InputField[] inputFields;
    [SerializeField]
    private TMP_InputField[] tmpInputFields;
    [SerializeField]
    public GameObject[] privateparameterObjects;
    
    private TouchScreenKeyboard overlayKeyboard;
    public static string inputText = "";

    void Start()
    {
        foreach (Button button in buttons)
        {
            button.onClick.AddListener(() => OnButtonClick(button));
        }

        foreach (Toggle toggle in toggles)
        {
            toggle.onValueChanged.AddListener((bool isOn) => OnToggleValueChanged(toggle, isOn));
        }
        
        foreach (TMP_Dropdown dropdown in dropdowns)
        {
            dropdown.onValueChanged.AddListener((int value) => OnDropdownValueChanged(dropdown, value));
        }

        foreach (InputField inputField in inputFields)
        {
            inputField.onValueChanged.AddListener((string value) => OnInputFieldSelect(inputField));
        }

        foreach (TMP_InputField tmpInputField in tmpInputFields)
        {
           tmpInputField.onSelect.AddListener((string value) => OnTMPInputFieldSelect(tmpInputField));
        }

        canvas_anchor_offset = UI_canvas.transform.position - UI_anchor.position;
    }

    void Update()
    {
      
    }

    void OnTMPInputFieldSelect(TMP_InputField inputField)
    {
        Debug.Log(inputField.name + " was selected!");
        overlayKeyboard = TouchScreenKeyboard.Open(inputField.text, TouchScreenKeyboardType.ASCIICapable);
    }

    void OnButtonClick(Button button)
    {
        Debug.Log(button.name + " was clicked!");
        if (button.name == "Button_CreateObject")
        {
            Debug.Log(button.name + " was clicked!");
        }

        if (button.name == "Button_ResetObject")
        {
            Debug.Log(button.name + " was clicked!");
        }
    }

    void OnToggleValueChanged(Toggle toggle, bool isOn)
    {
        Debug.Log("Toggle " + toggle.name + " is " + (isOn ? "On" : "Off"));
    }

    void OnDropdownValueChanged(TMP_Dropdown dropdown, int value)
    {

        Debug.Log(dropdown.name + " selected: " + dropdown.options[value].text);
    }

    void OnInputFieldSelect(InputField inputField)
    {
        Debug.Log(inputField.name + " was selected!");
        overlayKeyboard = TouchScreenKeyboard.Open(inputField.text, TouchScreenKeyboardType.Default);
    }

    public void HideCanvas()
    {
        if (UI_canvas != null)
        {
            UI_canvas.SetActive(false);
        }
    }
    
    public void ShowCanvas()
    {
        if (UI_canvas != null && oculus_hands[1].IsTracked && sceneCamera != null)
        {
            Vector3 handThumbTipPosition = Vector3.zero;

            foreach (var b in oculus_skeletons[1].Bones)
            {
                if (b.Id == OVRSkeleton.BoneId.Hand_ThumbTip)
                {
                    handThumbTipPosition = b.Transform.position;
                    break;
                }
            }
            UI_canvas.SetActive(true);
            UI_anchor.position = handThumbTipPosition + sceneCamera.transform.forward * 0.2f;
            UI_anchor.rotation = Quaternion.LookRotation(sceneCamera.transform.forward);
            UI_canvas.transform.position = UI_anchor.position + canvas_anchor_offset;
            UI_canvas.transform.rotation = UI_anchor.rotation;
        }
    }

    void OnDestroy()
    {
        
    }
}