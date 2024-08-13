using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using TMPro;
using Oculus.Interaction;

class UIManager : MonoBehaviour
{
    [SerializeField]
    private GameObject Mpm3DObject;
    private List<GameObject> createdObjectLists = new List<GameObject>(); 
    private GameObject selectedObject;
    
    // UI Components
    public GameObject UI_canvas;
    public Transform UI_anchor;
    private Vector3 canvas_anchor_offset;
    [SerializeField]
    private Camera sceneCamera;
    [SerializeField]
    private OVRHand[] oculus_hands;
    [SerializeField]
    private OVRSkeleton[] oculus_skeletons;
    public Button[] buttons;
    public GameObject mergePrompt;
    private bool isMerging = false;
    public Toggle[] toggles;
    public TMP_Dropdown[] dropdowns;
    public InputField[] inputFields;
    public TMP_InputField[] tmpInputFields;
    public GameObject[] parameterObjects;
    public TouchScreenKeyboard overlayKeyboard;

    void Start()
    {
        canvas_anchor_offset = UI_canvas.transform.position - UI_anchor.position;
        
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
        foreach (GameObject parameter in parameterObjects)
        {
            TMP_Text parameter_text = parameter.transform.Find("Name").GetComponent<TMP_Text>();
            Slider parameter_slider = parameter.GetComponentInChildren<Slider>();
            if (parameter_text != null && parameter_slider != null)
            {
                string initial_text = parameter_text.text;
                parameter_text.text = initial_text + ": " + parameter_slider.value.ToString("F2");
                parameter_slider.onValueChanged.AddListener((float value) => 
                {
                    parameter_text.text = initial_text + ": " + value.ToString("F2");
                });
            }
        }
    }
    
    void Update()
    {
        // Find the last grabbed object as the selected object for further manipulations
        if (createdObjectLists != null)
        {
            foreach (var createdObject in createdObjectLists)
            {
                var _grabbable = createdObject.GetComponent<Grabbable>();
                // if (_grabbable.SelectingPointsCount > 0)
                // {
                //     // Merge the object with another object
                //     if (isMerging)
                //     {
                //         GameObject objectToMerge = createdObject;
                //         if (selectedObject != null && objectToMerge != null && selectedObject != objectToMerge)
                //         {
                //             Mpm3DGaussian_part_multi mpm3DSimulation = selectedObject.GetComponent<Mpm3DGaussian_part_multi>();
                //             if (mpm3DSimulation != null)
                //             {
                //                 mpm3DSimulation.MergeAndUpdate(objectToMerge.GetComponent<Mpm3DGaussian_part_multi>());
                //                 Debug.Log("Merge object " + selectedObject.name + " with object " + objectToMerge.name);
                //                 isMerging = false;
                //                 mergePrompt.SetActive(false);
                //                 objectToMerge = null;
                //             }
                //         }              
                //     }
                //     else
                //     {
                //         selectedObject = createdObject;
                //         Debug.Log("Object " + selectedObject.name + " is selected");
                //     }
                //     break;
                // }
            }
        }
    }
    
    void CreateNewMpm3DObject()
    {   
        if (Mpm3DObject != null)
        {
            // Position and name
            Vector3 position = sceneCamera.transform.position + sceneCamera.transform.forward * 0.1f;
            position.x -= 0.2f;
            Quaternion rotation = Quaternion.LookRotation(sceneCamera.transform.forward);
            
            GameObject newMpm3DObject = Instantiate(Mpm3DObject, position, rotation);
            createdObjectLists.Add(newMpm3DObject);
            // Use the just created object as the selected object
            selectedObject = newMpm3DObject;
            newMpm3DObject.name = "Mpm3DObject_" + createdObjectLists.Count;
            
            // Store the object parameters when creating the object
            Mpm3DGaussian_part_multi mpm3DSimulation = newMpm3DObject.GetComponent<Mpm3DGaussian_part_multi>();
            
            // foreach (Toggle toggle in toggles)
            // {
            //     if (toggle.name == "Toggle_FixObject")
            //     {
            //         mpm3DSimulation.isFixed = toggle.isOn;
            //     }
            // }
            
            // foreach (TMP_Dropdown dropdown in dropdowns)
            // {
            //     if (dropdown.name == "Dropdown_MaterialType")
            //     {
            //         mpm3DSimulation.materialType = (Mpm3DGaussian_part_multi.MaterialType)Enum.Parse(typeof(Mpm3DGaussian_part_multi.MaterialType), dropdown.value.ToString());
            //     }
            //     if (dropdown.name == "Dropdown_PlasticityType")
            //     {
            //         mpm3DSimulation.plasticityType = (Mpm3DGaussian_part_multi.PlasticityType)Enum.Parse(typeof(Mpm3DGaussian_part_multi.PlasticityType), dropdown.value.ToString());
            //     }
            //     if (dropdown.name == "Dropdown_StressType")
            //     {
            //         mpm3DSimulation.stressType = (Mpm3DGaussian_part_multi.StressType)Enum.Parse(typeof(Mpm3DGaussian_part_multi.StressType), dropdown.value.ToString());
            //     }
            // }
            
            // foreach (GameObject parameter in parameterObjects)
            // {
            //     if (parameter.name == "Parameter_E")
            //     {
            //         mpm3DSimulation._E = parameter.GetComponentInChildren<Slider>().value;
            //     }
            //     else if (parameter.name == "Parameter_SigY")
            //     {
            //         mpm3DSimulation._SigY = parameter.GetComponentInChildren<Slider>().value;
            //     }
            //     else if (parameter.name == "Parameter_Nu")
            //     {
            //         mpm3DSimulation._nu = parameter.GetComponentInChildren<Slider>().value;
            //     }
            //     else if (parameter.name == "Parameter_ColideFactor")
            //     {
            //         mpm3DSimulation.colide_factor = parameter.GetComponentInChildren<Slider>().value;
            //     }
            // }
            
            // Debug.Log("Mpm3DObject" + newMpm3DObject.name + " is created and selected");
            // Debug.Log("isFixed: " + mpm3DSimulation.isFixed);
            // Debug.Log("materialType: " + mpm3DSimulation.materialType);
            // Debug.Log("plasticityType: " + mpm3DSimulation.plasticityType);
            // Debug.Log("stressType: " + mpm3DSimulation.stressType);
            // Debug.Log("E: " + mpm3DSimulation._E);
            // Debug.Log("SigY: " + mpm3DSimulation._SigY);
            // Debug.Log("nu: " + mpm3DSimulation._nu);
            // Debug.Log("colide_factor: " + mpm3DSimulation.colide_factor);
        }
    }
    
    void OnButtonClick(Button button)
    {
        Debug.Log(button.name + " was clicked!");
        
        // Create a new object in the scene
        if (button.name == "Button_CreateObject")
        {
            CreateNewMpm3DObject();
        }
        // Merge the object with another object
        if (button.name == "Button_MergeObject")
        {
            if (selectedObject != null)
            {
                if (button.GetComponentInChildren<TMP_Text>().text == "Merge")
                {
                    button.GetComponentInChildren<TMP_Text>().text = "Cancel Merging";
                    mergePrompt.SetActive(true);
                    mergePrompt.GetComponent<TMP_Text>().text = "Please select an object to merge";
                    isMerging = true;
                }
                else
                {
                    button.GetComponentInChildren<TMP_Text>().text = "Merge";
                    mergePrompt.SetActive(false);
                    isMerging = false;
                }
            }
        }
        // Reset the object with original parameters
        if (button.name == "Button_ResetObject")
        {
            if (selectedObject != null)
            {
                Mpm3DGaussian_part_multi mpm3DSimulation = selectedObject.GetComponent<Mpm3DGaussian_part_multi>();
                mpm3DSimulation.Reset();
            }
        }
        // Delete the object from the scene
    }
    
    void OnToggleValueChanged(Toggle toggle, bool isOn)
    {
        Debug.Log("Toggle " + toggle.name + " is " + (isOn ? "On" : "Off"));

        // Enable/Disable the edit mode
        if (toggle.name == "Toggle_EnterModelingMode")
        {
            Text toggle_label = toggle.GetComponentInChildren<Text>();
            toggle_label.text = isOn ? "Exit Modeling Mode" : "Enter Modeling Mode";
        }
        
        // Enable/Disable fix object in place
        if (toggle.name == "Toggle_FixObject")
        {
            
        }
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
    
    void OnTMPInputFieldSelect(TMP_InputField inputField)
    {
        Debug.Log(inputField.name + " was selected!");
        overlayKeyboard = TouchScreenKeyboard.Open(inputField.text, TouchScreenKeyboardType.ASCIICapable);
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
            // Set position and rotation of the UI canvas
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
    public void ShowSlectedObjectCanvas()
    {
        if (UI_canvas != null && oculus_hands[1].IsTracked && sceneCamera != null)
        {   
            // Open the UI canvas for the object just grabbed
            if (selectedObject != null)
            {
                Mpm3DGaussian_part_multi mpm3DSimulation = selectedObject.GetComponent<Mpm3DGaussian_part_multi>();
                if (mpm3DSimulation != null)
                {
                    // Set the toggle, dropdown, and slider values accordingly
                    foreach (Toggle toggle in toggles)
                    {
                        if (toggle.name == "Toggle_FixObject")
                        {
                            toggle.isOn = mpm3DSimulation.isFixed;
                        }
                    }
                    
                    foreach (TMP_Dropdown dropdown in dropdowns)
                    {
                        if (dropdown.name == "Dropdown_MaterialType")
                        {
                            dropdown.value = (int)mpm3DSimulation.materialType;
                        }
                        if (dropdown.name == "Dropdown_PlasticityType")
                        {
                            dropdown.value = (int)mpm3DSimulation.plasticityType;
                        }
                        if (dropdown.name == "Dropdown_StressType")
                        {
                            dropdown.value = (int)mpm3DSimulation.stressType;
                        }
                    }
                    
                    foreach (GameObject parameter in parameterObjects)
                    {
                        if (parameter.name == "Parameter_E")
                        {
                            parameter.GetComponentInChildren<Slider>().value = mpm3DSimulation._E;
                        }
                        else if (parameter.name == "Parameter_SigY")
                        {
                            parameter.GetComponentInChildren<Slider>().value = mpm3DSimulation._SigY;
                        }
                        else if (parameter.name == "Parameter_Nu")
                        {
                            parameter.GetComponentInChildren<Slider>().value = mpm3DSimulation._nu;
                        }
                        else if (parameter.name == "Parameter_ColideFactor")
                        {
                            parameter.GetComponentInChildren<Slider>().value = mpm3DSimulation.colide_factor;
                        }
                    }
                }
            }
            
            // Set position and rotation of the UI canvas
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

[System.Serializable]
public class ObjectParameters
{
    public enum MaterialType
    {
        Customize,
        Clay,
        Dough,
        Elastic_Material
    }
    public enum PlasticityType
    {
        Von_Mises,
        Drucker_Prager,
        Clamp,
        Elastic
    }
    
    public enum StressType
    {
        NeoHookean,
        Kirchhoff
    }
    public string name;
    public bool isFixed = false;
    public bool isInModelingMode = true;
    public MaterialType materialType = MaterialType.Customize;
    public PlasticityType plasticityType = PlasticityType.Von_Mises;
    public StressType stressType = StressType.NeoHookean;
    public float E = 1e6f;
    public float SigY = 1e5f;
    public float nu = 0.3f;
    public float colide_factor = 0.5f;
    public float friction_k = 0.4f;
    public float p_rho = 1000;
    public float min_clamp = 0.1f;
    public float max_clamp = 0.1f;
    public float friction_angle = 30;
}