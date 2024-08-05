using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using TMPro;
using System.Threading.Tasks;

class UIManager : MonoBehaviour
{
    public Button[] buttons;
    public TMP_Dropdown[] dropdowns;
    public InputField[] inputFields;

    public TMP_InputField[] tmpInputFields;

    private TouchScreenKeyboard overlayKeyboard;
    public static string inputText = "";

    void Start()
    {
        foreach (Button button in buttons)
        {
            button.onClick.AddListener(() => OnButtonClick(button));
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

    void OnDropdownValueChanged(TMP_Dropdown dropdown, int value)
    {

        Debug.Log(dropdown.name + " selected: " + dropdown.options[value].text);
    }

    void OnInputFieldSelect(InputField inputField)
    {
        Debug.Log(inputField.name + " was selected!");
        overlayKeyboard = TouchScreenKeyboard.Open(inputField.text, TouchScreenKeyboardType.Default);
    }
}