using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class Translate : MonoBehaviour
{
    public OpusMT bot;

    // Public variables to link the UI components in the Unity Inspector
    public TMP_Text chatLogText;
    public TMP_InputField chatInputField;

    private OpusMT.Status lastBotStatus = OpusMT.Status.Init;

    private void OnEnable()
    {
        if (bot != null)
        {
            bot.OnResponseGenerated += OnBotResponseGenerated;
            bot.OnStatusChanged += OnBotStatusChanged;

            OnBotStatusChanged(bot.status);
        }
        else
        {
            OnBotStatusChanged(OpusMT.Status.Ready);
        }
    }

    private void OnDisable()
    {
        if (bot != null)
        {
            bot.OnStatusChanged -= OnBotStatusChanged;
            bot.OnResponseGenerated -= OnBotResponseGenerated;
        }
    }

    void OnBotStatusChanged(OpusMT.Status status)
    {
        if (status == OpusMT.Status.Generate)
        {
            chatLogText.text = "";
        }

        if (status == OpusMT.Status.Ready)
        {
            chatInputField.enabled = true;

            chatInputField.Select();
            chatInputField.ActivateInputField();
        }
        else
        {
            chatInputField.enabled = false;
        }

        TextMeshProUGUI placeholder = chatInputField.placeholder as TextMeshProUGUI;
        if (placeholder != null)
        {
            placeholder.text = status.ToString();
        }

        lastBotStatus = status;
    }

    void OnBotResponseGenerated(string response)
    {
        chatLogText.text = response;
    }

    void Start()
    {
        chatInputField.onSubmit.AddListener(SendInputMessage);
    }

    public void SendInputMessage(string message)
    {
        SendMessage();
    }

    public void SendMessage()
    {
        string message = chatInputField.text;
        if (!string.IsNullOrWhiteSpace(message))
        {
            chatInputField.text = "";
            chatInputField.ActivateInputField();
        }

        if (bot != null)
        {
            bot.Prompt(message);
        }
    }
}
