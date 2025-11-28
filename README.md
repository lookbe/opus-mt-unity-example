# Unity Opus-MT ONNX Example

This project demonstrates how to run **Opus-MT machine translation models** inside **Unity** using **ONNX Runtime** and a modified **SentencePiece** implementation.

---

## 📦 Setup

### 1. Convert Opus-MT Model to ONNX
Choose any Opus-MT language model from Hugging Face (e.g., `en → id`, `id → en`, etc.) and convert it to ONNX format.

After conversion, you should obtain:

- `encoder_model.onnx`
- `decoder_model.onnx`
- `decoder_with_past_model.onnx`

Place all three files inside Unity’s **StreamingAssets** folder.

---

### 2. Download Tokenizer Files
From the same Opus-MT model repository, download:

- `source.spm`
- `target.spm`
- `vocab.json`

Place these three files **directly inside `StreamingAssets`** as well.

---

### 3. Run the Example Scene
Open the provided **example Unity scene** and press play.  
The script will automatically load the ONNX models and SentencePiece tokenizer from StreamingAssets.

---

## ▶️ Demo Video

Watch the translation demo here:  
https://www.youtube.com/watch?v=phkIhTfa2o4

---

## 🔗 References

### SentencePiece Library (Modified)
Original source:  
https://github.com/Foorcee/libSentencePiece

### ONNX Runtime for Unity
https://github.com/asus4/onnxruntime-unity

---
