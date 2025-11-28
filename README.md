# Unity Opus-MT ONNX Example

This project demonstrates how to run **Opus-MT machine translation models** inside **Unity** using **ONNX Runtime** and a modified **SentencePiece** implementation.

> ⚠️ **Note:** This example currently supports **Windows only** because ONNX Runtime for Unity (CPU build) and the SentencePiece native plugin here are built for Windows.

---

## 📦 Setup

### 1. Convert Opus-MT Model to ONNX  
You have two options:

#### **Option A — Download Ready-Made ONNX Models**
Some Opus-MT models on Hugging Face already include exported ONNX versions.  
Download the following files:

- `encoder_model.onnx`
- `decoder_model.onnx`
- `decoder_with_past_model.onnx`

#### **Option B — Convert the Model Yourself**
Choose any Opus-MT language model from Hugging Face (e.g., `en → id`, `id → en`, etc.) and convert it manually to ONNX.

The required output files are:

- `encoder_model.onnx`
- `decoder_model.onnx`
- `decoder_with_past_model.onnx`

Place all three files inside Unity’s **`StreamingAssets`** folder.

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

## ☕ Support the Developer

If this helps you, consider supporting me:

[<img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" width="200">](https://www.buymeacoffee.com/lookbe)

---

## 🔗 References

### SentencePiece Library (Modified)
Original source:  
https://github.com/Foorcee/libSentencePiece

### ONNX Runtime for Unity
https://github.com/asus4/onnxruntime-unity
