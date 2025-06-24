# KelantanLingoAI 🗣️🇲🇾

**KelantanLingoAI** is an AI-powered API that translates sentences from the **Kelantanese dialect** to **Modern Standard Malay**. This project fine-tunes a T5 Transformer model using domain-specific data to preserve and enhance accessibility to regional Malay dialects.

---

## ✨ Features

- Translate Kelantan dialect → Standard Malay
- Fine-tuned on real-world conversational data
- FastAPI backend with Transformers and PyTorch
- Deployed via Docker on Hugging Face Spaces
- Designed to integrate into mobile apps (e.g. React Native)

---

## 📦 API Endpoint

> **POST** `/translate`

**Request JSON:**

```json
{
  "input": "Kawe nok gi make nasik lauk dale kedai tu"
}
```
