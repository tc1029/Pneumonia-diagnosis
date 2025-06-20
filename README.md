# 🩻 Pneumonia X-ray Classifier

A web application to classify chest X-ray images into:

- ✅ **NORMAL** – Healthy lungs  
- ⚠️ **PNEUMONIA** – Typical pneumonia infection  
- 🦠 **COVID-19** – COVID-19 related pneumonia  

This app is built using **Streamlit**, based on a VGG16 convolutional neural network trained on open Kaggle data. It allows medical learners or researchers to **upload an image and instantly receive an AI-based diagnosis**.

---

## 🚀 Live Demo

👉 [Try it on Hugging Face Spaces](https://your-space-name.hf.space) ← *(replace this after deployment)*

---

## 🧠 Features

- Pretrained **VGG16 + Dense classifier**  
- Upload `.jpg`, `.png`, or `.jpeg` X-ray images  
- Outputs class prediction and confidence scores  
- Displays sample misclassifications (coming soon)

---

## 🛠 Installation (Optional for Local Use)

If you wish to run this app locally:

```bash
git clone https://github.com/tc1029/Pneumonia-diagnosis.git
cd Pneumonia-diagnosis
pip install -r requirements.txt
streamlit run app.py
