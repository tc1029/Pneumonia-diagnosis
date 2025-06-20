import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 載入模型（請確保 best_model.h5 與此檔案在同一資料夾）
model = load_model("best_model.h5")
class_names = ['COVID', 'NORMAL', 'PNEUMONIA']

# 頁面標題
st.set_page_config(page_title="肺炎影像判讀系統", page_icon="🩻")
st.title("🩻 肺炎影像智慧判讀系統")
st.markdown("這是一個基於 VGG16 模型的 X 光影像分類器，用來判斷肺部是否為 COVID-19、一般肺炎，或正常狀態。")

# 上傳圖片介面
uploaded_file = st.file_uploader("請上傳胸腔 X 光影像 (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 顯示圖片
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='上傳的影像', use_column_width=True)

    # 按鈕觸發預測
    if st.button("🔍 預測"):
        # 前處理：縮放與轉陣列
        image_resized = image.resize((224, 224))
        img_array = img_to_array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 模型預測
        prediction = model.predict(img_array)[0]
        pred_index = np.argmax(prediction)
        pred_class = class_names[pred_index]
        confidence = float(prediction[pred_index]) * 100

        # 顯示結果
        st.success(f"🧠 模型預測為：**{pred_class}**（信心值：{confidence:.2f}%）")
else:
    st.info("請先上傳一張影像")
