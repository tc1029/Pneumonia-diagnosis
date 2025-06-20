# Pneumonia X-ray Image Classifier (Colab-ready)
# Author: [Your Name Here]
# Description: Classify chest X-rays into NORMAL, COVID-19, PNEUMONIA using VGG16 transfer learning

# ===============================
# ✅ Step 1: 掛載 Google Drive（可略過）
# ===============================
from google.colab import drive
import os

# drive.mount('/content/drive', force_remount=True)  # 部署時不需掛載 Google Drive

# 設定資料目錄（部署時建議預放在 huggingface Spaces 本地資料夾）
data_dir = 'chest_xray_split'

# ===============================
# ✅ Step 2: 建立資料載入器
# ===============================
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)
val_generator = val_test_datagen.flow_from_directory(
    os.path.join(data_dir, 'val'),
    target_size=img_size,
    ba
