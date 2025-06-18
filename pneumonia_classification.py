# Pneumonia X-ray Image Classifier (Colab-ready)
# Author: [Your Name Here]
# Description: Classify chest X-rays into NORMAL, COVID-19, PNEUMONIA using VGG16 transfer learning

# ===============================
# ✅ Step 1: 掛載 Google Drive
# ===============================
from google.colab import drive
import os

drive.mount('/content/drive', force_remount=True)

# 設定路徑（請依據你的 Drive 實際資料夾名稱）
data_dir = '/content/drive/MyDrive/hands-on/chest_xray_split'

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
    batch_size=batch_size,
    class_mode='categorical'
)
test_generator = val_test_datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ===============================
# ✅ Step 3: 建立模型（VGG16 + 自定分類層）
# ===============================
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in vgg_base.layers:
    layer.trainable = False

model = Sequential([
    vgg_base,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================
# ✅ Step 4: 模型訓練與保存
# ===============================
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

checkpoint_path = '/content/drive/MyDrive/hands-on/best_model.h5'
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop, model_checkpoint]
)

# 準確率 / 損失視覺化
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()

# ===============================
# ✅ Step 5: 模型評估與錯誤分析
# ===============================
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

# 測試集評估
loss, acc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")

# 混淆矩陣與分類報告
y_probs = model.predict(test_generator)
y_pred = np.argmax(y_probs, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=class_names))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 顯示錯誤影像
import matplotlib.pyplot as plt
errors = np.where(y_pred != y_true)[0]
print(f"Number of misclassified images: {len(errors)}")

for i in range(min(6, len(errors))):
    idx = errors[i]
    img_path = test_generator.filepaths[idx]
    img = plt.imread(img_path)
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {class_names[y_true[idx]]}, Pred: {class_names[y_pred[idx]]}")
    plt.axis('off')
    plt.show()
