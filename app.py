import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 📁 Görsel veri klasörü
base_dir = r"C:\Users\sudea\DeepLearningBrainCancer\dataset\Brain_Cancer"

# 🧪 Veri ön işleme: normalleştirme ve validation ayırma
img_size = (150, 150)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 🧠 Az Katmanlı CNN Modeli
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),  # 1. katman
    MaxPooling2D(2, 2),                                                # 2. katman
    Flatten(),                                                         # 3. katman
    Dense(32, activation='relu'),                                      # 4. katman
    Dense(3, activation='softmax')                                     # 5. katman (output)
])

# ⚙️ Modeli derle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🏋️‍♀️ Eğitimi başlat
history = model.fit(train_data, validation_data=val_data, epochs=30)

# 🎯 Doğrulama seti üzerindeki başarı
val_loss, val_acc = model.evaluate(val_data)
print(f"\n📊 Validation Accuracy: {val_acc:.4f}")

# 📈 Accuracy ve Loss grafiklerini çiz
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Doğruluk (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kayıp')
plt.plot(history.history['val_loss'], label='Doğrulama Kayıp')
plt.title('Kayıp (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.tight_layout()
plt.show()
