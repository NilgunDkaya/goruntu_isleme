# 🧠 Face Mask Detection – Transfer Learning

## 📌 Proje Özeti

Bu proje kapsamında, insan yüzlerinde maske takılı olup olmadığını tespit eden bir görüntü sınıflandırma modeli geliştirilmiştir. **Transfer learning** yöntemiyle ResNet50 modeli kullanılmıştır.

---

## 🛠️ Kullanılan Model Mimarisi

- **Base Model:** ResNet50 (weights='imagenet', include_top=False, frozen)
- **Eklenen Katmanlar:**
  - Flatten
  - Dense (128 units, ReLU)
  - Dropout (0.5)
  - Dense (1 unit, Sigmoid activation)

```python
# Model Katmanları (Özet)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

📊 Eğitim Süreci ve Metrikler
Veri Seti: Face Mask Detection Dataset (with_mask & without_mask)

Data Split: %80 eğitim, %20 doğrulama

Görsel Boyutu: 224 x 224 px

Normalization: 0-1 arası

Loss Function: binary_crossentropy

Optimizer: Adam (lr=0.0001)

Epochs: 10

Batch Size: 32

🎯 Eğitim Sonuçları

![Figure1](https://github.com/user-attachments/assets/7f6e3d13-75dd-4d5c-b892-1d1d544dacfb)

![Figure2](https://github.com/user-attachments/assets/4da0693d-1d9d-4bbe-8779-09717b6bb8dd)


