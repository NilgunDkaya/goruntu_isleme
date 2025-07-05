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



```python
# Model Katmanları (Özet)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions) ```



![Figure2](https://gith![Figure1](https://github.com/user-attachments/assets/76f1a7df-e4cb-4d25-b304-741783ff500e)
ub.com/user-attachments/assets/da8fc029-2919-4e60-b216-babbdc7b53bb)
![Figure1](https://github.com/user-attachments/assets/85ece279-8323-40eb-ad24-4c09eb8a78b6)
