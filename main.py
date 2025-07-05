import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import random

# GPU konfigürasyonu (opsiyonel)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Dataset path (Windows)
dataset_path = r"C:\Users\Nilgun\Desktop\Turkcell_GYK\Goruntu_Isleme\ödev\Face_Mask_Detection_Dataset"

# Data generator - %80 train, %20 val
train_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2,
                                   horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # freeze

# Üst katmanlar
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)  # binary classification

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Accuracy ve Loss grafik
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Loss')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title('Accuracy')
plt.show()

# Confusion Matrix ve Classification Report
val_generator.reset()
Y_pred = model.predict(val_generator)
y_pred = (Y_pred > 0.5).astype(int)
y_true = val_generator.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['without_mask','with_mask']))

# 5 örnek görsel üzerinde tahmin (tek plot içinde yan yana)
class_names = ['with_mask', 'without_mask']
sample_indices = random.sample(range(len(val_generator.filenames)), 5)

plt.figure(figsize=(20,5))  # geniş figür

for idx, i in enumerate(sample_indices):
    img_path = os.path.join(val_generator.directory, val_generator.filenames[i])
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred_prob = model.predict(img_array)[0][0]
    pred_label = class_names[int(pred_prob > 0.5)]

    # Gerçek label
    true_label_index = val_generator.classes[i]
    true_label = class_names[true_label_index]

    plt.subplot(1, 5, idx+1)
    plt.imshow(img)
    plt.title(f"Gerçek: {true_label}\nTahmin: {pred_label}\nProb: {pred_prob:.2f}")
    plt.axis('off')

plt.tight_layout()
plt.show()
