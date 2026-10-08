import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers

model = tf.keras.models.load_model("my_model.keras")
test_ds = tf.keras.utils.image_dataset_from_directory(
    r"C:\Users\jorge\OneDrive\Desktop\CNN\data\Testing",
        label_mode = 'categorical',
        batch_size = 32,
        image_size = (180,180),
        shuffle = False
    )
class_names = test_ds.class_names
norm_layer = layers.Rescaling(1/255)
test_ds = test_ds.map(lambda x, y: (norm_layer(x), y))
y_true_onehot = np.concatenate([labels for images, labels in test_ds], axis=0)  # shape (total, 4)
y_true = np.argmax(y_true_onehot, axis=1)  # shape (total,) — flat list of indices
y_pred = np.argmax(model.predict(test_ds), axis = 1)
print(classification_report(y_true, y_pred, target_names=class_names))
print(confusion_matrix(y_true, y_pred))

  
    

