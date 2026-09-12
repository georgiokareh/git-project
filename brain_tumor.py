import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
tf.random.set_seed(42)
train_ds = tf.keras.utils.image_dataset_from_directory(
    r"C:\Users\jorge\OneDrive\Desktop\CNN\data\Training",
    label_mode = 'categorical',
    batch_size = 32,
    image_size = (180,180),
    validation_split = 0.2,
    subset = "training",
    seed = 42
)
#get images from a folder
cv_ds = tf.keras.utils.image_dataset_from_directory(
    r"C:\Users\jorge\OneDrive\Desktop\CNN\data\Training",
    label_mode = 'categorical',
    batch_size = 32,
    image_size = (180,180),
    validation_split = 0.2,
    subset = "validation", 
    seed = 42 # to decide the way it the images are shuffled
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    r"C:\Users\jorge\OneDrive\Desktop\CNN\data\Testing",
        label_mode = 'categorical',
        batch_size = 32,
        image_size = (180,180),
    )
class_names = train_ds.class_names
print(class_names)
norm_layer = layers.Rescaling(1/255)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
# to add data to the training set
train_ds = train_ds.map(lambda x, y: (data_augmentation(norm_layer(x), training=True), y))
cv_ds = cv_ds.map(lambda x, y: (norm_layer(x), y))
test_ds = test_ds.map(lambda x, y: (norm_layer(x), y))
plt.figure(figsize=(10,8))
for images, labels in train_ds.take(1): #takes a random batch of images from folder
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis('off')
plt.show()
model = models.Sequential(
    [
        tf.keras.Input(shape=(180, 180, 3)),
        layers.Conv2D(
            filters = 32,
            kernel_size = (3,3),
            activation = 'relu',
        ),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(
            filters = 64,
            kernel_size = (3,3),
            activation = 'relu',
        ),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(
            filters = 128,
            kernel_size = (3,3),
            activation = 'relu',
        ),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation = 'relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation ='linear')



    ]
)
model.compile(
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
    optimizer = tf.keras.optimizers.Adam(0.001), #grad descent
    metrics = ['accuracy']
)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# stop early in the model.fit process
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='my_model.keras', monitor='val_loss', save_best_only=True)

history = model.fit(train_ds, validation_data=cv_ds, epochs=20, callbacks=[early_stop, checkpoint]) 
#returns an object inside of which is a dictionary

fig, ax = plt.subplots(1, 2, figsize=(12,5))
# to get the x axes to show two separate comps
ax[0].plot(history.history['accuracy'], label='Train accuracy')
ax[0].plot(history.history['val_accuracy'], label='Val accuracy')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel("acc")
ax[0].set_title('accuracy vs val_accuracy per epoch')
ax[1].plot(history.history['loss'], label = 'loss')
ax[1].plot(history.history['val_loss'], label = 'val_loss')
ax[1].set_xlabel('epoch')
ax[1].set_title('loss vs val_loss per epoch')
ax[1].set_ylabel("loss")
ax[0].legend()
ax[1].legend()
plt.show()