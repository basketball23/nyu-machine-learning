import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras import datasets

# Loading the Dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Scale pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# make labels one-hot representation so we can use it for the CNN
train_labels_one_hot = tf.one_hot(train_labels, 10)[:, 0, :]
test_labels_one_hot = tf.one_hot(test_labels, 10)[:, 0, :]

# Categories that the dataset seperates the data by:
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
# Determining the original size of the test and train data
print(f"Train Images: {train_images.shape}")
print(f"Test Images: {test_images.shape}")
print(f"Train Label: {train_labels.shape}")
print(f"Test Label: {test_labels.shape}")
# Creating the CNN
cnn_model = Sequential([
    # Convolutions
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Convolutions
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Fully Connected Layers
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
# Printing a summary of the dataset
cnn_model.summary()

#adding optimizer, loss, and metixs
cnn_model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)
#training the cnn
history = cnn_model.fit(train_images, train_labels_one_hot, batch_size=64, epochs=100)

#Plotting the graph of the CNN's accuracy
plt.plot(history.history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks([])
plt.yticks([])
cnn_model.save('cifar_model.keras')

cnn_preds = cnn_model.predict(test_images)
cnn_y_pred = np.argmax(cnn_preds, axis=-1)

cnn_test_acc = np.mean(cnn_y_pred == test_labels)
print(f"Accuracy: {cnn_test_acc}")
plt.figure(figsize=(8, 8))
n = 2
for i in range(n * n):
    plt.subplot(n, n, i + 1)
    random_idx = np.random.randint(0, test_images.shape[0])
    plt.imshow(test_images[random_idx], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Label: {class_names[test_labels[random_idx].item()]}, Prediction: {class_names[cnn_y_pred[random_idx]]}")

plt.show()