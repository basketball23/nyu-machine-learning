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


#plotting the graphs as we go 
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     # The CIFAR labels happen to be arrays, 
#     # which is why you need the extra index
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

# Determining the original size of the test and train data
print(f"Train Images: {train_images.shape}")
print(f"Test Images: {test_images.shape}")
print(f"Train Label: {train_labels.shape}")
print(f"Test Label: {test_labels.shape}")


# Train Images: (50000, 32, 32, 3)
# Test Images: (10000, 32, 32, 3)
# Train Label: (50000, 1)
# Test Label: (10000, 1)

# [[[0.69803922 0.69019608 0.74117647]
#   [0.69803922 0.69019608 0.74117647]
#   [0.69803922 0.69019608 0.74117647]
#   ...
#   [0.66666667 0.65882353 0.70588235]
#   [0.65882353 0.65098039 0.69411765]
#   [0.64705882 0.63921569 0.68235294]]


print(train_images[30000])
print(train_labels[30000])

# Creating the CNN
cnn_model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Second Convolutional Block
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
cnn_model.summary() # 191,050 weights

#adding optimizer, loss, and metixs
cnn_model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

#training the cnn
history = cnn_model.fit(train_images, train_labels_one_hot, batch_size=64, epochs=20)

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
n = 4
for i in range(n * n):
    plt.subplot(n, n, i + 1)
    random_idx = np.random.randint(0, test_images.shape[0])
    plt.imshow(test_images[random_idx], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Label: {test_labels[random_idx]}, Prediction: {cnn_y_pred[random_idx]}")
    plt.tight_layout()

plt.show()