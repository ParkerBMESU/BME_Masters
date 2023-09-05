import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Set the image directory path
image_directory = 'C:/Users/Irshaad Parker/Desktop/MEngSc Biomedical Engineering 2022/PlateClassifier/'

# Set the image size
SIZE = 150

# Define the labels and categories
labels = ['treated', 'untreated']
categories = ['1x', '2x', '5x', '10x', '25x', '50x']

# Create empty lists to store the data and labels
dataset = []
label_treated = []
label_extent = []

# Load the treated images
for category in categories:
    image_path = os.path.join(image_directory, 'Training', category, 'treated')
    images = os.listdir(image_path)
    for image_name in images:
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(os.path.join(image_path, image_name))
            image = Image.fromarray(image, 'RGB')
            image = image.resize((SIZE, SIZE))
            dataset.append(np.array(image))
            label_treated.append(0)  # Treated image
            label_extent.append(categories.index(category))  # Category index for extent

# Load the untreated images
image_path = os.path.join(image_directory, 'Training', 'untreated')
images = os.listdir(image_path)
for image_name in images:
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(os.path.join(image_path, image_name))
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label_treated.append(1)  # Untreated image
        label_extent.append(0)  # Category index 0 for untreated (1x)

# Convert the data and labels to NumPy arrays
dataset = np.array(dataset)
label_treated = np.array(label_treated)
label_extent = np.array(label_extent)

# Split the data into training and testing sets for treated vs untreated classification
X_train_treated, X_test_treated, y_train_treated, y_test_treated = train_test_split(
    dataset, label_treated, test_size=0.2, random_state=42
)

# Split the data into training and testing sets for extent classification
X_train_extent, X_test_extent, y_train_extent, y_test_extent = train_test_split(
    dataset, label_extent, test_size=0.2, random_state=42
)

# Normalize the image data
X_train_treated = X_train_treated.astype('float32') / 255.0
X_test_treated = X_test_treated.astype('float32') / 255.0

X_train_extent = X_train_extent.astype('float32') / 255.0
X_test_extent = X_test_extent.astype('float32') / 255.0

# Convert the labels to categorical format
y_train_treated = to_categorical(y_train_treated, num_classes=2)
y_test_treated = to_categorical(y_test_treated, num_classes=2)

y_train_extent = to_categorical(y_train_extent, num_classes=len(categories))
y_test_extent = to_categorical(y_test_extent, num_classes=len(categories))

# Define the model architecture for treated vs untreated classification
model_treated = Sequential()
model_treated.add(Conv2D(32, (3, 3), input_shape=(SIZE, SIZE, 3)))
model_treated.add(Activation('relu'))
model_treated.add(MaxPooling2D(pool_size=(2, 2)))

model_treated.add(Conv2D(32, (3, 3)))
model_treated.add(Activation('relu'))
model_treated.add(MaxPooling2D(pool_size=(2, 2)))

model_treated.add(Conv2D(64, (3, 3)))
model_treated.add(Activation('relu'))
model_treated.add(MaxPooling2D(pool_size=(2, 2)))

model_treated.add(Flatten())
model_treated.add(Dense(64))
model_treated.add(Activation('relu'))
model_treated.add(Dropout(0.5))
model_treated.add(Dense(2))  # Output layer with 2 units (treated vs untreated)
model_treated.add(Activation('softmax'))

model_treated.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model for treated vs untreated classification
history_treated = model_treated.fit(X_train_treated, y_train_treated, epochs=10, batch_size=32, validation_data=(X_test_treated, y_test_treated))
model_treated.save('C:/Users/Irshaad Parker/Desktop/MEngSc Biomedical Engineering 2022/Plate Classifier/model_treated_10epochs.h5')


# Define the model architecture for extent classification
model_extent = Sequential()
model_extent.add(Conv2D(32, (3, 3), input_shape=(SIZE, SIZE, 3)))
model_extent.add(Activation('relu'))
model_extent.add(MaxPooling2D(pool_size=(2, 2)))

model_extent.add(Conv2D(32, (3, 3)))
model_extent.add(Activation('relu'))
model_extent.add(MaxPooling2D(pool_size=(2, 2)))

model_extent.add(Conv2D(64, (3, 3)))
model_extent.add(Activation('relu'))
model_extent.add(MaxPooling2D(pool_size=(2, 2)))

model_extent.add(Flatten())
model_extent.add(Dense(64))
model_extent.add(Activation('relu'))
model_extent.add(Dropout(0.5))
model_extent.add(Dense(len(categories)))  # Output layer with units equal to the number of categories
model_extent.add(Activation('softmax'))

model_extent.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model for extent classification
history_extent = model_extent.fit(X_train_extent, y_train_extent, epochs=10, batch_size=32, validation_data=(X_test_extent, y_test_extent))
model_extent.save('C:/Users/Irshaad Parker/Desktop/MEngSc Biomedical Engineering 2022/Plate Classifier/model_extent_10epochs.h5')


# Evaluate the trained model for treated vs untreated classification
loss_treated, accuracy_treated = model_treated.evaluate(X_test_treated, y_test_treated)
print(f"Treated vs Untreated - Loss: {loss_treated}, Accuracy: {accuracy_treated}")

# Evaluate the trained model for extent classification
loss_extent, accuracy_extent = model_extent.evaluate(X_test_extent, y_test_extent)
print(f"Extent Classification - Loss: {loss_extent}, Accuracy: {accuracy_extent}")

# Plot model performance curves
plt.figure(figsize=(12, 6))

# Plot treated vs untreated classification accuracy
plt.subplot(1, 2, 1)
plt.plot(history_treated.history['accuracy'], label='Training Accuracy')
plt.plot(history_treated.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Treated vs Untreated Classification Model Performance')
plt.legend()

# Plot extent classification accuracy
plt.subplot(1, 2, 2)
plt.plot(history_extent.history['accuracy'], label='Training Accuracy')
plt.plot(history_extent.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Extent Classification Model Performance')
plt.legend()

plt.tight_layout()
plt.show()

# Set the image directory paths for test images
test_image_paths = [
    'C:/Users/Irshaad Parker/Desktop/MEngSc Biomedical Engineering 2022/Plate data/NAP and ATZ Experiments/ATZ/ProcessedimagestreatedATZ1000xiv/TreatedImagesATZ1000xiv_28.jpg',
    'C:/Users/Irshaad Parker/Desktop/MEngSc Biomedical Engineering 2022/Plate data/NAP and ATZ Experiments/ATZ/ProcessedimagestreatedATZ1000xiv/TreatedImagesATZ1000xiv_28.jpg',
    'C:/Users/Irshaad Parker/Desktop/MEngSc Biomedical Engineering 2022/Plate data/NAP and ATZ Experiments/ATZ/ProcessedimagestreatedATZ1000xiv/TreatedImagesATZ1000xiv_28.jpgg']



for idx, test_image_path in enumerate(test_image_paths):
    test_image = cv2.imread(test_image_path)
    test_image = Image.fromarray(test_image, 'RGB')
    test_image = test_image.resize((SIZE, SIZE))
    test_image = np.array(test_image)
    test_image = test_image.astype('float32') / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    # Predict the treated vs untreated class
    prediction_treated = model_treated.predict(test_image)
    treated_label = labels[np.argmax(prediction_treated)]

    # Plot the test image along with the classification and extent
    plt.figure(figsize=(8, 6))
    img_resized = cv2.resize(test_image[0], (0, 0), fx=0.25, fy=0.25)
    plt.imshow(img_resized)
    plt.title(f"Test Image {idx + 1}\nPredicted: {treated_label}")
    plt.axis('off')

    if treated_label == 'treated':
        # Predict the extent category
        prediction_extent = model_extent.predict(test_image)
        extent_index = np.argmax(prediction_extent)
        extent_category = categories[extent_index]
        print(extent_category)
        plt.text(10, SIZE + 20, f"Extent: {extent_category}", color='Black', fontsize=12)
    else:
        plt.text(10, SIZE + 20, "Extent: N/A", color='Red', fontsize=12)

    plt.savefig(f'results_test_image_{idx + 1}.png', bbox_inches='tight')
    plt.show()

# Plot the test image along with the classification and extent
plt.figure(figsize=(8, 6))
img_resized = cv2.resize(test_image[0], (0, 0), fx=0.25, fy=0.25)
plt.imshow(img_resized)
plt.title(f"Predicted: {treated_label}")
plt.axis('off')

test_image_path = 'C:/Users/Irshaad Parker/Desktop/MEngSc Biomedical Engineering 2022/Plate data/NAP and ATZ Experiments/ATZ/ProcessedimagestreatedATZ1000xiv/TreatedImagesATZ1000xiv_28.jpg'
test_image = cv2.imread(test_image_path)
test_image = Image.fromarray(test_image, 'RGB')
test_image = test_image.resize((SIZE, SIZE))
test_image = np.array(test_image)
test_image = test_image.astype('float32') / 255.0
test_image = np.expand_dims(test_image, axis=0)

# Predict the treated vs untreated class
prediction_treated = model_treated.predict(test_image)
treated_label = labels[np.argmax(prediction_treated)]



if treated_label == 'treated':
    # Predict the extent category
    prediction_extent = model_extent.predict(test_image)
    extent_index = np.argmax(prediction_extent)
    extent_category = categories[extent_index]
    plt.text(10, SIZE + 20, f"Extent: {extent_category}", color='Black', fontsize=12)
else:
    plt.text(10, SIZE + 20, "Extent: N/A", color='Red', fontsize=12)

plt.show()





