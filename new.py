
def func():
    print("hello")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- SIMPLE LINEAR REGRESSION ----------
# Use only one feature: 'bmi' (index 2)
X_simple = X_scaled[:, 2].reshape(-1, 1)  # Reshape for sklearn

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Simple Linear Regression:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel('BMI (Standardized)')
plt.ylabel('Target')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

# ---------- MULTIPLE LINEAR REGRESSION ----------
# Train-test split using all features
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\nMultiple Linear Regression:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Plot predicted vs actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Multiple Linear Regression")
plt.grid(True)
plt.show()


cifar10


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Load and normalize CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation = 'softmax'))  # No activation (logits) since we use from_logits=True in loss

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)


#############################################


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))  # softmax for classification

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Object Detection: Predicting the class of a test image
def predict_image(image_index):
    # Get a random test image and normalize it
    image = x_test[image_index]
    image_input = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict the class of the image
    prediction = model.predict(image_input)
    predicted_class = np.argmax(prediction)  # Get the class with the highest probability

    # Display the image and prediction
    plt.imshow(image)
    plt.title(f"Predicted: {class_names[predicted_class]}")
    plt.axis('off')  # Hide axes
    plt.show()

# Test: Predict an image from the test set (you can change the index)
predict_image(11)

#########################################################


## data augmentation

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10

# Set random seed for reproducibility
tf.random.set_seed(42)

# Load and Normalize CIFAR-10 Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(x_train)

# Model WITHOUT Data Augmentation
model_no_aug = Sequential()
model_no_aug.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model_no_aug.add(MaxPooling2D(2,2))
model_no_aug.add(Conv2D(64, (3,3), activation='relu'))
model_no_aug.add(MaxPooling2D(2,2))
model_no_aug.add(Flatten())
model_no_aug.add(Dense(128, activation='relu'))
model_no_aug.add(Dense(10, activation='softmax'))

model_no_aug.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train WITHOUT augmentation
history_no_aug = model_no_aug.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

# Model WITH Data Augmentation
model_aug = Sequential()
model_aug.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model_aug.add(MaxPooling2D(2,2))
model_aug.add(Conv2D(64, (3,3), activation='relu'))
model_aug.add(MaxPooling2D(2,2))
model_aug.add(Flatten())
model_aug.add(Dense(128, activation='relu'))
model_aug.add(Dense(10, activation='softmax'))

model_aug.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train WITH augmentation
history_aug = model_aug.fit(datagen.flow(x_train, y_train, batch_size=32), validation_data=(x_test, y_test), epochs=10)

# Plot Accuracy and Loss Comparison
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_no_aug.history['accuracy'], label='Train Acc (No Aug)')
plt.plot(history_no_aug.history['val_accuracy'], label='Val Acc (No Aug)')
plt.plot(history_aug.history['accuracy'], label='Train Acc (Aug)')
plt.plot(history_aug.history['val_accuracy'], label='Val Acc (Aug)')
plt.title('Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history_no_aug.history['loss'], label='Train Loss (No Aug)')
plt.plot(history_no_aug.history['val_loss'], label='Val Loss (No Aug)')
plt.plot(history_aug.history['loss'], label='Train Loss (Aug)')
plt.plot(history_aug.history['val_loss'], label='Val Loss (Aug)')
plt.title('Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split
import yfinance as yf

# Download historical stock data
company = 'META'
start_date = '2010-01-01'
end_date = '2025-04-20'
df = yf.download(company, start=start_date, end=end_date, interval="1d", progress=False)


# Check if data was downloaded
if df.empty:
    print("No data downloaded. Please check ticker symbol, dates, or your internet connection.")
else:
    # Prepare input and target data
    data = df['Close'].values
    X = data[:-1].reshape(-1, 1, 1)  # shape = (samples, timesteps=1, features=1)
    y = data[1:]                     # target is next day's price

    # Split data into training and testing (80/20), preserving time order
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Build SimpleRNN model
    model = Sequential([
        SimpleRNN(16, activation='relu', input_shape=(1, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Make predictions
    predictions = model.predict(X_test)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title(f'{company} Stock Price Prediction using SimpleRNN (1-Day Lag)')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
