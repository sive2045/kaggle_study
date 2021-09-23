import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

"""
MNIST convert test 
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# load data
train = pd.read_csv("c:/Users/Administrator/Documents/Workspace/kaggle_study/digit-recognizer/train.csv")
test = pd.read_csv('c:/Users/Administrator/Documents/Workspace/kaggle_study/digit-recognizer/test.csv')

features = train.drop('label', axis=1)
y_train = train['label']

# Train images
X_ = np.array(features)
X_train = X_.reshape(X_.shape[0],28,28)
X_train = np.expand_dims(X_train, -1)

# Test images
X_test = np.array(test)
X_test = X_test.reshape(X_test.shape[0],28,28)
X_test = np.expand_dims(X_test, -1)

print("train shape :", X_train.shape, "test shape :", X_test.shape)

# Scale images to the [0, 1] range
x_train = X_train.astype("float32") / 255
x_test = X_test.astype("float32") / 255


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)

# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()

# Train the model
batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# predict
preds = np.argmax(model.predict(X_test), axis=1)

sub = pd.read_csv('c:/Users/Administrator/Documents/Workspace/kaggle_study/digit-recognizer/sample_submission.csv')
sub['Label'] = preds
sub.to_csv('c:/Users/Administrator/Documents/Workspace/kaggle_study/digit-recognizer/submission.csv', index=False)