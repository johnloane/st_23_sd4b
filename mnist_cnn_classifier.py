import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.layers import Flatten, Dropout
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import random
import requests
from PIL import Image
import cv2

np.random.seed(0)
num_samples = []
num_classes = 10
num_pixels = 784


def main():
    X_train, y_train, X_test, y_test = pre_process_data()
    model = create_leNet_model()
    print(model.summary())
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=200, verbose=1, shuffle=1)
    plot_performance(history)
    evaluate_model(model, X_test, y_test)
    img = test_model_with_image(model)
    visualize_convolution_output(model, img)


def visualize_convolution_output(model, img):
    layer1 = Model(inputs=model.layers[0].input, outputs = model.layers[0].output)
    layer2 = Model(inputs=model.layers[0].input, outputs = model.layers[2].output)
    visual_layer1 = layer1.predict(img)
    visual_layer2 = layer2.predict(img)
    plt.figure(figsize = (10, 6))
    for i in range(30):
        plt.subplot(6, 5, i+1)
        plt.imshow(visual_layer1[0, :, :, i], cmap = plt.get_cmap('jet'))
        plt.axis('off')
    plt.show()
    plt.figure(figsize = (10, 6))
    for i in range(15):
        plt.subplot(3, 5, i+1)
        plt.imshow(visual_layer2[0, :, :, i], cmap = plt.get_cmap('jet'))
        plt.axis('off')
    plt.show()



def test_model_with_image(model):
    url = 'https://colah.github.io/posts/2014-10-Visualizing-MNIST/img/mnist_pca/MNIST-p1815-4.png'
    response = requests.get(url, stream=True)
    print(response)
    img = Image.open(response.raw)
    plt.imshow(img)
    plt.show()
    img_array = np.asarray(img)
    resized = cv2.resize(img_array, (28,28))
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(grayscale)
    image = image/255
    image = image.reshape(1, 28, 28, 1)
    prediction = np.argmax(model.predict(image), axis=-1)
    print("Predicted digit: ", str(prediction))
    return image




def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=1)
    print("Test score: ", score[0])
    print("Test accuracy: ", score[1])


def plot_performance(h):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.legend(['loss', 'validation_loss'])
    plt.title("Loss")
    plt.xlabel('epoch')
    plt.show()


def create_leNet_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def one_hot_encode_labels(y_train, y_test):
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return y_train, y_test


def pre_process_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    label_shape_match(X_train, y_train, X_test, y_test)
    print_sample_images(X_train, y_train)
    print(num_samples)
    plot_num_samples(num_samples)
    y_train, y_test = one_hot_encode_labels(y_train, y_test)
    X_train, X_test = convert_zero_one(X_train, X_test)
    X_train, X_test = reshape_data(X_train, X_test)
    return X_train, y_train, X_test, y_test


def reshape_data(x_train, x_test):
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    return x_train, x_test

def convert_zero_one(x_train, x_test):
    x_train = x_train/255
    x_test = x_test/255
    return x_train, x_test


def plot_num_samples(samples):
    plt.figure(figsize=(12,4))
    plt.bar(range(0, num_classes), samples)
    plt.title("Distribution of the training data")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()


def print_sample_images(x_train, y_train):
    cols = 5
    fig, axs = plt.subplots(nrows = num_classes, ncols=cols, figsize=(5,10))
    fig.tight_layout()
    for i in range(cols):
        for j in range(num_classes):
            x_selected = x_train[y_train==j]
            axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)-1), :, :], cmap=plt.get_cmap("gray"))
            axs[j][i].axis("off")
            if i == 2:
                num_samples.append(len(x_selected))
    plt.show()


def label_shape_match(x_train, y_train, x_test, y_test):
    assert(x_train.shape[0] == y_train.shape[0]), "The number of training images is not equal to the number of training labels"
    assert(x_test.shape[0] == y_test.shape[0]), "The number of test images is not equal to the number of test labels"
    assert(x_train.shape[1:] == (28, 28)), "The training images are not 28x28"
    assert(x_test.shape[1:] == (28, 28)), "The test images are not 28x28"

if __name__ == '__main__':
    main()
