# Matplotlib used to plot images and figures
# How to plot an image with matplotlib ?
# plt.imshow(your_img) where your_img is of shape (h, w, 3)
# If the image is in grayscale you can use plt.imshow(your_img, cmap="gray")
# plt.title("Your title") You can set a title before to plot the image
# plt.show()
import matplotlib.pyplot as plt
# Tensorflow
import tensorflow as tf
# Numpy for Matrix/Vector manipulation
import numpy as np


# Utils methods -->
# load_fashion_mnist :
# Input : ()
# Return : Tuple of 4 values of type numpy (vector/matrix) : images, targets, images_test, targets_test
from utils import load_fashion_mnist

def main():
    # Name of each target in the dataset
    # targets_names[id] give you the name associated with the target id
    targets_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    # TODO 1)
    # Load the fashion mnist dataset and print the shape of each output
    # Does the shape of each output make sense ?
    # NOTE : You can search online how to print the shape of a numpy array

    # TODO 2)
    # In order to plot an image, we first need to reshape it.
    # The image to plot should be of shape (h, w, 3) or (h, w) if the image
    # is in grayscale. Once you have reshaped the image, you can plot it with
    # matplotlib. Plot multiple images with the associated target as a title
    # to check what each class looks like.
    # NOTE : You can search online how to reshape a numpy array.

    # TODO 3)
    # Create a Sequential Model using the appropriate activation function
    # at each Dense layer.
    model = tf.keras.models.Sequential()
    # TODO : Add the layers in the sequential model above

    # The following compile method is used to init the model with a loss
    # function to minimize with the Stochastic Gradient Descent method.
    # The accuracy give us the percentage of good prediction.
    # UNCOMMENT the following once you have done the previous step
    #model.compile(
    #    loss="sparse_categorical_crossentropy",
    #    optimizer="sgd",
    #    metrics=["accuracy"]
    #)

    # TODO 4) Run a simple prediction on the first element of your dataset
    # and print the output. Does the output make sense for you ?


    # TODO 5) Train the model using model.fit(
    # history = model.fit(your_inputs, your_targets, batch_size=8, epochs=10, validation_split=0.2)
    # Inputs :
    # your_inputs : The images you want to use to train the model
    # your_targets : The classes associated with each image of the train set
    # batch_size : The number of image the model is going to train on at each iteration
    # epochs : The number of time to go through all the dataset
    # validation_split : The percentage of data to keep for the validation.
    # Outputs :
    # history.history["loss"] give you a list with the progress of the loss on the training set
    # history.history["val_loss"] give you a list with the progress of the loss on the validation set
    # history.history["acc"] give you a list with the progress of the accuracy on the training set
    # history.history["val_acc"] give you a list with the progress of the accuracy on the validation set

    # TODO 6)
    # Plot the result of the training withe matplotlib using plt.plot
    # Plot each curve from the training to check the progress of your model
    # Is there an overfitting ?
    # plt.plot(values, label="")
    # plt.legend(loc='upper right')
    # plt.show()


    # TODO 7)
    # Evaluate the model on the test set using the model.evaluate() method
    # model.evaluate(inputs, targets)
    # The method return an array with the loss and accuracy on the test set.


if __name__ == '__main__':
    main()
