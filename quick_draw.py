import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# TODO 1 ) To start this project you need to download the following files.
# https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pli=1
# - airplane.npy
# - book.npy
# - car.npy
# - dog.npy
# - face.npy
# - apple.npy
# - brain.npy
# - chair.npy
# - eye.npy
# - The Eiffel Tower.npy

# TODO 2) Then you need to store the file in a folder named: quick_draw_dataset

# Utils methods -->
# load_fashion_mnist :
# Input : ()
# Return : Tuple of 5 values of type numpy (vector/matrix) : images, targets, images_test, targets_test, labels
# where labels is the name of each target id.
from utils import load_quickdraw_dataset

# plot_quickdraw_dataset :
# Input : (images, targets, labels)
# Method used to show some image and the associated labels in the dataset
from utils import plot_quickdraw_dataset

def main():
    # TODO 3)
    # Load the Quick Draw Dataset and print the shape of each output

    # TODO 4)
    # Plot the quickdraw_dataset using the plot_quickdraw_dataset method

    # UNCOMMENT the following line once you have done the previous steps
    #images = np.expand_dims(images, axis=-1)

    # TODO 5)
    # Build a sequential model with the appropriate layers
    # The layers could be : Dense, Flatten, Conv2D or MaxPooling2D

    # The following compile method is used to init the model with a loss
    # function to minimize with the AdamOptimizer.
    # The accuracy give us the percentage of good prediction.
    # UNCOMMENT the following once you have done the previous step
    #model.compile(
    #    loss="sparse_categorical_crossentropy",
    #    optimizer="adam",
    #    metrics=["accuracy"]
    #)

    # TODO 6)
    # Make a prediction using the first image of the dataset and print
    # the model output

    # TODO 7) Train the model using model.fit(
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

    # TODO 8)
    # Plot the result of the training withe matplotlib using plt.plot
    # Plot each curve from the training to check the progress of your model
    # Is there an overfitting ?
    # plt.plot(values, label="")
    # plt.legend(loc='upper right')
    # plt.show()

    # TODO 7
    # Save the model with the model.save("") method


if __name__ == '__main__':
    main()
