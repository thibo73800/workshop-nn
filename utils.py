from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

def load_fashion_mnist():
    # Fashio MNIST
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (images, targets), (images_test, targets_test) = fashion_mnist.load_data()

    # Get only a subpart of the dataset
    images = images[:10000]
    targets = targets [:10000]

    # Reshape the dataset and convert to float
    images = images.reshape(-1, 784)
    images = images.astype(float)
    images_test = images_test.reshape(-1, 784)
    images_test = images_test.astype(float)

    images = images / 255
    images_test = images_test / 255

    return images, targets, images_test, targets_test, draw_class

def load_quickdraw_dataset():
    dataset_dir = "quick_draw_dataset"
    files = [name for name in os.listdir(dataset_dir) if ".npy" in name]
    max_size_per_cl = 1500
    draw_class = []

    # Evalueate the size of the dataset
    size = 0
    for name in files:
        draws = np.load(os.path.join(dataset_dir, name))
        draws = draws[:max_size_per_cl] # Take only 10 000 draw
        size += draws.shape[0]

    images = np.zeros((size, 28, 28))
    targets = np.zeros((size,))

    it = 0
    t = 0
    for name in files:
        # Open each dataset and add the new class
        draw_class.append(name.replace("full_numpy_bitmap_", "").replace(".npy", ""))
        draws = np.load(os.path.join(dataset_dir, name))
        draws = draws[:max_size_per_cl] # Take only 10 000 draw
        # Add images to the buffer
        images[it:it+draws.shape[0]] = np.invert(draws.reshape(-1, 28, 28))
        targets[it:it+draws.shape[0]] = t
        # Iter
        it += draws.shape[0]
        t += 1

    images = images.astype(np.float32)

    # Shuffle dataset
    indexes = np.arange(size)
    np.random.shuffle(indexes)
    images = images[indexes]
    targets = targets[indexes]

    images, images_test, targets, targets_test = train_test_split(images, targets, test_size=0.33)

    images = images / 255.
    images_test = images_test / 255.


    return images, targets, images_test, targets_test, draw_class

def plot_quickdraw_dataset(images, targets, draw_class):
    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns*rows +1):
        index = np.random.randint(len(images))
        img = images[index]
        fig.add_subplot(rows, columns, i)
        plt.title(draw_class[int(targets[index])])
        plt.imshow(img, cmap="gray")
    plt.show()
