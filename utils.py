from mnist import MNIST
from skimage import feature
# import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import os
import struct
import gzip
import shutil
import scipy


thismodule = sys.modules[__name__]


def one_hot(x, size):
    if not hasattr(x, "__len__"):  # x is a scalar
        x = np.array([x])
    ret = np.zeros((len(x), size))
    for i in range(len(x)):
        ret[i, int(x[i])] = 1.0
    return ret


def plot_from_files(list_of_files, labels=None, title_name=None, different_plots=False, axis_labels=None):
    """
    Reads each file present in the `list_of_files`, extracts a 1D vector, and plots them in a single figure.
    If each array read from file had different sizes, the x-axis is scaled so as to match the maximum sized array's plot
    :param list_of_files: list of strings denothing path of files to be read
    :param title_name: Title of the plot
    """

    plt.figure()
    if title_name is not None:
        plt.title(title_name)

    if different_plots:
        assert(axis_labels == None or len(axis_labels) == len(list_of_files))
    else:
        assert(axis_labels == None or len(axis_labels) == 1)

    for i in range(len(list_of_files)):
        filepath = list_of_files[i]
        file_lines = open(filepath, "r").readlines()
        data = []
        for line in file_lines:
            curr_line = []
            for num in line.split(" "):
                curr_line.append(float(num))
            data.append(curr_line)
        data = np.array(data)
        if different_plots:
            if axis_labels is not None:
                if axis_labels[i][0] is not "":
                    plt.xlabel(axis_labels[i][0])
                if axis_labels[i][1] is not "":
                    plt.ylabel(axis_labels[i][1])
            plt.title(title_name[i])
            plt.plot(data[:, 0], data[:, 1])
            plt.show()
        else:
            if labels is not None:
                plt.plot(data[:, 0], data[:, 1], label=labels[i])
            else:
                plt.plot(data[:, 0], data[:, 1])

    if not different_plots:
        if axis_labels is not None:
            if axis_labels[0][0] is not "":
                plt.xlabel(axis_labels[0][0])
            if axis_labels[0][1] is not "":
                plt.ylabel(axis_labels[0][1])

        if labels is not None:
            plt.legend()
        plt.show()


def read_from_file(filepath):
    file_lines = open(filepath, "r").readlines()
    data = []
    for line in file_lines:
        curr_line = []
        for num in line.split(" "):
            curr_line.append(float(num))
        data.append(curr_line)
    return np.squeeze(np.array(data))


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def read_mnist_char():
    dataset_loc = './data/characters/'
    train_images = read_idx(os.path.join(dataset_loc, 'emnist-balanced-train-images-idx3-ubyte'))
    train_labels = read_idx(os.path.join(dataset_loc, 'emnist-balanced-train-labels-idx1-ubyte'))
    test_images = read_idx(os.path.join(dataset_loc, 'emnist-balanced-test-images-idx3-ubyte'))
    test_labels = read_idx(os.path.join(dataset_loc, 'emnist-balanced-test-labels-idx1-ubyte'))
    return train_images, train_labels, test_images, test_labels


def read_mnist_digits(dataset_loc='./data/'):
    TRAIN_IMAGES = 'train-images-idx3-ubyte'
    TRAIN_LABELS = 'train-labels-idx1-ubyte'
    TEST_IMAGES = 't10k-images-idx3-ubyte'
    TEST_LABELS = 't10k-labels-idx1-ubyte'
    files = [TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS]
    for filename in files:
        filepath = os.path.join(dataset_loc, filename)
        if not os.path.exists(filepath):
            assert os.path.exists(filepath + '.gz'), "File " + filename + ".gz is not downloaded"
            with gzip.open(filepath + '.gz', 'r') as f_in:
                with open(filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    train_images = read_idx(os.path.join(dataset_loc, TRAIN_IMAGES))
    train_labels = read_idx(os.path.join(dataset_loc, TRAIN_LABELS))
    test_images = read_idx(os.path.join(dataset_loc, TEST_IMAGES))
    test_labels = read_idx(os.path.join(dataset_loc, TEST_LABELS))
    return train_images, train_labels, test_images, test_labels


def get_proper_image(image):
    image = np.squeeze(image)
    mn, mx = np.min(image), np.max(image)
    image = image / (mx - mn)
    image = image * 255.0
    return image.astype(np.uint8)


def save_image(path, image):
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


def confusion_matrix(predicted_labels, ground_truth_labels, num_classes):
    mat = np.zeros((num_classes, num_classes))
    for x, y in zip(predicted_labels, ground_truth_labels):
        mat[int(y), int(x)] += 1
    return mat


def metrics(predicted_labels, ground_truth_labels, num_classes):
    tp = np.zeros((num_classes,))
    tn = np.zeros((num_classes,))
    fp = np.zeros((num_classes,))
    fn = np.zeros((num_classes,))
    all_tp = 0.0

    for x, y in zip(predicted_labels, ground_truth_labels):
        if x == y:
            all_tp += 1
            for i in range(num_classes):
                if i == int(x):
                    tp[i] += 1
                else:
                    tn[i] += 1
        else:
            for i in range(num_classes):
                if i == int(x):
                    fp[i] += 1
                elif i == int(y):
                    fn[i] += 1
                else:
                    tn[i] += 1

    total_accuracy = all_tp / len(ground_truth_labels)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2.0 * precision * recall / (precision + recall)

    return total_accuracy, np.mean(precision), np.mean(recall), np.mean(f1_score)


""" Data Augmentation """


def random_deskew(image, height, width):
    # change 1D image to 2D image representation
    original_shape = image.shape
    image = image.reshape(original_shape[:-1] + (height,  width))

    center = (width // 2, height // 2)
    angle = (np.random.rand() - 0.5) * 60.0
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated.reshape(original_shape)


def add_noise(img, std=0.01):
    noise = np.random.normal(loc=0.0, scale=std, size=img.shape)
    return img + noise


def augment_batch(batch):
    h, w = 28, 28
    ret = np.array([random_deskew(img, h, w) for img in batch]) + np.random.normal(0, 5.0, batch.shape)
    # ret = np.array([add_noise(img) for img in ret])
    # ret = batch.copy()
    return ret


def sift_features(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img, None)


def hog_features(img):
    h, w = 28, 28
    H = feature.hog(img.reshape(h, w), orientations=9, pixels_per_cell=(7, 7), block_norm='L2')
    return H
