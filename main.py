from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import skimage.transform as im
import tensorflow as tf
from model import *
import numpy as np
import random
import joblib
import struct
import utils
import time
import os

loc = './'
log_dir = loc + 'log_dir/'
model_dir = loc + 'models/'

model_name = 'CAEtest_tanh'
validate = False
load_saved_model = False
model_address = os.path.join(model_dir, model_name)
epochs = 20
bs = 64
lr = 1e-3

train_images, train_labels, test_images, test_labels = utils.read_mnist_digits()
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))
train_images = train_images / 255.0
test_images = test_images / 255.0


if validate:
    assert not load_saved_model
    hidden_layers = [[1000], [1500], [2000]]
    for sizes in hidden_layers:
        print("Hidden layer shape =", sizes)
        model = AutoEncoder(hidden_layer_sizes=sizes, input_dims=(784,), output_dims=(784,), activation_fn=tf.nn.tanh)
        print("Splitting data into training and validation data")
        train_images, validation_images, train_labels, validation_labels =\
            train_test_split(train_images, train_labels, train_size=0.85, random_state=29)
        tik = time.clock()
        model.train(input_images=train_images, input_labels=train_images,
                    valid_images=validation_images, valid_labels=validation_images,
                    learning_rate=lr, batch_size=bs, num_epochs=epochs,
                    checkpoint_path=os.path.join(model_dir, model_name),
                    summary_path=os.path.join(log_dir, model_name),
                    save_model=False, write_summary=False)
        print("Time taken to complete training = {:.4f} sec\n".format(time.clock() - tik))
else:
    if not load_saved_model:
        # model = AutoEncoder(hidden_layer_sizes=[782, 64, 8, 64, 782], input_dims=(784,), output_dims=(784,), activation_fn=tf.nn.tanh)
        model = UnpoolDeconvCAE(input_dims=(28,28,1), output_dims=(28,28,1), activation_fn=tf.nn.tanh)

        tik = time.clock()
        model.train(input_images=train_images, input_labels=train_images,
                    valid_images=test_images, valid_labels=test_images,
                    learning_rate=lr, batch_size=bs, num_epochs=epochs,
                    checkpoint_path=os.path.join(model_dir, model_name),
                    summary_path=os.path.join(log_dir, model_name),
                    save_model=True, write_summary=False)
        loss = model.score(test_images, test_images)
        print("MSE error Deconv = ", loss)

        print("Time taken to complete training = {:.4f} sec".format(time.clock() - tik))
    else:
        # model = AutoEncoder(hidden_layer_sizes=[782, 64, 8, 64, 782], input_dims=(784,), output_dims=(784,), activation_fn=tf.nn.tanh)
        model = DeconvCAE(input_dims=(28,28,1), output_dims=(28,28,1), activation_fn=tf.nn.tanh)
        model.load(os.path.join(model_dir, 'DeconvCAE2_tanh'))

        print("Model restored")


        model.plot_filters(layer_name='Decoder/deconv2', mat_shape=(3, 3), fig_shape=(9, 9), filter_idx=4, plot_title='DeconvCAE')

        print("MSE score in test images = {:.6f}".format(model.score(test_images, test_images)))


        # generated_images = np.random.rand(4, 784)
        samples = test_images[:4].reshape(-1, 28, 28, 1)
        # samples += np.random.normal(0, 0.0, size=samples.shape)

        """
        fig, axes = plt.subplots(2, 2, figsize=(6, 6), subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.2, wspace=0.0))
        fig.suptitle("True images", fontsize=16)
        reconst = samples
        for i, ax in enumerate(axes.flat):
            mx, mn = np.max(reconst[i]), np.min(reconst[i])
            img = (reconst[i].copy() - (mn + mx) / 2.0) / (mx - mn)
            new_img = im.resize(img.reshape(28,28), (28, 28))
            ax.imshow(new_img, cmap='gray')
        plt.show()
        """

        latent_vec = model.transform(samples)
        reconst = model.inverse_transform(latent_vec)

        """
        fig, axes = plt.subplots(2, 2, figsize=(6, 6), subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.2, wspace=0.0))
        fig.suptitle("Reconstructed images", fontsize=16)
        for i, ax in enumerate(axes.flat):
            mx, mn = np.max(reconst[i]), np.min(reconst[i])
            img = (reconst[i].copy() - (mn + mx) / 2.0) / (mx - mn)
            new_img = im.resize(img.reshape(28, 28), (28, 28))
            ax.imshow(new_img, cmap='gray')
        plt.show()
        """

        reconst = model.inverse_transform(latent_vec + np.random.normal(0, 0.1, size=latent_vec.shape))

        """
        fig, axes = plt.subplots(2, 2, figsize=(6, 6), subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.2, wspace=0.0))
        fig.suptitle("Reconstructed images from noisy latent vector", fontsize=16)
        for i, ax in enumerate(axes.flat):
            mx, mn = np.max(reconst[i]), np.min(reconst[i])
            img = (reconst[i].copy() - (mn + mx) / 2.0) / (mx - mn)
            new_img = im.resize(img.reshape(28, 28), (28, 28))
            ax.imshow(new_img, cmap='gray')
        plt.show()
        """
