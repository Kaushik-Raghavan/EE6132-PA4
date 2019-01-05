from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import skimage.transform as im
import tensorflow as tf
import numpy as np
import random
import joblib
import struct
import time
import os


FREQUENCY_OF_VERBOSE = 1
FREQUENCY_OF_SUMMARY_UPDATE = 15


class Model:

    def __init__(self, hidden_layer_sizes=None, activation_fn=None, input_dims=(784,), output_dims=(784,), output_fn=tf.identity):
        self.input = None
        self.activation_fn = activation_fn
        self.comp_graph = tf.Graph()
        with self.comp_graph.as_default():
            self.is_training = tf.placeholder(dtype=tf.bool, name='is_in_training_mode')

            self.image_batch = tf.placeholder(dtype=tf.float32, shape=(None,) + input_dims, name='feed_input')
            self.output_batch = tf.placeholder(dtype=tf.float32, shape=(None,) + output_dims, name='feed_labels')
            self.input_shape = self.image_batch.shape
            self.output_shape = self.output_batch.shape
            self.sequence_len = tf.placeholder(dtype=tf.float32, shape=(None,), name='sequence_length')
            self.output_fn = output_fn

    def build(self, hidden_layer_sizes=None):
        pass

    def get_loss(self, predictions, gt_labels):
        with self.comp_graph.as_default():
            print(gt_labels.shape, predictions.shape)
            loss_mean = tf.losses.mean_squared_error(labels=tf.cast(gt_labels, dtype=predictions.dtype), predictions=predictions)
            return loss_mean

    def train(self, input_images, input_labels, learning_rate=1e-4, batch_size=16, num_epochs=10,
              valid_images=None, valid_labels=None,
              summary_path='log_dir/', checkpoint_path="./models/", save_model=False, write_summary=False):
        N = len(input_images)

        with self.comp_graph.as_default():
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                opt = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam_optimizer')
                train_op = opt.minimize(self.loss_op, global_step=tf.train.get_global_step(), name='training_operations')
            saver = tf.train.Saver(max_to_keep=5)

        if write_summary:
            names = ['training', 'valid_loss', 'valid_acc']
            for name in names:
                folderpath = os.path.join(summary_path, name)
                if not os.path.exists(folderpath):
                    os.makedirs(folderpath)
        summary_writer_train_loss = tf.summary.FileWriter(os.path.join(summary_path, 'training/'), self.sess.graph)
        summary_writer_valid_loss = tf.summary.FileWriter(os.path.join(summary_path, 'valid_loss/'), self.sess.graph)
        summary_writer_valid_acc = tf.summary.FileWriter(os.path.join(summary_path, 'valid_acc/'), self.sess.graph)
        train_loss_summary = tf.summary.scalar('Train_loss', self.loss_op)
        valid_loss_summary = tf.summary.scalar('Validation_loss', self.loss_op)
        valid_acc_summary = tf.summary.scalar('Validation_accuracy', self.accuracy_op)

        with self.comp_graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

        training_error, test_error = [], []

        for epoch in range(num_epochs):
            print('Epoch {}'.format(epoch))
            data = list(zip(input_images, input_labels))
            random.shuffle(data)
            input_images, input_labels = zip(*data)
            input_images = np.array(input_images)
            input_labels = np.array(input_labels)

            num_batches = int(N / batch_size)
            if N % batch_size is not 0:
                num_batches += 0

            ckpt_step = num_batches // FREQUENCY_OF_SUMMARY_UPDATE
            if num_batches % FREQUENCY_OF_SUMMARY_UPDATE is not 0:
                ckpt_step += 1

            verbose_step = num_batches // FREQUENCY_OF_SUMMARY_UPDATE
            if verbose_step % FREQUENCY_OF_SUMMARY_UPDATE is not 0:
                verbose_step += 1
            verbose_step = 3 * ckpt_step

            start_idx = 0

            old_kernel = np.zeros((3, 3, 16, 16))
            for step in range(num_batches):

                end_idx = min(start_idx + batch_size, N)
                if (step + 1) % ckpt_step is 0:
                    feed_dict = {self.image_batch: input_images[start_idx: end_idx],
                                 self.output_batch: input_labels[start_idx: end_idx],
                                 self.is_training: True}
                    _, loss, train_summary = self.sess.run([train_op, self.loss_op, train_loss_summary], feed_dict=feed_dict)

                    if write_summary:
                        with self.comp_graph.as_default():
                            summary_writer_train_loss.add_summary(train_summary, step + 1 + num_batches * epoch)

                    verbose = "Step = {}: Training loss = {:.5f}".format(step + 1, loss)
                    if valid_images is not None and valid_labels is not None:
                        feed_dict = {self.image_batch: valid_images, self.output_batch: valid_labels, self.is_training: True}
                        valid_loss, loss_summary, acc_summary = \
                            self.sess.run([self.loss_op, valid_loss_summary, valid_acc_summary],
                                          feed_dict=feed_dict)

                        if write_summary:
                            with self.comp_graph.as_default():
                                summary_writer_valid_acc.add_summary(acc_summary, step + 1 + num_batches * epoch)
                                summary_writer_valid_loss.add_summary(loss_summary, step + 1 + num_batches * epoch)

                        valid_verbose = "Validation loss = {:.5f};".format(valid_loss)
                        verbose = verbose + '; ' + valid_verbose
                        test_error.append([epoch * num_batches + step, valid_loss])

                    if (step + 1) % (verbose_step) is 0:
                        print(verbose)

                else:
                    feed_dict = {self.image_batch: input_images[start_idx: end_idx], self.output_batch: input_labels[start_idx: end_idx], self.is_training: True}
                    _, loss = self.sess.run([train_op, self.loss_op], feed_dict=feed_dict)
                    verbose = "Step = {}: Training loss = {:.5f}".format(step + 1, loss)

                start_idx += batch_size

                training_error.append([epoch * num_batches + step, loss])

            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            if save_model:
                with self.comp_graph.as_default():
                    saver.save(self.sess, os.path.join(checkpoint_path, 'model'), global_step=epoch)
                print("Checkpoint created after {} epochs".format(epoch + 1))

        print(len(training_error), len(test_error))
        np.savetxt(os.path.join(summary_path, "training_error.txt"), training_error)
        np.savetxt(os.path.join(summary_path, "validation_error.txt"), test_error)

    def transform(self, test_images):
        with self.comp_graph.as_default():
            feed_dict = {self.image_batch: test_images, self.is_training: False}
            latent_vector = self.sess.run(self.latent_vector, feed_dict=feed_dict)
        return latent_vector

    def inverse_transform(self, latent_vector):
        with self.comp_graph.as_default():
            feed_dict = {self.latent_vector: latent_vector, self.is_training: False}
            outp = self.sess.run(self.output, feed_dict=feed_dict)
        return outp

    def score(self, test_images, test_labels):
        with self.comp_graph.as_default():
            feed_dict = {self.image_batch: test_images, self.output_batch: test_labels, self.is_training: False}
            test_accuracy = self.sess.run(self.loss_op, feed_dict=feed_dict)
        return test_accuracy

    def predict(self, images, one_hot_output=False):
        with self.comp_graph.as_default():
            out = self.sess.run(self.output_fn(self.output), feed_dict={self.image_batch: images, self.is_training: False})
            return out

    def load(self, model_dir):
        """ Loads the model stored at the latest check point in the given directory """
        with self.comp_graph.as_default():
            latest_checkpoint = tf.train.latest_checkpoint(model_dir)
            saver = tf.train.Saver()
            self.sess.run(tf.local_variables_initializer())
            saver.restore(self.sess, latest_checkpoint)

    def get_filter_val(self, layer_name):
        kernel_val, bias_val = None, None
        kernel_val = self.comp_graph.get_tensor_by_name(layer_name + '/kernel:0').eval(session=self.sess)
        bias_val = self.comp_graph.get_tensor_by_name(layer_name + '/bias:0').eval(session=self.sess)
        return kernel_val, bias_val

    def get_bn_params(self, layer_name):
        gamma, beta = None, None
        gamma = self.comp_graph.get_tensor_by_name(layer_name + '/gamma:0').eval(session=self.sess)
        beta = self.comp_graph.get_tensor_by_name(layer_name + '/beta:0').eval(session=self.sess)
        moving_mean = self.comp_graph.get_tensor_by_name(layer_name + '/moving_mean:0').eval(session=self.sess)
        moving_var = self.comp_graph.get_tensor_by_name(layer_name + '/moving_variance:0').eval(session=self.sess)
        return gamma, beta, moving_mean, moving_var

    def plot_filters(self, layer_name, mat_shape=(28, 28), fig_shape=(28, 28), filter_idx=None, plot_title=None):
        kernel_val, bias_val = self.get_filter_val(layer_name)
        if filter_idx is None:
            # Not convolutional filter
            kernel_val = np.transpose(np.squeeze(kernel_val))
        else:
            # Convolutional filter
            kernel_val = np.squeeze(kernel_val[:, :, :, filter_idx])
            kernel_val = np.transpose(kernel_val, axes=(2, 0, 1))

        if plot_title is None:
            plot_title = layer_name

        fig, axes = plt.subplots(2, 4, figsize=(10, 5), subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
        fig.suptitle(plot_title, fontsize=16)
        for i, ax in enumerate(axes.flat):
            new_img = im.resize(np.reshape(kernel_val[i], mat_shape), fig_shape)
            ax.imshow(new_img, cmap='hot')
        plt.show()

    def unpool(self, net, mask, stride):
        ## unpool function credits: https://github.com/yselivonchyk/Tensorflow_WhatWhereAutoencoder/blob/master/WhatWhereAutoencoder.py
        assert mask is not None
        with self.comp_graph.as_default():
            mask = tf.stop_gradient(mask)
            with tf.name_scope('UnPool2D'):
                ksize = [1, stride, stride, 1]
                input_shape = net.get_shape().as_list()
                batch_size = tf.shape(net, out_type=tf.int64)[0]
                # calculation new shape
                output_shape = (batch_size, input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
                # calculation indices for batch, height, width and feature maps
                one_like_mask = tf.ones_like(mask)
                batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[batch_size, 1, 1, 1])
                b = one_like_mask * batch_range
                y = mask // (output_shape[2] * output_shape[3])
                x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
                feature_range = tf.range(output_shape[3], dtype=tf.int64)
                f = one_like_mask * feature_range
                # transpose indices & reshape update values to one dimension
                updates_size = tf.size(net)
                indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
                values = tf.reshape(net, [updates_size])
                ret = tf.scatter_nd(indices, values, output_shape)
                return ret


class AutoEncoder(Model):
    def __init__(self, hidden_layer_sizes, input_dims=(784), output_dims=(784,), activation_fn=None, output_fn=tf.identity):
        Model.__init__(self, hidden_layer_sizes, activation_fn=activation_fn, input_dims=input_dims, output_dims=output_dims, output_fn=output_fn)
        self.build(hidden_layer_sizes=hidden_layer_sizes, output_dims=output_dims)
        self.sess = tf.Session(graph=self.comp_graph)

    def build(self, hidden_layer_sizes, output_dims=(784,)):
        with self.comp_graph.as_default():
            layer_idx = 0
            latent_layer_idx = len(hidden_layer_sizes) // 2
            outp = self.image_batch
            for sz in hidden_layer_sizes[:latent_layer_idx]:
                outp = tf.layers.dense(inputs=outp, units=sz, activation=self.activation_fn, name='DenseLayer_' + str(layer_idx))
                layer_idx += 1
            self.latent_vector = tf.layers.dense(inputs=outp, units=hidden_layer_sizes[latent_layer_idx],
                                                 activation=self.activation_fn, name='LatentVectorLayer')
            outp = self.latent_vector
            for sz in hidden_layer_sizes[latent_layer_idx + 1:]:
                outp = tf.layers.dense(inputs=outp, units=sz, activation=self.activation_fn, name='DenseLayer_' + str(layer_idx))
                layer_idx += 1
            self.output = tf.layers.dense(inputs=outp, units=output_dims[0], activation=None, name='OutputLayer')

            print(" ", self.output.shape, self.output_batch.shape)
            self.loss_op = self.get_loss(tf.reshape(self.output, tf.shape(self.output_batch)), self.output_batch)
            pred_output = self.output_fn(tf.reshape(self.output_fn(self.output), shape=tf.shape(self.output_batch)))
            _, self.accuracy_op = tf.metrics.accuracy(labels=self.output_batch, predictions=pred_output, name='accuracy_op')


class SparseAutoEncoder(Model):
    def __init__(self, hidden_layer_sizes, l1_weight, input_dims=(784), output_dims=(784,), activation_fn=None, output_fn=tf.identity):
        Model.__init__(self, hidden_layer_sizes, activation_fn=activation_fn, input_dims=input_dims, output_dims=output_dims, output_fn=output_fn)
        self.build(l1_weight=l1_weight, hidden_layer_sizes=hidden_layer_sizes, output_dims=output_dims)
        self.sess = tf.Session(graph=self.comp_graph)

    def build(self, l1_weight, hidden_layer_sizes, output_dims=(784,)):
        with self.comp_graph.as_default():
            layer_idx = 0
            latent_layer_idx = len(hidden_layer_sizes) // 2
            outp = self.image_batch
            for sz in hidden_layer_sizes[:latent_layer_idx]:
                outp = tf.layers.dense(inputs=outp, units=sz, activation=self.activation_fn, name='DenseLayer_' + str(layer_idx))
                layer_idx += 1
            self.latent_vector = tf.layers.dense(inputs=outp, units=hidden_layer_sizes[latent_layer_idx],
                                                 activation=self.activation_fn, name='LatentVectorLayer')
            outp = self.latent_vector
            for sz in hidden_layer_sizes[latent_layer_idx + 1:]:
                outp = tf.layers.dense(inputs=outp, units=sz, activation=self.activation_fn, name='DenseLayer_' + str(layer_idx))
                layer_idx += 1
            self.output = tf.layers.dense(inputs=outp, units=output_dims[0], activation=None, name='OutputLayer')

            print(" ", self.output.shape, self.output_batch.shape)
            self.loss_op = self.get_loss(tf.reshape(self.output, tf.shape(self.output_batch)), self.output_batch) +\
                           tf.losses.absolute_difference(tf.zeros(shape=tf.shape(self.latent_vector)), self.latent_vector, weights=l1_weight)
            pred_output = self.output_fn(tf.reshape(self.output_fn(self.output), shape=tf.shape(self.output_batch)))
            _, self.accuracy_op = tf.metrics.accuracy(labels=self.output_batch, predictions=pred_output, name='accuracy_op')


class DenoisingAutoEncoder(Model):
    def __init__(self, hidden_layer_sizes, std, input_dims=(784), output_dims=(784,), activation_fn=None, output_fn=tf.identity):
        Model.__init__(self, hidden_layer_sizes, activation_fn=activation_fn, input_dims=input_dims, output_dims=output_dims, output_fn=output_fn)
        self.build(std=std, hidden_layer_sizes=hidden_layer_sizes, output_dims=output_dims)
        self.sess = tf.Session(graph=self.comp_graph)

    def build(self, std, hidden_layer_sizes, output_dims=(784,)):
        with self.comp_graph.as_default():
            layer_idx = 0
            latent_layer_idx = len(hidden_layer_sizes) // 2
            noise_generator = tf.distributions.Normal(loc=0.0, scale=std, name='NoiseGenerator')
            random_num = int((float(time.clock()) / 1009) * 1000000)
            outp = self.image_batch + noise_generator.sample(tf.shape(self.image_batch), seed=random_num)
            for sz in hidden_layer_sizes[:latent_layer_idx]:
                outp = tf.layers.dense(inputs=outp, units=sz, activation=self.activation_fn, name='DenseLayer_' + str(layer_idx))
                layer_idx += 1
            self.latent_vector = tf.layers.dense(inputs=outp, units=hidden_layer_sizes[latent_layer_idx],
                                                 activation=self.activation_fn, name='LatentVectorLayer')
            outp = self.latent_vector
            for sz in hidden_layer_sizes[latent_layer_idx + 1:]:
                outp = tf.layers.dense(inputs=outp, units=sz, activation=self.activation_fn, name='DenseLayer_' + str(layer_idx))
                layer_idx += 1
            self.output = tf.layers.dense(inputs=outp, units=output_dims[0], activation=None, name='OutputLayer')

            print(" ", self.output.shape, self.output_batch.shape)
            self.loss_op = self.get_loss(tf.reshape(self.output, tf.shape(self.output_batch)), self.output_batch)
            pred_output = self.output_fn(tf.reshape(self.output_fn(self.output), shape=tf.shape(self.output_batch)))
            _, self.accuracy_op = tf.metrics.accuracy(labels=self.output_batch, predictions=pred_output, name='accuracy_op')


class UnpoolCAE(Model):
    def __init__(self, input_dims=(28, 28, 1), output_dims=(28, 28, 1),
                 activation_fn=None, output_fn=tf.identity):
        Model.__init__(self, activation_fn=activation_fn, input_dims=input_dims, output_dims=output_dims, output_fn=output_fn)
        self.build(output_dims=output_dims)
        self.sess = tf.Session(graph=self.comp_graph)

    def build(self, output_dims=(28, 28, 1)):
        with self.comp_graph.as_default():

            with tf.variable_scope("Encoder", reuse=False):
                encoder_conv1 = tf.layers.conv2d(inputs=self.image_batch, filters=8, kernel_size=3, strides=1, padding='same',
                                         activation=self.activation_fn, name='conv1')
                maxpool1, argmax1 = tf.nn.max_pool_with_argmax(input=encoder_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                               padding='SAME', name='max_pool1')
                encoder_conv2 = tf.layers.conv2d(inputs=maxpool1, filters=16, kernel_size=3, strides=1, padding='same',
                                         activation=self.activation_fn, name='conv2')
                maxpool2, argmax2 = tf.nn.max_pool_with_argmax(input=encoder_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                               padding='SAME', name='max_pool2')
                encoder_conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=3, strides=1, padding='same',
                                         activation=self.activation_fn, name='conv3')
                maxpool3, argmax3 = tf.nn.max_pool_with_argmax(input=encoder_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                               padding='SAME', name='max_pool3')

            self.latent_vector = maxpool3

            with tf.variable_scope("Decoder", reuse=False):
                unpool1 = self.unpool(self.latent_vector, argmax3, 2)
                unpool1 = unpool1[:, :-1, :-1, :]
                decoder_conv1 = tf.layers.conv2d(inputs=unpool1, filters=16, kernel_size=3, strides=1, padding='same',
                                                 activation=self.activation_fn, name='conv1')
                unpool2 = self.unpool(decoder_conv1, argmax2, 2)
                decoder_conv2 = tf.layers.conv2d(inputs=unpool2, filters=8, kernel_size=3, strides=1, padding='same',
                                                 activation=self.activation_fn, name='conv2')
                unpool3 = self.unpool(decoder_conv2, argmax1, 2)
                decoder_conv3 = tf.layers.conv2d(inputs=unpool3, filters=1, kernel_size=3, strides=1, padding='same',
                                                 activation=self.activation_fn, name='conv3')

            self.output = decoder_conv3

            self.loss_op = self.get_loss(tf.reshape(self.output, tf.shape(self.output_batch)), self.output_batch)
            pred_output = self.output_fn(tf.reshape(self.output_fn(self.output), shape=tf.shape(self.output_batch)))
            _, self.accuracy_op = tf.metrics.accuracy(labels=self.output_batch, predictions=pred_output, name='accuracy_op')


class DeconvCAE(Model):
    def __init__(self, input_dims=(28, 28, 1), output_dims=(28, 28, 1),
                 activation_fn=None, output_fn=tf.identity):
        Model.__init__(self, activation_fn=activation_fn, input_dims=input_dims, output_dims=output_dims, output_fn=output_fn)
        self.build(output_dims=output_dims)
        self.sess = tf.Session(graph=self.comp_graph)

    def build(self, output_dims=(28, 28, 1)):
        with self.comp_graph.as_default():

            with tf.variable_scope("Encoder", reuse=False):
                encoder_conv1 = tf.layers.conv2d(inputs=self.image_batch, filters=8, kernel_size=3, strides=1, padding='same',
                                         activation=self.activation_fn, name='conv1')
                maxpool1 = tf.layers.max_pooling2d(inputs=encoder_conv1, pool_size=2, strides=2,
                                                   padding='SAME', name='max_pool1')
                encoder_conv2 = tf.layers.conv2d(inputs=maxpool1, filters=16, kernel_size=3, strides=1, padding='same',
                                         activation=self.activation_fn, name='conv2')
                maxpool2 = tf.layers.max_pooling2d(inputs=encoder_conv2, pool_size=2, strides=2,
                                                   padding='SAME', name='max_pool2')
                encoder_conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=3, strides=1, padding='same',
                                         activation=self.activation_fn, name='conv3')
                maxpool3 = tf.layers.max_pooling2d(inputs=encoder_conv3, pool_size=2, strides=2,
                                                   padding='SAME', name='max_pool3')

            self.latent_vector = maxpool3
            last_conv_layer_shape = encoder_conv3.get_shape().as_list()[1:]
            print(last_conv_layer_shape)

            with tf.variable_scope("Decoder", reuse=False):
                kernel = tf.get_variable('decoder_conv1_filter', shape=[3, 3, 16, 16], dtype=tf.float32,
                                         initializer=tf.glorot_uniform_initializer, trainable=True)
                decoder_conv1 = tf.nn.conv2d_transpose(value=self.latent_vector, filter=kernel,
                                                       output_shape=[tf.shape(self.latent_vector)[0], 7, 7, 16], #+ last_conv_layer_shape,
                                                       strides=[1, 2, 2, 1], padding='SAME', name='deconv1')
                # decoder_conv1 = tf.layers.conv2d_transpose(inputs=self.latent_vector, filters=16, kernel_size=3, strides=2, padding='same',
                #                                            activation=self.activation_fn, name='deconv1_')
                decoder_conv2 = tf.layers.conv2d_transpose(inputs=decoder_conv1, filters=8, kernel_size=3, strides=2, padding='same',
                                                           activation=self.activation_fn, name='deconv2')
                decoder_conv3 = tf.layers.conv2d_transpose(inputs=decoder_conv2, filters=1, kernel_size=3, strides=2, padding='same',
                                                           activation=self.activation_fn, name='deconv3')

            self.output = decoder_conv3

            self.loss_op = self.get_loss(tf.reshape(self.output, tf.shape(self.output_batch)), self.output_batch)
            pred_output = self.output_fn(tf.reshape(self.output_fn(self.output), shape=tf.shape(self.output_batch)))
            _, self.accuracy_op = tf.metrics.accuracy(labels=self.output_batch, predictions=pred_output, name='accuracy_op')


class UnpoolDeconvCAE(Model):
    def __init__(self, input_dims=(28, 28, 1), output_dims=(28, 28, 1),
                 activation_fn=None, output_fn=tf.identity):
        Model.__init__(self, activation_fn=activation_fn, input_dims=input_dims, output_dims=output_dims, output_fn=output_fn)
        self.build(output_dims=output_dims)
        self.sess = tf.Session(graph=self.comp_graph)

    def build(self, output_dims=(28, 28, 1)):
        with self.comp_graph.as_default():

            with tf.variable_scope("Encoder", reuse=False):
                encoder_conv1 = tf.layers.conv2d(inputs=self.image_batch, filters=8, kernel_size=3, strides=1, padding='same',
                                         activation=self.activation_fn, name='conv1')
                maxpool1 = tf.layers.max_pooling2d(inputs=encoder_conv1, pool_size=2, strides=2,
                                                            padding='SAME', name='max_pool1')
                encoder_conv2 = tf.layers.conv2d(inputs=maxpool1, filters=16, kernel_size=3, strides=1, padding='same',
                                         activation=self.activation_fn, name='conv2')
                maxpool2, argmax2 = tf.nn.max_pool_with_argmax(input=encoder_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                               padding='SAME', name='max_pool2')
                encoder_conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=3, strides=1, padding='same',
                                         activation=self.activation_fn, name='conv3')
                maxpool3, argmax3 = tf.nn.max_pool_with_argmax(input=encoder_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                               padding='SAME', name='max_pool3')

            self.latent_vector = maxpool3

            with tf.variable_scope("Decoder", reuse=False):
                unpool1 = self.unpool(self.latent_vector, argmax3, 2)
                unpool1 = unpool1[:, :-1, :-1, :]
                decoder_conv1 = tf.layers.conv2d(inputs=unpool1, filters=16, kernel_size=3, strides=1, padding='same',
                                                 activation=self.activation_fn, name='conv1')
                unpool2 = self.unpool(decoder_conv1, argmax2, 2)
                decoder_conv2 = tf.layers.conv2d(inputs=unpool2, filters=8, kernel_size=3, strides=1, padding='same',
                                                 activation=self.activation_fn, name='conv2')
                decoder_deconv1 = tf.layers.conv2d_transpose(inputs=decoder_conv2, filters=1, kernel_size=3, strides=2, padding='same',
                                                             activation=self.activation_fn, name='deconv1')

            self.output = decoder_deconv1
            print(" ", self.output.shape, self.output_batch.shape)

            self.loss_op = self.get_loss(tf.reshape(self.output, tf.shape(self.output_batch)), self.output_batch)
            pred_output = self.output_fn(tf.reshape(self.output_fn(self.output), shape=tf.shape(self.output_batch)))
            _, self.accuracy_op = tf.metrics.accuracy(labels=self.output_batch, predictions=pred_output, name='accuracy_op')


class RevUnpoolDeconvCAE(Model):
    def __init__(self, input_dims=(28, 28, 1), output_dims=(28, 28, 1),
                 activation_fn=None, output_fn=tf.identity):
        Model.__init__(self, activation_fn=activation_fn, input_dims=input_dims, output_dims=output_dims, output_fn=output_fn)
        self.build(output_dims=output_dims)
        self.sess = tf.Session(graph=self.comp_graph)

    def build(self, output_dims=(28, 28, 1)):
        with self.comp_graph.as_default():

            with tf.variable_scope("Encoder", reuse=False):
                maxpool1 = tf.layers.max_pooling2d(inputs=self.image_batch, pool_size=2, strides=2,
                                                            padding='SAME', name='max_pool1')
                encoder_conv1 = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=3, strides=1, padding='same',
                                         activation=self.activation_fn, name='conv1')
                maxpool2, argmax2 = tf.nn.max_pool_with_argmax(input=encoder_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                               padding='SAME', name='max_pool2')
                encoder_conv2 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=3, strides=1, padding='same',
                                         activation=self.activation_fn, name='conv2')
                maxpool3, argmax3 = tf.nn.max_pool_with_argmax(input=encoder_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                               padding='SAME', name='max_pool3')
                encoder_conv3 = tf.layers.conv2d(inputs=maxpool3, filters=16, kernel_size=3, strides=1, padding='same',
                                         activation=self.activation_fn, name='conv3')

            self.latent_vector = encoder_conv3

            with tf.variable_scope("Decoder", reuse=False):
                decoder_conv1 = tf.layers.conv2d(inputs=self.latent_vector, filters=16, kernel_size=3, strides=1, padding='same',
                                                 activation=self.activation_fn, name='conv1')
                unpool1 = self.unpool(decoder_conv1, argmax3, 2)
                unpool1 = unpool1[:, :-1, :-1, :]
                decoder_conv2 = tf.layers.conv2d(inputs=unpool1, filters=8, kernel_size=3, strides=1, padding='same',
                                                 activation=self.activation_fn, name='conv2')
                unpool2 = self.unpool(decoder_conv2, argmax2, 2)

                decoder_deconv1 = tf.layers.conv2d_transpose(inputs=unpool2, filters=1, kernel_size=3, strides=2, padding='same',
                                                             activation=self.activation_fn, name='deconv1')

            self.output = decoder_deconv1
            print(" ", self.output.shape, self.output_batch.shape)

            self.loss_op = self.get_loss(tf.reshape(self.output, tf.shape(self.output_batch)), self.output_batch)
            pred_output = self.output_fn(tf.reshape(self.output_fn(self.output), shape=tf.shape(self.output_batch)))
            _, self.accuracy_op = tf.metrics.accuracy(labels=self.output_batch, predictions=pred_output, name='accuracy_op')
