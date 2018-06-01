from sklearn.decomposition import sparse_encode
import tensorflow as tf
from models.base import base
import models.utils as utils
import pdb
import numpy as np
import time
from models.lcaSC import lcaSC

class tbLearn(base):
    def buildModel(self):
        with tf.device(self.params.device):
            #Dictionary elements
            dict_shape = [self.params.num_classes, self.params.dict_size, self.params.num_features]
            self.dict_var = utils.weight_variable(dict_shape, "dictionary")
            norm_dict = tf.norm(self.dict_var, axis=2, keepdims=True)
            self.normalize_dict = self.dict_var.assign(self.dict_var/norm_dict)
            #Binary classification
            classifier_shape = [self.params.num_classes, self.params.dict_size, 1]
            self.classifier_var = utils.weight_variable(classifier_shape, "class_weights")

            self.images = tf.placeholder(tf.float32,
                    shape=[self.params.batch_size, self.params.num_features],
                    name = "images")
            self.labels = tf.placeholder(tf.int32,
                    shape = [self.params.batch_size],
                    name = "labels")

            self.norm_images = (self.images - tf.reduce_mean(self.images, axis=1, keepdims=True))/tf.norm(self.images, axis=1, keepdims=True)

            self.scObj = lcaSC(self.norm_images, self.dict_var, self.params.l1_weight, self.params.sc_lr)

            ##Set binary labels for each class
            #onehot_labels = tf.reshape(tf.one_hot(self.labels, self.params.num_classes), [1, 0])
            ##go from [0, 1] to [-1, 1]
            #onehot_labels = onehot_labels * 2 - 1
            #tf.log(1 + tf.exp(self.norm_images

            ##new_classifier = self.classifier_var -
            ##self.update_classifier =


    def trainStep(self, step, trainDataObj):
        (images, labels) = trainDataObj.getData(self.params.batch_size)

        start_time = time.time()
        feed_dict = {self.images: images, self.labels: labels}
        self.sess.run(self.normalize_dict)
        self.scObj(self.sess, feed_dict)

        [recon_err, l1_sparsity, loss, nnz] = self.sess.run(
                [self.scObj.recon_error, self.scObj.l1_sparsity, self.scObj.loss, self.scObj.nnz], feed_dict=feed_dict)
        end_time = time.time()
        print("LCA (", end_time-start_time, "): \trecon_error", recon_err, "\tl1_sparsity", l1_sparsity, "\tloss", loss, "\tnnz", nnz)

        #np_dict_var = self.sess.run(self.dict_var)
        #norm_images = self.sess.run(self.norm_images, feed_dict=feed_dict)
        #sc_array_size = [self.params.num_classes, self.params.batch_size, self.params.dict_size]
        #sc_array = np.zeros(sc_array_size)
        #start_time = time.time()
        #for i_class in range(self.params.num_classes):
        #    class_dict = np_dict_var[i_class, ...]
        #    sc_array[i_class, :, :] = sparse_encode(norm_images, class_dict, alpha=self.params.l1_weight)
        #end_time = time.time()

        #recon = np.matmul(sc_array, np_dict_var)
        #error = norm_images[np.newaxis, ...] - recon
        #recon_error = .5 * np.mean(np.sum(error**2, axis=2))
        #l1_sparsity = np.sum(np.abs(sc_array))
        #nnz = np.count_nonzero(sc_array) / (self.params.num_classes * self.params.batch_size * self.params.dict_size)
        #loss = recon_error + self.params.l1_weight * l1_sparsity
        #print("lasso_lars (", end_time-start_time, "): \trecon_error", recon_error, "\tl1_sparsity", l1_sparsity, "\tloss", loss, "\tnnz", nnz)

        pdb.set_trace()

    #Should return the number correct
    def evalModel(self, images, labels):
        return 0

    def evalModelSummary(self, images, labels, injectAcc):
        #feed_dict = {self.injectBool: True, self.injectAcc:injectAcc}
        feed_dict = {}
        self.writeTestSummary(feed_dict)

