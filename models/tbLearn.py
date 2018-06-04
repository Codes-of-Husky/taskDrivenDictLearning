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
            with tf.name_scope("Variables"):
                #Dictionary elements
                D_shape = [self.params.num_classes, self.params.dict_size, self.params.num_features]
                if(self.params.init_weights is None):
                    self.D = utils.l2_weight_variable(D_shape, "dictionary")
                else:
                    if(len(self.params.init_weights.shape) == 2):
                        init_weights = self.params.init_weights[np.newaxis, ...]
                        init_weights = np.tile(init_weights, [self.params.num_classes, 1, 1])
                    else:
                        init_weights = self.params.init_weights
                    self.D = tf.Variable(init_weights.astype(np.float32), name="dictionary")

                #Binary classification
                W_shape = [self.params.num_classes, self.params.dict_size, self.params.num_features]
                self.W = utils.weight_variable(W_shape, "class_weights")

                self.input = tf.placeholder(tf.float32,
                        shape=[self.params.batch_size, self.params.num_features],
                        name = "input")
                self.labels = tf.placeholder(tf.int64,
                        shape = [self.params.batch_size],
                        name = "labels")

                self.norm_input = (self.input - tf.reduce_mean(self.input, axis=1, keepdims=True))/tf.norm(self.input, axis=1, keepdims=True)

                #Set binary labels for each class
                onehot_labels = tf.transpose(tf.one_hot(self.labels, self.params.num_classes), [1, 0])
                #go from [0, 1] to [-1, 1]
                onehot_labels = onehot_labels * 2 - 1

                #Add to tensorboard
                self.varDict["D"] = self.D
                self.varDict["W"] = self.W
                self.varDict["labels"] = self.labels
                self.varDict["onehot_labels"] = onehot_labels
                if(self.params.image_shape is not None):
                    reshape_image = tf.reshape(self.norm_input, (self.params.batch_size,) + self.params.image_shape)
                    self.imageDict["norm_image"] = reshape_image
                else:
                    self.varDict["norm_input"] = self.norm_input

            with tf.name_scope("SC"):
                self.scObj = lcaSC(self.norm_input, self.D, self.params.l1_weight, self.params.sc_lr)
                sc_activation = self.scObj.activation

                self.varDict["sc_activation"] = sc_activation
                self.scalarDict["sc_recon_err"] = self.scObj.recon_error
                self.scalarDict["sc_l1_sparsity"] = self.scObj.l1_sparsity
                self.scalarDict["sc_loss"] = self.scObj.loss
                self.scalarDict["sc_nnz"] = self.scObj.nnz
                if(self.params.image_shape is not None):
                    reshape_recon = tf.reshape(self.scObj.recon, (self.params.num_classes, self.params.batch_size,) + self.params.image_shape)
                    for i in range(self.params.num_classes):
                        self.imageDict["recon_class_"+str(i)] = reshape_recon[i, ...]
                else:
                    self.varDict["recon"] = self.scObj.recon

            with tf.name_scope("feedforward"):
                tile_input = tf.tile(self.norm_input[tf.newaxis, :, :], [self.params.num_classes, 1, 1])
                feed_forward = tf.matmul(tile_input, self.W, transpose_b=True)
                #Taking inner product of feed_forward with sc_activation
                #i.e., diag(matmul(feed_forward, sc_activation))
                feed_forward = tf.reduce_sum(feed_forward * sc_activation, axis=2)
                self.est_labels = tf.argmax(feed_forward, axis=0)

                self.varDict["feed_forward"] = feed_forward
                self.varDict["est_labels"] = self.est_labels

            with tf.variable_scope('accuracy'):
                #Calculate accuracy
                self.injectBool = tf.placeholder_with_default(False, shape=(), name="injectBool")
                self.injectAcc = tf.placeholder_with_default(0.0, shape=None, name="injectAcc")
                calc_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.est_labels, self.labels), tf.float32))

                accuracy = tf.cond(self.injectBool, lambda: self.injectAcc, lambda: calc_accuracy)
                self.scalarDict["accuracy"] = accuracy
                #TODO inject test accuracy here

            with tf.name_scope("loss"):
                supervised_loss = tf.reduce_mean(tf.log(1 + tf.exp(-onehot_labels * feed_forward))) + (self.params.weight_decay/2) * tf.norm(self.W)
                self.scalarDict["supervised_loss"] = supervised_loss

            with tf.name_scope("opt"):
                D_covar = tf.matmul(self.D, self.D, transpose_b=True) + self.params.l2_weight*tf.eye(self.params.dict_size, batch_shape=[self.params.num_classes])
                #Calculate supervised gradients
                [sup_grad_wrt_a, sup_grad_wrt_W] = tf.gradients(supervised_loss, [sc_activation, self.W])

                #D_covar^-1 * gradient
                sup_grad_wrt_a = tf.transpose(sup_grad_wrt_a, [0, 2, 1])
                beta = tf.matrix_solve_ls(D_covar, sup_grad_wrt_a)

                #compute learning rate
                train_step = tf.Variable(0, name='train_step', dtype=tf.int64)
                #Updates the tf train_step with the global timestep of the object
                update_timestep = tf.assign_add(train_step, 1)
                lr = tf.minimum(self.params.start_lr, self.params.start_lr*(self.params.decay_time/tf.cast(train_step, tf.float32)))

                #Update W
                #Note that the paper adds weight decay on W, but this is encompassed into the gradient wrt W
                update_W = tf.assign_add(self.W,  -lr * sup_grad_wrt_W)
                D_grad_term_1 = tf.matmul(sc_activation, tf.matmul(beta, -self.D, transpose_a=True), transpose_a=True)
                D_grad_term_2 = tf.matmul(beta, (tile_input - self.scObj.recon))
                update_D = tf.assign_add(self.D, -lr*(D_grad_term_1 + D_grad_term_2))

                #Normalize D
                norm_D = tf.norm(self.D, axis=2, keepdims=True)
                #Only normalize if norm > 1, i.e., l2 dict element always <= 1
                norm_D = tf.maximum(tf.ones(D_shape), norm_D)
                #Normalize after update
                with tf.control_dependencies([update_D]):
                    normalize_D= self.D.assign(self.D/norm_D)

                #Group all update ops
                #Always make sure the tf timestep is in sync with global timestep
                with tf.control_dependencies([update_timestep]):
                    self.update_step = tf.group(update_W, update_D)

                self.scalarDict["learning_rate"] = lr


    def trainStep(self, step, trainDataObj):
        (input, labels) = trainDataObj.getData(self.params.batch_size)

        feed_dict = {self.input: input, self.labels: labels}
        #Compute sc
        self.scObj(self.sess, feed_dict)
        #Update variables
        self.sess.run(self.update_step, feed_dict=feed_dict)
        if(step%self.params.write_step == 0):
            self.writeTrainSummary(feed_dict)

    #Should return the number correct
    def evalModel(self, input, labels):
        nbatch = input.shape[0]
        feed_dict = {self.input: input, self.labels:labels}
        #Run sparse coding
        self.scObj(self.sess, feed_dict)
        #Calculate estimated labels
        est_labels = self.sess.run(self.est_labels, feed_dict=feed_dict)
        correct_count = float(np.sum(est_labels == labels))/nbatch
        return correct_count

    def evalModelSummary(self, input, labels, injectAcc):
        feed_dict = {self.input:input, self.labels:labels,
                self.injectBool: True, self.injectAcc:injectAcc}
        self.writeTestSummary(feed_dict)

        #self.writeTestSummary(feed_dict)
        pass

