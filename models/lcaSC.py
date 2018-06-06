import tensorflow as tf
import models.utils as utils
import numpy as np
import pdb

class lcaSC(object):
    def __init__(self, inputNode, dictionary, alpha, lr):
        [batch, f] = inputNode.get_shape().as_list()
        [num_class, dict_size, f] = dictionary.get_shape().as_list()

        actShape = [num_class, batch, dict_size]
        self.potential_init = tf.random_uniform(actShape, 0, 1.05*alpha, dtype=tf.float32)

        self.potential = utils.weight_variable(actShape, "potential", 1e-3)
        self.activation = utils.weight_variable(actShape, "activation", 1e-3)
        self.recon = tf.matmul(self.activation, dictionary)
        expand_input = tf.expand_dims(inputNode, 0)
        error = expand_input - self.recon
        #self.recon_error = 0.5 * tf.reduce_mean(tf.reduce_sum(error**2, axis=2))
        self.recon_error = 0.5 * tf.reduce_sum(error**2, axis=[1, 2])
        #self.l1_sparsity = tf.reduce_sum(tf.abs(self.activation))
        self.l1_sparsity = tf.reduce_sum(tf.abs(self.activation), axis=[1, 2])
        self.nnz = tf.count_nonzero(self.activation, axis=[1, 2]) / (batch * dict_size)
        self.loss = self.recon_error + alpha * self.l1_sparsity

        self.calc_activation = self.activation.assign(tf.sign(self.potential) * tf.nn.relu(tf.abs(self.potential) - alpha))

        self.reset_potential = self.potential.assign(self.potential_init)

        opt = tf.train.AdamOptimizer(lr)
        #Calculate recon gradient wrt activation
        recon_grad = opt.compute_gradients(self.recon_error, [self.activation])
        #Apply gradient (plus shrinkage) to potential
        #d_potential = [(recon_grad[0][0] + (self.potential - self.activation)/(num_class*batch), self.potential)]
        d_potential = [(recon_grad[0][0] + (self.potential - self.activation), self.potential)]
        self.train_step = opt.apply_gradients(d_potential)

    def __call__(self, sess, feed_dict, max_iterations=100, verbose=False):
        sess.run(self.reset_potential)
        sess.run(self.calc_activation)
        for it in range(max_iterations):
            sess.run(self.train_step, feed_dict=feed_dict)
            sess.run(self.calc_activation)
            if(verbose):
                #Calc stats
                [recon_err, l1_sparsity, loss, nnz] = sess.run(
                        [self.recon_error, self.l1_sparsity, self.loss, self.nnz], feed_dict=feed_dict)
                print(it, ": \trecon_error", np.mean(recon_err), "\tl1_sparsity", np.mean(l1_sparsity), "\tloss", np.mean(loss), "\tnnz", np.mean(nnz))

