from sklearn.decomposition import sparse_encode
import tensorflow as tf
from models.base import base
import models.utils as utils
import pdb
import numpy as np
import time

class tbLearn(base):
    def buildModel(self):
        with tf.device(self.params.device):
            #Dictionary elements
            dict_shape = [self.params.num_classes, self.params.dict_size, self.params.num_features]
            self.dict_var = utils.weight_variable(dict_shape)
            #Binary classification
            classifier_shape = [self.params.num_classes, self.params.dict_size, 1]
            self.classifier_var = utils.weight_variable(classifier_shape)



    def trainStep(self, step, trainDataObj):
        (image, gt) = trainDataObj.getData(self.params.batch_size)
        np_dict_var = self.sess.run(self.dict_var)
        sc_array_size = [self.params.num_classes, self.params.batch_size, self.params.dict_size]
        sc_array = np.zeros(sc_array_size)

        start_time = time.time()
        for i_class in range(self.params.num_classes):
            class_dict = np_dict_var[i_class, ...]
            sc_array[i_class, :, :] = sparse_encode(image, class_dict, alpha=self.params.l1_weight)
        end_time = time.time()

        print(end_time - start_time)
        pdb.set_trace()

    #Should return the number correct
    def evalModel(self, images, labels):
        return 0

    def evalModelSummary(self, images, labels, injectAcc):
        #feed_dict = {self.injectBool: True, self.injectAcc:injectAcc}
        feed_dict = {}
        self.writeTestSummary(feed_dict)

