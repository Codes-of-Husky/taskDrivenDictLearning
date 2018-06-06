from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pdb
from sklearn.decomposition import dict_learning

"""
An object that handles data input
"""
class mnistData(object):
    raw_image_shape = (28, 28, 1)
    num_features = 784
    num_classes = 10

    #TODO set random seed
    def __init__(self, path, flatten=True, augment=True):
        self.mnist = input_data.read_data_sets(path, one_hot=False)

        self.num_train_examples = self.mnist.train.num_examples
        self.num_test_examples = self.mnist.test.num_examples
        self.flatten=flatten
        self.augment = augment

        self.train_images = self.mnist.train.images
        self.train_labels = self.mnist.train.labels
        self.test_images = self.mnist.test.images
        self.test_labels = self.mnist.test.labels

    def getData(self, numExample):
        images, labels = self.mnist.train.next_batch(numExample)

        #Reshape into y, x, f
        images = np.reshape(images, (numExample,) + self.raw_image_shape)
        if(self.augment):
            #Generate random y and x offsets
            shift = np.random.randint(-1, 2, [2])
            #Only shift in y and x dimension
            shift = [0, shift[0], shift[1], 0]
            images = np.roll(images, shift[0], axis=1)
            images = np.roll(images, shift[1], axis=2)

        if(self.flatten):
            images = np.reshape(images, (numExample, self.num_features))

        return (images, labels)


    def getTestData(self):
        test_images = self.test_images

        if(not self.flatten):
            #Reshape into y, x, f
            test_images = np.reshape(test_images,
                    (self.num_test_examples,) + self.raw_image_shape)

        return (test_images, self.test_labels)

    def getNormSample(self, num_sample):
        sample_idx = np.random.permutation(self.num_train_examples)[:num_sample]
        sample_train_images = self.train_images[sample_idx, :]
        #Normalize images
        norm_train_images = (sample_train_images - np.mean(sample_train_images, axis=1, keepdims=True))/np.linalg.norm(sample_train_images, axis=1, keepdims=True)
        return norm_train_images

    def getPCA(self, num_init, num_sample=None):
        assert(num_init <= np.min(self.train_images.shape))
        if(num_sample is None):
            num_sample = num_init * 5

        data = self.getNormSample(num_sample)
        [u, s, v] = np.linalg.svd(np.transpose(data))
        return np.transpose(u[:, :num_init])

    def getDict(self, num_init, alpha, num_sample=None):
        assert(num_init <= np.min(self.train_images.shape))
        if(num_sample is None):
            num_sample = 5000
        data = self.getNormSample(num_sample)
        print("Running sklearn dict_learn for initial dictionary")
        dictionary = dict_learning(data, num_init, alpha, verbose=2, n_jobs=-1, max_iter=50)[1]
        print("Done")
        return dictionary

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    path = "/home/slundquist/mountData/datasets/mnist"
    obj = mnistData(path, flatten=False)
    (train_data, train_gt) = obj.getData(10)

    plt.figure()
    r_data = train_data[0, :, :, 0]
    plt.imshow(r_data, cmap="gray")
    plt.show()
