from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

"""
An object that handles data input
"""
class mnistData(object):
    raw_image_shape = (28, 28, 1)
    num_features = 784
    num_classes = 10

    #TODO set random seed
    def __init__(self, path, flatten=True):
        self.mnist = input_data.read_data_sets(path, one_hot=False)

        self.num_train_examples = self.mnist.train.num_examples
        self.num_test_examples = self.mnist.test.num_examples
        self.flatten=flatten

        self.train_images = self.mnist.train.images
        self.train_labels = self.mnist.train.labels
        self.test_images = self.mnist.test.images
        self.test_labels = self.mnist.test.labels

    def getData(self, numExample):
        images, labels = self.mnist.train.next_batch(numExample)
        if(not self.flatten):
            #Reshape into y, x, f
            images = np.reshape(images, (numExample,) + self.raw_image_shape)
        return (images, labels)

    def getTestData(self):
        test_images = self.test_images

        if(not self.flatten):
            #Reshape into y, x, f
            test_images = np.reshape(test_images,
                    (self.num_test_examples,) + self.raw_image_shape)

        return (test_images, self.test_labels)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    path = "/home/slundquist/mountData/datasets/mnist"
    obj = mnistData(path, flatten=False)
    (train_data, train_gt) = obj.getData(10)

    plt.figure()
    r_data = train_data[0, :, :, 0]
    plt.imshow(r_data, cmap="gray")
    plt.show()
