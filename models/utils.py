import tensorflow as tf
import numpy as np

def weight_variable(shape, name, std=0.01):
    initial = tf.truncated_normal(shape, stddev=std)
    return tf.Variable(initial, name=name)

def l2_weight_variable(shape, name, std=0.01):
    initial = np.random.normal(scale=std, size=shape).astype(np.float32)
    norm = np.linalg.norm(initial, axis=-1, keepdims=True)
    return tf.Variable(initial/norm, name=name)


#[-1, 1] -> [0, 255]
def normImage(img, normalize):
    with tf.name_scope("normImage"):
        #Clip img first
        if(normalize):
            img = tf.clip_by_value(img, -1.0, 1.0)
            norm_img = (img + 1)/2
        else:
            norm_img = img
        outImg = norm_img
        return tf.cast(tf.clip_by_value(outImg*255, 0, 255), tf.uint8)
