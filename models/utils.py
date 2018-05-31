import tensorflow as tf

def weight_variable(shape, std=0.01):
    initial = tf.truncated_normal(shape, stddev=std)
    return tf.Variable(initial)

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
