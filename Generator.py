import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

class Generator:
    """Class to implement a CNN-Generator
    Args: 
        folder: folder with images
    Returns: dictionary of images
    """    

    def __init__(self, folder):
        self._folder = folder
        self._image_dict = {}

    def _conv2d(layer_input, filters, f_size=4, bn=True):
    """Generator Basic Downsampling Block"""
    d = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=2,
                               padding='same')(layer_input)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    if bn:
        d = tf.keras.layers.BatchNormalization(momentum=0.8)(d)
    return d


    def _deconv2d(layer_input, pre_input, filters, f_size=4, dropout_rate=0):
        """Generator Basic Upsampling Block"""
        u = tf.keras.layers.UpSampling2D(size=2)(layer_input)
        u = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=1,
                                padding='same')(u)
        u = tf.keras.layers.BatchNormalization(momentum=0.8)(u)
        u = tf.keras.layers.ReLU()(u)

        if dropout_rate:
            u = tf.keras.layers.Dropout(dropout_rate)(u)
            
        u = tf.keras.layers.Concatenate()([u, pre_input])
        return u

        
    def build_generator(condition_input_shape=(32, 128, 1), filters=64,
                        instruments=4, latent_shape=(2, 8, 512)):
        """Buld Generator"""
        c_input = tf.keras.layers.Input(shape=condition_input_shape)
        z_input = tf.keras.layers.Input(shape=latent_shape)

        d1 = _conv2d(c_input, filters, bn=False)
        d2 = _conv2d(d1, filters * 2)
        d3 = _conv2d(d2, filters * 4)
        d4 = _conv2d(d3, filters * 8)

        d4 = tf.keras.layers.Concatenate(axis=-1)([d4, z_input])

        u4 = _deconv2d(d4, d3, filters * 4)
        u5 = _deconv2d(u4, d2, filters * 2)
        u6 = _deconv2d(u5, d1, filters)

        u7 = tf.keras.layers.UpSampling2D(size=2)(u6)
        output = tf.keras.layers.Conv2D(instruments, kernel_size=4, strides=1,
                                padding='same', activation='tanh')(u7)  # 32, 128, 4

        generator = tf.keras.models.Model([c_input, z_input], output, name='Generator')

        return generator