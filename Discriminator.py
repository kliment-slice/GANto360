import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

class Discriminator:
    """Class to implement a CNN-Generator
    Args: 
        folder: folder with images
    Returns: dictionary of images
    """    

    def __init__(self, folder):
        self._folder = folder
        self._image_dict = {}

    def _build_critic_layer(self, layer_input, filters, f_size=4):
        """
        This layer decreases the spatial resolution by 2:

            input:  [batch_size, in_channels, H, W]
            output: [batch_size, out_channels, H/2, W/2]
        """
        d = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=2,
                                padding='same')(layer_input)
        # Critic does not use batch-norm
        d = tf.keras.layers.LeakyReLU(alpha=0.2)(d) 
        return d


    def build_critic(pianoroll_shape=(32, 128, 4), filters=64):
        """WGAN critic."""
        
        condition_input_shape = (32,128,1)
        groundtruth_pianoroll = tf.keras.layers.Input(shape=pianoroll_shape)
        condition_input = tf.keras.layers.Input(shape=condition_input_shape)
        combined_imgs = tf.keras.layers.Concatenate(axis=-1)([groundtruth_pianoroll, condition_input])


        
        d1 = _build_critic_layer(combined_imgs, filters)
        d2 = _build_critic_layer(d1, filters * 2)
        d3 = _build_critic_layer(d2, filters * 4)
        d4 = _build_critic_layer(d3, filters * 8)

        x = tf.keras.layers.Flatten()(d4)
        logit = tf.keras.layers.Dense(1)(x)

        critic = tf.keras.models.Model([groundtruth_pianoroll,condition_input], logit,
                                            name='Critic')
        

        return critic