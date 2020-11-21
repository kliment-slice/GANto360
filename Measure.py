import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

class Measure:
    """Class to implement a CNN-Generator
    Args: 
        folder: folder with images
    Returns: dictionary of images
    """    

    def __init__(self, folder):
        self._folder = folder
        self._image_dict = {}

    # Define the different loss functions

    def ssim(self, critic_fake_output):
        """ ssim GAN loss
        (Generator)  -D(G(z|c))
        """
        return -tf.reduce_mean(critic_fake_output)

    def generator_loss(critic_fake_output):
        """ Wasserstein GAN loss
        (Generator)  -D(G(z|c))
        """
        return -tf.reduce_mean(critic_fake_output)


    def wasserstein_loss(critic_real_output, critic_fake_output):
        """ Wasserstein GAN loss
        (Critic)  D(G(z|c)) - D(x|c)
        """
        return tf.reduce_mean(critic_fake_output) - tf.reduce_mean(
            critic_real_output)


    def compute_gradient_penalty(critic, x, fake_x):
        
        c = tf.expand_dims(x[..., 0], -1)
        batch_size = x.get_shape().as_list()[0]
        eps_x = tf.random.uniform(
            [batch_size] + [1] * (len(x.get_shape()) - 1))  # B, 1, 1, 1, 1
        inter = eps_x * x + (1.0 - eps_x) * fake_x

        with tf.GradientTape() as g:
            g.watch(inter)
            disc_inter_output = critic((inter,c), training=True)
        grads = g.gradient(disc_inter_output, inter)
        slopes = tf.sqrt(1e-8 + tf.reduce_sum(
            tf.square(grads),
            reduction_indices=tf.range(1, grads.get_shape().ndims)))
        gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))
        
        return gradient_penalty