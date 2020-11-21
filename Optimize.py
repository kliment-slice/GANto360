import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

class Optimizer:
    """Class to implement a CNN-Generator
    Args: 
        folder: folder with images
    Returns: dictionary of images
    """    

    def __init__(self, folder):
        self._folder = folder
        self._image_dict = {}

    # Define the different loss functions

    def adam(self, critic_fake_output):
        """ ssim GAN loss
        (Generator)  -D(G(z|c))
        """
        # Setup Adam optimizers for both G and D
        generator_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.5, beta_2=0.9)
        critic_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.5, beta_2=0.9)

        # We define our checkpoint directory and where to save trained checkpoints
        ckpt = tf.train.Checkpoint(generator=generator,
                                generator_optimizer=generator_optimizer,
                                critic=critic,
                                critic_optimizer=critic_optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, check_dir, max_to_keep=5)
