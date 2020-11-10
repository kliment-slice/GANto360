import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

class LoadImages:
    """Class to load images, convert them to numpy arrays and bundle them in a dict
    Args: 
        folder: folder with images
    Returns: dictionary of images
    """    

    def __init__(self, folder):
        self._folder = folder
        self._image_dict = {}

    def load_images(self):
        data = {}
        folder = self._folder
        for index, image in enumerate(os.listdir(folder)):
            raw_photo = load_img(f'{folder}/{image}')
            converted_photo = img_to_array(raw_photo)
            data[index] = converted_photo
        self._image_dict = data
        return data
