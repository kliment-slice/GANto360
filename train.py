import os
import matplotlib.pyplot as plt
from LoadImages import LoadImages
from Split import SplitSets
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
# from Generator import Generator
# from Discriminator import Discriminator
# from Metrics import MSE, SSIM

class Train():
    """Class to load images, convert them to numpy arrays and bundle them in a dict
    GAN nomenclature:
    src - source, phone images
    dst - destination, 360 images
    Args: 
        folder: folder with images
    Returns: dictionary of images
    """    

    def __init__(self, test_path):
        self.test_path = test_path
        self.dst_train_dict = dict()
        self.dst_val_dict = dict()
        self.src_train_dict = dict()
        self.src_val_dict = dict()
        self.test_array = np.array()

    def Execute(self):
        self.PrepImages()
        self.TrainGAN()

    def loadTestImage(self):
        test_image = load_img(f'{self.test_path}')
        test_array = img_to_array(test_image)
        test_index = [i for i, x in enumerate(os.listdir('D:/360GAN/RegVid_100')) if x == 'VID_20201101_174245455 27920.jpg'][0]
        return test_index, test_array

    def PrepImages(self):
        path_dst = 'D:/360GAN/360Vid_100'
        path_src = 'D:/360GAN/RegVid_100'
        test_index, self.test_array = self.loadTestImage()
        dst_dict = LoadImages(path_dst).load_images()
        src_dict = LoadImages(path_src).load_images()

        #drop test image from  set
        del src_dict[test_index]
        del dst_dict[test_index]
        
        assert len(dst_dict) == len(src_dict), 'size mismatch (# of images dst != src)'
        n_images = len(src_dict)
        #image index to test on (counting starts from 0)

        # this particular split
        split1 = SplitSets()
        train_set, val_set = split1.split_indexes(number_images=n_images, size_train_frac=0.8)

        self.dst_train_dict = {train_key: dst_dict[train_key] for train_key in list(train_set)}
        self.dst_val_dict = {val_key: dst_dict[val_key] for val_key in list(val_set)}
        self.src_train_dict = {train_key: src_dict[train_key] for train_key in list(train_set)}
        self.src_val_dict = {val_key: src_dict[val_key] for val_key in list(val_set)}

        assert len(self.dst_train_dict) == len(self.src_train_dict), 'size mismatch (# of images in training dst != src)'
        assert len(self.dst_val_dict) == len(self.src_val_dict), 'size mismatch (# of images in validation dst != src)'
        assert (len(self.dst_train_dict) + len(self.dst_val_dict)) == len(dst_dict), 'size mismatch (# of images in training and validation != total images)'
       
    def TrainGAN(self):
        path_dst = 'D:/360GAN/360Vid_100'

