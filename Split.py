import random

class SplitSets:
    """Class to split indexes of images into train, validation, and test
        Called by train.py to split image indexes after loading them in LoadImages.py 
    """    
    def __init__(self):
        self.train_set = []
        self.val_set = []

    def split_indexes(self, number_images, size_train_frac, test_image_index):
        """method to split the indexes and update the attributes train_set and val_set
        Args:
            number_images (int): total number of images
            size_train_frac (float): fraction [0,1] representing percentage of images to use for training
            test_image_index (int): some phone image we want to generate a 360 from
        """        
        set_indexes = set(range(number_images))
        set_indexes.remove(test_image_index)
        train_index_set = set(random.sample(set_indexes, int(size_train_frac*number_images)))
        set_indexes.difference_update(train_index_set)
        val_index_set = set_indexes

        #check repetition in train and val; sum of elements
        assert len(train_index_set.intersection(val_index_set)) == 0
        assert (1 + len(train_index_set) + len(val_index_set)) == number_images 

        self.train_set = train_index_set
        self.val_set = val_index_set