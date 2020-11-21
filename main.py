import os
from keras.preprocessing.image import save_img
from LoadImages import LoadImages
from train import Train
# from pickledmodel import Model

"""
Main to call training or load pickle model
to predict/generate 360 image from a select phone image (path provided)

"""
def main():
    #next step----
    #keras load img
    #pickle load nn model
    # predict newly loaded img
    #save new img
    img_test_path = 'D:/360GAN/RegVid/VID_20201101_174245455 27920.jpg'
    #current step---
    #pass img path and set aside to test model on after training
    #call train script
    #or load pickle and interpolate here
    train1 = Train(img_test_path)
    train1.Execute()
    generatedImage = train1.generatedImage
    save_img('generated.jpg', generatedImage)

    return None

if __name__ == "__main__":
    main()