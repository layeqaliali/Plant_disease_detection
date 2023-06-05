# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
#works for health, miner, rust, phoma, cercospora
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(150, 150))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 150, 150, 3)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

def run_example():
    # load the image
    img = load_image('final_data/test/3_rust/ferr090.jpg')
    # load model
    model = load_model('final_model1.h5')
    # predict the class
    result = model.predict_classes(img)
    if result == 0:
        print('diseased coffee leaves')
        print('disease name is health')
        print('class number is:')
        print(str(result[0]))
    if result == 1:
        print('diseased coffee leaves')
        print('disease name is miner')
        print('class number is:')
        print(str(result[0]))
    if result == 2:
        print('diseased coffee leaves')
        print('disease name is rust')
        print('class number is:')
        print(str(result[0]))
    if result == 3:
        print('diseased coffee leaves')
        print('disease name is phoma')
        print('class number is:')
        print(str(result[0]))
    if result == 4:
        print('diseased coffee leaves')
        print('disease name is cercospora')
        print('class number is:')
        print(str(result[0]))
    if result == 5:
        print('diseased rice leaves')
        print('disease name is bacterial leaf blight')
        print('class number is:')
        print(str(result[0]))
    if result == 6:
        print('diseased rice leaves')
        print('disease name is brown spot')
        print('class number is:')
        print(str(result[0]))
    if result == 7:
        print('diseased rice leaves')
        print('disease name is leaf smut')
        print('class number is:')
        print(str(result[0]))
    if result == 8:
        print('healthy leaves')
        print('class number is:')
        print(str(result[0]))

run_example()