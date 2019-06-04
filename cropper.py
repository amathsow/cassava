from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


base_train = '/home/pjk/cassava/train'
base_valid = '/home/pjk/cassava/validation'
base_test  = '/home/pjk/cassava/test/0'
folder_list = ['/cbb', '/cbsd', '/cgm', '/cmd', '/healthy']




def crop(im):
    if (im.size[1] >= im.size[0]):
        crop_shift = int((im.size[1] - im.size[0])/2)
        im = im.crop((0, crop_shift, im.size[0], im.size[0] + crop_shift))
    elif (im.size[0] > im.size[1]):
        crop_shift = int((im.size[0] - im.size[1])/2)
        im = im.crop((crop_shift, 0, im.size[1] + crop_shift, im.size[1]))
    im = im.resize((224,224), resample=Image.LANCZOS)
    return im

# test images
for image_name in os.listdir(base_test):
	image_filepath = base_test + '/' + image_name
	pil_img = Image.open(image_filepath)
	cropped_img = crop(pil_img)
	cropped_img.save(image_filepath)


#train or validation
for folder in folder_list:
	print(folder)
	for image_name in os.listdir(base_valid + folder):
		image_filepath = base_valid + folder + '/' + image_name
		pil_img = Image.open(image_filepath)
		cropped_img = crop(pil_img)
		cropped_img.save(image_filepath)

for folder in folder_list:
	print(folder)
	for image_name in os.listdir(base_train + folder):
		image_filepath = base_train + folder + '/' + image_name
		pil_img = Image.open(image_filepath)
		cropped_img = crop(pil_img)
		cropped_img.save(image_filepath)