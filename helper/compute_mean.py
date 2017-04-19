import os, PIL
import numpy as np
from PIL import Image

# n = 20 to avoid whitespace issues in sat images
def compute_mean(my_dir):

	allfiles=os.listdir(my_dir)
	imlist=[filename for filename in allfiles if  filename[-5:] in [".tiff"]]
	imlist = sorted(imlist)

	avg = np.zeros(3)

	for im in imlist:

		img = np.array(Image.open(my_dir+im),dtype=np.float)
		avg+= mean_filt_white(img)/len(imlist)

	print avg

def mean_filt_white(my_img):
	avg = np.zeros(3)
	white_map = np.all(my_img != np.array([255,255,255]),axis=2)
	for ch in range(3):
		avg[ch] = my_img[white_map,ch].mean()
	return avg

compute_mean('data/mass_merged/train/sat/')