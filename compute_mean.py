import os, numpy, PIL
from PIL import Image

# n = 20 to avoid whitespace issues in sat images
def compute_mean(my_dir):

	allfiles=os.listdir(my_dir)
	imlist=[filename for filename in allfiles if  filename[-5:] in [".tiff"]]
	imlist = sorted(imlist)

	avg = numpy.zeros(3)

	for im in imlist:
		avg+= numpy.array(Image.open(my_dir+im),dtype=numpy.float).mean(axis=(0,1))/len(imlist)

	print avg

compute_mean('data/mass_merged/train/sat/')