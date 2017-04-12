import os, numpy, PIL
from PIL import Image

# n = 20 to avoid whitespace issues in sat images
def compute_mean(my_dir,n=20):

	allfiles=os.listdir(my_dir)
	imlist=[filename for filename in allfiles if  filename[-5:] in [".tiff"]]
	imlist = sorted(imlist)
	imlist = imlist[:n]

	avg = numpy.zeros(3)

	for im in imlist:
		avg+= numpy.array(Image.open(my_dir+im),dtype=numpy.float).mean(axis=(0,1))/float(n)

	print avg

compute_mean('../data/mass_merged/train/sat/')