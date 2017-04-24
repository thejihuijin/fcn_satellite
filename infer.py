
import numpy as np
import scipy.misc
from PIL import Image

import caffe
import sys

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
raw_img = Image.open('data/mass_merged/test/sat/22828930_15.tiff')
img = np.array(raw_img, dtype=np.float32)
img = img[0:500,0:500,:] # subsample
img = img[:,:,::-1]
img -= np.array((80.76832175,82.40158693,73.67652711))
img = img.transpose((2,0,1))

# init
caffe.set_device(0)
caffe.set_mode_gpu()

# load net
net = caffe.Net('ig_fcn8/deploy.prototxt', 'models/deploy.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *img.shape)
net.blobs['data'].data[...] = img
# run net and take argmax for prediction

net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)

img = np.zeros((out.shape[0],out.shape[1],3))
img[out == 0,2] = 255
img[out == 1,0] = 255
img[out == 2,1] = 255

scipy.misc.imsave('examples/net_1_10_5.jpg', img)

im = Image.open('data/mass_merged/test/map/22828930_15.png')
in_ = np.array(im, dtype=np.float32)
in_ = in_[0:500,0:500,:]
scipy.misc.imsave('examples/gt.jpg', in_)