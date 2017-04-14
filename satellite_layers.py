import caffe

import numpy as np
from PIL import Image
import scipy.io
import random

class SatelliteDataLayer(caffe.Layer):

    def setup(self, bottom, top):

        # config
        params = eval(self.param_str)
        self.input_dir = params['input_dir']
        self.output_dir = params['output_dir']
        self.mean = np.array((95.58390035,97.0751154,89.11036257), dtype=np.float32)
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.data_set = params.get('data_set')
        self.text_ind = self.input_dir+'/'+self.data_set+'.txt'

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        self.indices = open(self.text_ind, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if self.data_set == 'valid':
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data,self.label = self.load_image(self.indices[self.idx])

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        Load output image (no preprocess)
        """

        im = Image.open('{}/{}.tiff'.format(self.input_dir, idx[:-5]))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))

        label = np.array(Image.open('{}/{}.png'.format(self.output_dir, idx[:-5])),dtype=np.int32).transpose((2,0,1))/255
        label = label[0,:,:]*0 + label[1,:,:]*1+ label[2,:,:]*255

        in_,label = self.random_patch(in_,label)


        return in_,label

    def random_patch(self,img,lbl,h=256,w=256, sz = 1500):
        x = random.randint(0, sz-h)
        y = random.randint(0, sz-w)
        return img[:,x:x+h,y:y+w], lbl[x:x+h,y:y+w]