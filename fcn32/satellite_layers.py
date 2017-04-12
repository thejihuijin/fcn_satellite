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
        self.mean = np.array((76.7005894, 78.12125889, 67.95231933), dtype=np.float32)
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.data_set = params.get('data_set')

        print self.data_set

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        raise Exception('stopped')

        # load indices for images and labels
        split_f  = '{}/ImageSets/Main/{}.txt'.format(self.input_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if self.data_set == 'valid'
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
        """
        im = Image.open('{}/JPEGImages/{}.jpg'.format(self.input_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))

        label = np.array(Image.open('{}/JPEGImages/{}.jpg'.format(self.output_dir, idx)),dtype=np.float32)

        return in_,label

    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        The full 400 labels are translated to the 59 class task labels.
        """
        label_400 = scipy.io.loadmat('{}/trainval/{}.mat'.format(self.output_dir, idx))['LabelMap']
        label = np.zeros_like(label_400, dtype=np.uint8)
        for idx, l in enumerate(self.labels_59):
            idx_400 = self.labels_400.index(l) + 1
            label[label_400 == idx_400] = idx + 1
        label = label[np.newaxis, ...]
        return label