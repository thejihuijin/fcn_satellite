#https://github.com/shelhamer/fcn.berkeleyvision.org

import caffe
import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = '../ilsvrc-nets/vgg16fc.caffemodel'

# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# check weights
for k,v in solver.net.params.iteritems():
    print k, np.mean(np.abs(v[0].data))

# scoring
val = np.loadtxt('../data/mass_merged/valid/sat/valid.txt', dtype=str)

for _ in range(100):
	score.seg_tests(solver, False, val, layer='score')
	solver.step(500)