#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Code from: https://github.com/mitmul/ssai

import argparse
import glob
import os
import shutil
import time
import caffe
import numpy as np
import cv2 as cv
import lmdb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()


def create_merged_map():
    
    # copy sat images
    for data_type in ['train', 'test', 'valid']:
        out_dir = '../data/mass_merged/%s/sat' % data_type
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for fn in glob.glob('../data/mass_buildings/%s/sat/*.tiff' % data_type):
            shutil.copy(fn, '%s/%s' % (out_dir, os.path.basename(fn)))

    road_maps = dict([(os.path.basename(fn).split('.')[0], fn)
                      for fn in glob.glob('../data/mass_roads/*/map/*.tif')])

    # combine map images
    for data_type in ['train', 'test', 'valid']:
        out_dir = '../data/mass_merged/%s/map' % data_type
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for fn in glob.glob('../data/mass_buildings/%s/map/*.tif' % data_type):
            base = os.path.basename(fn).split('.')[0]
            building_map = cv.imread(fn, cv.IMREAD_GRAYSCALE)
            road_map = cv.imread(road_maps[base], cv.IMREAD_GRAYSCALE)
            _, building_map = cv.threshold(
                building_map, 0, 1, cv.THRESH_BINARY)
            _, road_map = cv.threshold(road_map, 0, 1, cv.THRESH_BINARY)
            h, w = road_map.shape
            merged_map = np.zeros((h, w))
            merged_map += building_map
            merged_map += road_map * 2
            merged_map = np.where(merged_map > 2, 2, merged_map)
            cv.imwrite('../data/mass_merged/%s/map/%s.tif' % (data_type, base),
                       merged_map)
            print(merged_map.shape, fn)
            merged_map = np.array([np.where(merged_map == 0, 1, 0),
                                   np.where(merged_map == 1, 1, 0),
                                   np.where(merged_map == 2, 1, 0)])
            merged_map = merged_map.swapaxes(0, 2).swapaxes(0, 1)
            cv.imwrite('../data/mass_merged/%s/map/%s.png' % (data_type, base),
                       merged_map * 255)

if __name__ == '__main__':

    if args.dataset == 'multi':
        create_merged_map()