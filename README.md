Implementation of a 3 class (background, roads, buildings) semantic segmentation for aerial satelite images using a FCN architecture and Caffe. 

The end goal of the segmentation is region proposals for 3D points clouds. The network is accordingly tuned to overpredict roads and building because the cost for predicting background when the true class is roads or building is very high for this specific application.


Significant portions of the following repositories were used:
1. https://github.com/shelhamer/fcn.berkeleyvision.org (FCN architecture)
2. https://github.com/mitmul/ssa (data download and pre-processing)

Path setup:

  ```bash
  export PYTHONPATH="[path]/fcn_satellite/helper:$PYTHONPATH"
  ```

Data preparation:
  ```bash
  cd scripts
  # download data
  ./download.sh
  # create merged labels
  ./merged.sh
  # create label txt files for each set
  cd [path]/fcn_satellite/data/mass_merged/{train/valid/test}/map
  ls *.png>{train/valid/test}.txt
  # create input txt files for each set
  cd [path]/fcn_satellite/data/mass_merged/{train/valid/test}/sat 
  ls>{train/valid/test}.txt
  # remove filename from end of txt files above
  ```
Download VGGconv weights (alternatively generate yourself):
  ```bash
  cd ilsvrc-nets
  # download weights
  vim caffemodel_url # follow link and name convention in file
  ```

Generate net and train:
  ```bash
  cd ig_fcn8
  python net.py # optionally change the H matrix
  python solve.py
  ```

Download deploy weights:
  ```bash
  cd models
  # download weights
  vim caffemodel_url # follow link and name convention in file
  ```

Run model on custom image:
  ```bash
  python infer.py # change image file names
  ```