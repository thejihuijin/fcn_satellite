Implementation of a 3 class (background, roads, buildings) semantic segmentation for aerial satelite images using a FCN architecture and Caffe. The end goal of the segmentation is region proposals for 3D points clouds. The network is accordingly tuned to overpredict roads and building because the cost for predicting background when the true class is roads or building is very high.


The following repositories were used:
1. https://github.com/shelhamer/fcn.berkeleyvision.org
2. https://github.com/mitmul/ssa

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
Download VGG Conv Weights:
  ```bash
  cd ilsvrc-nets
  # download weights
  vim caffemodel_url #follow link and name convention in file
  ```
Generate Net and Train:
  ```bash
  cd ig_fcn8
  python net.py # Optionally change the H matrix
  python solve.py
  ```