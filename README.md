**WARNING: The end goal of the segmentation is region proposals for 3D points clouds. The network is not tuned for pixelwise accuracy. Instead, the network is tuned to overpredict roads and buildings because the cost for incorrectly labeling a pixel is very high for this specific application. This can be seen in action in the examples directory.**

Implementation of a 3 class (background, roads, buildings) semantic segmentation for aerial satelite images using a FCN architecture and Caffe. 

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
Download VGGconv weights (optional - generate yourself):
  ```bash
  cd ilsvrc-nets
  # download weights
  vim caffemodel_url # follow link and name convention in file
  ```

[ig_fcn8] Train weights (optional):
  ```bash
  cd ig_fcn8
  python H.py # optional - change the H  matrix
  python net.py
  python solve.py
  ```

[ig_fcn8] Download pre-trained weights:
  ```bash
  cd ig_fcn8
  # download weights
  vim caffemodel_url # follow link and name convention in file
  ```

Run model on input image:
  ```bash
  python infer.py # change image file names
  ```