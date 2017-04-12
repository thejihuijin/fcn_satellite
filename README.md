1. ./download.sh
2. ./prepare.sh
3. $CAFFE_ROOT/tools/compute_image_mean [path]/fcn_satellite/data/mass_merged/lmdb/train_sat [path]fcn_satellite/data/mass_merged/mean_image.binaryproto
4. cd [path]/fcn_satellite/data/mass_merged/{train/valid/test}/map --> ls *.png>{train/valid/test}.txt 
5. cd [path]/fcn_satellite/data/mass_merged/{train/valid/test}/sat --> ls>{train/valid/test}.txt 