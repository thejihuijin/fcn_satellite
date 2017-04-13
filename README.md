
Path setup:

  ```bash
  export PYTHONPATH="[path]/fcn_satellite:$PYTHONPATH"
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