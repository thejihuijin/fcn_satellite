#! /bin/bash
export PYTHONPATH=".":$PYTHONPATH
python create_dataset.py --dataset multi
python create_dataset.py --dataset lmdb