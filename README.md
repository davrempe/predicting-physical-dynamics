# Predicting the Physical Dynamics of Unseen 3D Objects

## WACV 2020

Davis Rempe, Srinath Sridhar, He Wang, Leonidas J. Guibas

![Teaser](dynamics.png)

This repository contains the code and dataset used in the paper "Predicting the Physical Dynamics of Unseen 3D Objects" presented at WACV 2020. See the [project page](https://geometry.stanford.edu/projects/learningdynamicsWACV2020/).

## Setup
This code has been tested on Ubuntu 16.04 using Python 2.7 and Tensorflow 1.13.1 with CUDA 10.0.

Downloads.

## Structure
Example configs.

## Running
Train command.

`python scripts/train/topple_aa_train_classify.py @./data/configs/Cube5k.train.cfg`

Test command.

`python scripts/test/topple_aa_test_classify.py @./data/configs/Cube5k.test.cfg`

If you run into issues with pickle, may need to regenerate some preprocessing done on the data. For example for the cube data run:

`python scripts/data/calc_normalization_info.py --data_list ./data/sim/dataset_lists/Cube5k/all.txt --info_out ./data/sim/normalization_info/cube_5k.pkl`

If you use this code, please cite our work:
```latex
@inproceedings{RempeDynamics2020,
	author={Rempe, Davis and Sridhar, Srinath and Wang, He and Guibas, Leonidas J.},
	title={Predicting the Physical Dynamics of Unseen 3D Objects},
	journal={Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
	year={2020}
}
```
