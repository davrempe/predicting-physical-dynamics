# Predicting the Physical Dynamics of Unseen 3D Objects

## WACV 2020

Davis Rempe, Srinath Sridhar, He Wang, Leonidas J. Guibas

![Teaser](dynamics.png)

This repository contains the code and dataset used in the paper "Predicting the Physical Dynamics of Unseen 3D Objects" presented at WACV 2020. See the [project page](https://geometry.stanford.edu/projects/learningdynamicsWACV2020/).

## Setup
This code has been tested on Ubuntu 16.04 using Python 2.7 and Tensorflow 1.13.1 with CUDA 10.0. After installing CUDA, all dependencies can be installed from the root of this repo with:

`pip install -r requirements.txt`

### Downloads
In order to run the code, you must download and place a few things in the repo:
* **All simulated data** can be downloaded from [here](https://drive.google.com/open?id=197JIPbeJFtNzG75SDnUeWf6euiXdRojt). After unzipping, all data directories should be placed in `data/sim/`. For more information on data see [the readme](data).
* Pre-trained PointNet weights that we use to initialize training can be downloaded [here](https://drive.google.com/file/d/1R8EK4EMlEGM6hMn5U9v17mheP52mTzeD/view?usp=sharing) and should be placed in the `pretrained` directory. 
* If you want to visualize results, you will need to download the shape meshes [here](https://drive.google.com/open?id=1YWrgi6Uw7G0jqVu36BvJETjWcXGe0DaM). They can be placed anywhere.

## Structure
Data explanation and example configs.

Unity project.

## Running
Below are directions to train, test, and visualize results from our method.

### Training and Testing
Train command.

`python scripts/train/topple_aa_train_classify.py @./data/configs/Cube5k.train.cfg`

Test command.

`python scripts/test/topple_aa_test_classify.py @./data/configs/Cube5k.test.cfg`

If you run into issues with pickle, may need to regenerate some preprocessing done on the data. For example for the cube data run:

`python scripts/data/calc_normalization_info.py --data_list ./data/sim/dataset_lists/Cube5k/all.txt --info_out ./data/sim/normalization_info/cube_5k.pkl`

### Visualization
We use Unity. Run this scene and put in these paths after running inference with the `--output_pred` flag.

## Citation

If you use this code, please cite our work:
```latex
@inproceedings{RempeDynamics2020,
	author={Rempe, Davis and Sridhar, Srinath and Wang, He and Guibas, Leonidas J.},
	title={Predicting the Physical Dynamics of Unseen 3D Objects},
	journal={Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
	year={2020}
}
```
