# predicting-physical-dynamics
Code for "Predicting the Physical Dynamics of Unseen 3D Objects" presented at WACV 2020.

Structure.

Setup. Only tested on Ubuntu 16.04 using Tensorflow 1.13.1

Train command.

`python scripts/train/topple_aa_train_classify.py @./data/configs/Cube5k.train.cfg`

Test command.

`python scripts/test/topple_aa_test_classify.py @./data/configs/Cube5k.test.cfg`

If you run into issues with pickle, may need to regenerate some preprocessing done on the data. For example for the cube data run:

`python scripts/data/calc_normalization_info.py --data_list ./data/sim/dataset_lists/Cube5k/all.txt --info_out ./data/sim/normalization_info/cube_5k.pkl`
