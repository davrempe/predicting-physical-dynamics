## Data

All simulated data used in the paper can be downloaded by running `wget http://download.cs.stanford.edu/orion/predicting_physical_dynamics/SimData.zip` (1 GB). After unzipping, all contained data directories should be placed directly in the `sim` directory. There is one directory for each object category which contains simulation splits for that category. 

### Data Structure
The files in `dataset_lists` point to which data to load for each corresponding dataset, including the _Combined_ data which combines all simulations from all object categories together. A dataset list must be given as input to the training and test script. 

Additionally, a pickle file with some pre-computed normalization information must also be given as input. Though these pickle files are already given in the `normalization_info` directory, it may be necessary to re-compute them depending on your environment. To do this, from the root directory of this repo, run something like:

`python scripts/data/calc_normalization_info.py --data_list ./data/sim/dataset_lists/Cube5k/all.txt --info_out ./data/sim/normalization_info/cube_5k.pkl`

This example is specifically for the _Cube_ dataset, but is easily modified for other datasets by simply replacing `Cube5k` with the name of a different dataset (i.e. `Speakers10k_ObjSplit`). The term `ObjSplit` means the data is pre-split by unique objects so no training objects are seen at test time.

### Train and Test Configurations

Example training and testing configurations are contained in the `configs` directory. These are simply files containing all flags to be passed into the training and testing scripts. It is straightforward to modify these example configurations for different datasets or to change various parameters.

For training, the commands that will change between datasets are `--data_list` and `--norm_info` flags which are configured differently for each dataset as explained above. The rest are self-explanatory.

For testing, the flags determine which evaluations are performed:
* `--test_single_step` calculates _single step_ errors for the predicted sequences.
* `--test_roll_out` calculates _roll-out_ errors for the predicted sequences.
* `--test_topple_classify` evaluates the topple classification accuracy for the predicted sequences.
* `--output_pred` will save the predicted sequences to .json files which can later be visualized with Unity as [explained in the README](https://github.com/davrempe/predicting-physical-dynamics).

Please see the main paper for a description of each of the metrics.
