import argparse
from os.path import exists
import numpy as np 
from topple_data_loader import ToppleDataLoader
from topple_dataset import ToppleNormalizationInfo
import data_list_loader

'''
Script to calculate normalization info across multiple datasets. This information
can then be used to create multiple ToppleDataset instances that are all normalized
the same.
'''

parser = argparse.ArgumentParser()
parser.add_argument('--data_list', help='file containing list of datasets to calculate combined normalization info for.', required=True)
parser.add_argument('--info_out', help='.pkl file to write out norm info file to.', required=True)
flags = parser.parse_args()

dataset_list = flags.data_list
out_path = flags.info_out

datasets = data_list_loader.load_data_list(dataset_list)
print(datasets)

# initialize normalization info
norm_inf = ToppleNormalizationInfo()
inf = float('inf')
norm_inf.max_lin_vel = -inf
norm_inf.max_ang_vel = -inf
norm_inf.max_pos = -inf
norm_inf.max_rot = -inf
norm_inf.max_delta_rot = -inf
norm_inf.force_vec_max = -inf
norm_inf.pc_max = -inf
norm_inf.density_offset = inf
norm_inf.density_max = -inf
norm_inf.mass_offset = inf
norm_inf.mass_max = -inf
norm_inf.inertia_offset = inf
norm_inf.inertia_max = -inf
norm_inf.friction_offset = inf
norm_inf.friction_max = -inf

# go through each dataset, track normalization info across all datasets (find maxima)
data_loader = ToppleDataLoader()
for dataset in datasets:
    print(dataset)
    if not exists(dataset):
        print('Could not find dataset ' + str(dataset) + '!')
        quit()
    print('Finding normalization info for ' + str(dataset) + '...')
    # load the dataset
    data = data_loader.load_data([dataset])
    
    # max linear velocity
    for lin_vel_steps in data.lin_vel:
        lin_vel_arr = np.array(lin_vel_steps)
        cur_max = np.max(np.abs(lin_vel_arr))
        if norm_inf.max_lin_vel < cur_max:
            norm_inf.max_lin_vel = cur_max

    # max angular veloicty
    for ang_vel_steps in data.ang_vel:
        ang_vel_arr = np.array(np.abs(ang_vel_steps))
        cur_max = np.max(ang_vel_arr)
        if norm_inf.max_ang_vel < cur_max:
            norm_inf.max_ang_vel = cur_max
    
    # max delta pos
    for pos_steps in data.pos:
        pos_arr = np.array(pos_steps)
        delta_pos = pos_arr[1:, :] - pos_arr[:-1, :]
        max_dist = np.max(np.linalg.norm(delta_pos, axis=1))
        if norm_inf.max_pos < max_dist:
            norm_inf.max_pos = max_dist

    # max change in total rot
    for rot_steps in data.total_rot:
        rot_arr = np.array(rot_steps)
        delta_rot = np.abs(rot_arr[1:, :] - rot_arr[:-1, :])
        max_rot = np.max(delta_rot)
        if norm_inf.max_rot < max_rot:
            norm_inf.max_rot = max_rot

    # max delta_rot in axis-angle
    for delta_rot_steps in data.delta_rot:
        delta_rot_arr = np.array(delta_rot_steps)
        max_angle = np.max(np.linalg.norm(delta_rot_arr, axis=1))
        if norm_inf.max_delta_rot < max_angle:
            norm_inf.max_delta_rot = max_angle
    
    # max force vec
    force_vec_data_max = np.max(np.linalg.norm(data.force_vec, axis=1))
    if norm_inf.force_vec_max < force_vec_data_max:
        norm_inf.force_vec_max = force_vec_data_max

    # max point cloud point norm
    data_scale_max = np.max(data.scale)
    pc = np.copy(data.point_cloud)*data_scale_max
    data_pc_max = np.max(np.linalg.norm(pc, axis=2))
    if norm_inf.pc_max < data_pc_max:
        norm_inf.pc_max = data_pc_max
    
    # auxillary shape-related stuff
    # density
    data_density_offset = np.min(data.density)
    if norm_inf.density_offset > data_density_offset:
        norm_inf.density_offset = data_density_offset
    data_density_max = np.max(data.density)
    if norm_inf.density_max < data_density_max:
        norm_inf.density_max = data_density_max
    # mass
    data_mass_offset = np.min(data.mass)
    if norm_inf.mass_offset > data_mass_offset:
        norm_inf.mass_offset = data_mass_offset
    data_mass_max = np.max(data.mass)
    if norm_inf.mass_max < data_mass_max:
        norm_inf.mass_max = data_mass_max
    # inertia
    data_inertia_offset = np.min(data.inertia)
    if norm_inf.inertia_offset > data_inertia_offset:
        norm_inf.inertia_offset = data_inertia_offset
    data_inertia_max = np.max(data.inertia)
    if norm_inf.inertia_max < data_inertia_max:
        norm_inf.inertia_max = data_inertia_max
    # friction
    data_friction_offset = np.min(data.body_friction)
    if norm_inf.friction_offset > data_friction_offset:
        norm_inf.friction_offset = data_friction_offset
    data_friction_max = np.max(data.body_friction)
    if norm_inf.friction_max < data_friction_max:
        norm_inf.friction_max = data_friction_max


# account for constant density/mass/inertia
norm_inf.density_max -= norm_inf.density_offset
if norm_inf.density_max == 0:
    norm_inf.density_max = norm_inf.density_offset
    norm_inf.density_offset = 0
norm_inf.mass_max -= norm_inf.mass_offset
if norm_inf.mass_max == 0:
    norm_inf.mass_max = norm_inf.mass_offset
    norm_inf.mass_offset = 0
norm_inf.inertia_max -= norm_inf.inertia_offset
if norm_inf.inertia_max == 0:
    norm_inf.inertia_max = norm_inf.inertia_offset
    norm_inf.inertia_offset = 0
norm_inf.friction_max -= norm_inf.friction_offset
if norm_inf.friction_max == 0:
    norm_inf.friction_max = norm_inf.friction_offset
    norm_inf.friction_offset = 0

# save normalization info
norm_inf.print_out()
norm_inf.save(out_path)
