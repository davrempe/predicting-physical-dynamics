import numpy as np 
import os.path
from os import listdir, mkdir
from os.path import isfile, isdir, join, exists
import json
import csv

from transforms3d.quaternions import qeye, qinverse, qmult, qconjugate

class ToppleData(object):
    '''
    Structure to hold toppling simulation data.
    '''

    def __init__(self):
        #
        # global data - only a single value
        #

        # path to data roots
        self.roots = None
        # number of simulations in the data
        self.size = None

        #
        # per-shape data - one value per object we simulate
        #

        # point clouds for each object
        self.point_cloud = None

        # 
        # per-simulation data - one entry per simulation
        #

        # name of the simulated object
        self.shape_name = None
        # index in per-shape data that this sim corresponds to
        self.shape_idx = None
        # amount object was scaled (x, y, z)
        self.scale = None
        # impulse vector applied at the beginning of simulation
        self.force_vec = None
        # (x,y,z) position where impulse was applied
        self.force_pos = None
        # density of the object in kg/m^3
        self.density = None
        # friction of the object
        self.body_friction = None
        # mass of the object in kg
        self.mass = None
        # moment of inertia of the object around principle (x, y, z) axes
        self.inertia = None
        # whether the object tumbled during simulation
        self.toppled = None
        # the index in the data lists for this simulation that the object was considered toppled
        self.topple_idx = None

        # the following have a list of values, one for each timestep:
        # linear object velocities 
        self.lin_vel = None
        # angular object velocities about principle axes
        self.ang_vel = None
        # positions (of center of mass)
        self.pos = None
        # rotations (quaternions)
        self.rot = None
        # rotations (euler angles)
        self.rot_euler = None
        # cummulative rotations (euler angles)
        self.total_rot = None
        
        # delta rotations in quaternion representation (4-vec)
        self.delta_quat = None
        # delta rotations in axis-angle representation (single 3-vec)
        self.delta_rot = None
        # delta rotation ins axis-angle split representation (4-vec)
        self.delta_rot_split = None

class ToppleDataLoader(object):
    '''
    Class that loads in toppling simulation data from a given path into data structures. 
    It's "static" in that you can load any number of datasets with the same loader instance.
    '''

    def __init__(self):
        self.obj_dir = 'objects'
        self.sim_dir = 'sims'

    def load_json_vec(self, vec_dict):
        ''' Loads a json 3 (x, y, z) or 4 (x, y, z, w) vector into a numpy array '''
        np_vec = np.zeros((len(vec_dict)))
        if len(vec_dict) == 3:
            np_vec = np.array([vec_dict['x'], vec_dict['y'], vec_dict['z']], dtype=float)
        elif len(vec_dict) == 4:
            np_vec = np.array([vec_dict['w'], vec_dict['x'], vec_dict['y'], vec_dict['z']], dtype=float)
        return np_vec

    def load_json_vec_list(self, vec_list):
        return [self.load_json_vec(vec_dict) for vec_dict in vec_list]

    def load_data(self, roots):
        '''
        Loads the toppling data at the given path and returns a data object.

        The given 'roots' is a list of dataset directories to load into a dataset, each
        should contain an 'objects' directory which containts .pts files
        for each unique simulated objects, and a 'sims' directory which contains a directory
        for each unique object that holds .json files describing each performed simulation.
        '''
        data = ToppleData()
        data.roots = roots

        # load all point clouds from objects dir
        print('Loading point clouds...')
        pts_files = []
        for root in data.roots:
            points_path = join(root, self.obj_dir)
            all_files = [f for f in sorted(listdir(points_path)) if isfile(join(points_path, f))]
            pts_files += [join(points_path, f) for f in all_files if f.split('.')[-1] == 'pts']
        num_shapes = len(pts_files)

        # print(pts_files)

        if num_shapes == 0:
            print('No .pts files to load!')
            return None

        # peek at first .pts file to determine num_pts (assumes same for all objects)
        num_pts = -1
        with open(pts_files[0], 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                num_pts = 0
                for row in reader:
                    num_pts += 1
        print('Found ' + str(num_pts) + ' points per object.')

        # actually load point cloud data
        data.point_cloud = np.zeros((num_shapes, num_pts, 3))
        for i, pc_file in enumerate(pts_files):
            with open(pc_file, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                pt_cnt = 0
                for j, row in enumerate(reader):
                    pt = [float(x) for x in row[0:3]]
                    data.point_cloud[i, j, :] = pt
                    pt_cnt += 1
                if pt_cnt != num_pts:
                    print('Error: File ' + pc_file + ' contains ' + str(pt_cnt) + ' points!')

        print('Point clouds successfully loaded!')

        # set data size by taking first pass through data
        sim_obj_count = 0
        for root in data.roots:
            sims_path = join(root, self.sim_dir)
            sim_obj_count += len(listdir(sims_path))

        if sim_obj_count != num_shapes:
            print('Mismatch in number of .pts files and simulated objects! Only loading data for objects with a .pts file.')
        
        data_size = 0
        obj_sim_dirs = [f.replace('.pts', '').replace(self.obj_dir, self.sim_dir) for f in pts_files]
        for obj_dir in obj_sim_dirs:
            sim_all_files = [f for f in listdir(obj_dir) if isfile(join(obj_dir, f))]
            sim_json_files = [join(obj_dir, f) for f in sim_all_files if f.split('.')[-1] == 'json']
            data_size += len(sim_json_files)

        print('Found ' + str(data_size) + ' total simulations.')
        data.size = data_size

        # print(obj_sim_dirs)

        # actually load all the simulation data
        print('Loading simulation data...')
        data.shape_name = [] 
        data.shape_idx = np.zeros((data_size), dtype=np.int32)
        data.scale = np.zeros((data_size, 3), dtype=float)
        data.force_vec = np.zeros((data_size, 3), dtype=float)
        data.force_pos = np.zeros((data_size, 3), dtype=float)
        data.density = np.zeros((data_size), dtype=float)
        data.body_friction = np.zeros((data_size), dtype=float)
        data.mass = np.zeros((data_size), dtype=float)
        data.inertia = np.zeros((data_size, 3))
        data.toppled = []
        data.topple_idx = []

        # time-series data
        data.lin_vel = []
        data.ang_vel = []
        data.pos = []
        data.rot = []
        data.rot_euler = []
        data.total_rot = []
        data.delta_rot = []
        data.delta_rot_split = []
        data.delta_quat = []

        step_sum = 0
        cur_idx = 0
        for i, obj_dir in enumerate(obj_sim_dirs):
            sim_all_files = [f for f in listdir(obj_dir) if isfile(join(obj_dir, f))]
            sim_json_files = [join(obj_dir, f) for f in sim_all_files if f.split('.')[-1] == 'json']
            # go through and renumber so they can be sorted in the correct order
            sim_json_renumbered = []
            for j, json_file_path in enumerate(sim_json_files):
                path_prefix = '_'.join(json_file_path.split('_')[:-1]) + '_'
                sim_num = int(json_file_path.split('_')[-1].split('.')[0])
                new_num_str = '%06d' % (sim_num)
                sim_json_renumbered.append(path_prefix + new_num_str + '.json')
            sorted_order =  [j[0] for j in sorted(enumerate(sim_json_renumbered), key=lambda x:x[1])]
            sim_json_files_sorted = [sim_json_files[idx] for idx in sorted_order]
            sim_json_files = sim_json_files_sorted
            # print(sim_json_files)
            for j, sim_file in enumerate(sim_json_files):
                with open(sim_file, 'r') as f:
                    sim_dict = json.loads(f.readline())
                    data.shape_name.append(sim_dict['shape'])
                    data.shape_idx[cur_idx] = i
                    data.scale[cur_idx] = self.load_json_vec(sim_dict['scale'])
                    data.force_vec[cur_idx] = self.load_json_vec(sim_dict['forceVec'])
                    data.force_pos[cur_idx] = self.load_json_vec(sim_dict['forcePoint'])
                    data.density[cur_idx] = float(sim_dict['density'])
                    data.mass[cur_idx] = float(sim_dict['mass'])
                    data.inertia[cur_idx] = self.load_json_vec(sim_dict['inertia'])

                    data.lin_vel.append(self.load_json_vec_list(sim_dict['stepVel']))     
                    data.ang_vel.append(self.load_json_vec_list(sim_dict['stepAngVel']))
                    data.pos.append(self.load_json_vec_list(sim_dict['stepPos']))

                    data.rot.append(self.load_json_vec_list(sim_dict['stepRot']))

                    if 'deltaQuat' in sim_dict:
                        data.delta_quat.append(self.load_json_vec_list(sim_dict['deltaQuat']))

                    data.rot_euler.append(self.load_json_vec_list(sim_dict['stepEulerRot']))
                    data.total_rot.append(self.load_json_vec_list(sim_dict['stepTotalRot']))
                    step_sum += len(data.lin_vel[-1])

                    data.toppled.append(sim_dict['tumbled'])
                    if 'tumbleIdx' in sim_dict:
                        data.topple_idx.append(int(sim_dict['tumbleIdx']))

                    # only load axis-angle info if it's acutally in the file (for backwards compat)
                    if 'deltaAngleAxis' in sim_dict:
                        data.delta_rot.append(self.load_json_vec_list(sim_dict['deltaAngleAxis']))
                    if 'deltaAngleAxisSplit' in sim_dict:
                        data.delta_rot_split.append(self.load_json_vec_list(sim_dict['deltaAngleAxisSplit']))
                    if 'bodyFriction' in sim_dict:
                        data.body_friction[cur_idx] = float(sim_dict['bodyFriction'])

                    cur_idx += 1
                
        data.shape_name = np.array(data.shape_name)
        step_sum /= float(cur_idx)
        print('Simulation data loaded!')
        print('Avg num timesteps: ' + str(step_sum))
        topple_count = len(np.nonzero(np.array(data.toppled))[0])
        print('Num topple: ' + str(topple_count))

        return data