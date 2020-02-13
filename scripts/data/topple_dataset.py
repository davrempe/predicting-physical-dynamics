import numpy as np
import pickle
from os.path import exists, realpath
import sys
import math
from topple_data_loader import ToppleData, ToppleDataLoader
import transforms3d

class ToppleNormalizationInfo():
    '''
        Structure to hold all the normalization information for a dataset.
    '''

    def __init__(self):
        # max element of any linear vel vector
        self.max_lin_vel = None
        # max element of any angular vel vector
        self.max_ang_vel = None
        # max distance between positions in two contiguous timesteps
        self.max_pos = None
        # max change in rotation around any axis between two contiguous timesteps (for euler rot)
        self.max_rot = None
        # max angle of rotation between two steps for axis-angle representation
        self.max_delta_rot = None
        # max 2-norm of applied impulse vector
        self.force_vec_max = None
        # max 2-norm of a point in an object point cloud (used for point cloud and force pos)
        self.pc_max = None
        # normalization values for shape-related stuff
        self.density_offset = None
        self.density_max = None
        self.mass_offset = None
        self.mass_max = None
        self.inertia_offset = None
        self.inertia_max = None
        self.friction_offset = None
        self.friction_max = None

    def print_out(self):
        print({'max_lin_vel' : self.max_lin_vel, 'max_ang_vel' : self.max_ang_vel, 'max_pos' : self.max_pos, \
               'max_rot' : self.max_rot, 'max_delta_rot' : self.max_delta_rot, 'force_vec_max' : self.force_vec_max, 'pc_max' : self.pc_max, \
               'density_off' : self.density_offset, 'density_max' : self.density_max, 'mass_off' : self.mass_offset, \
               'mass_max' : self.mass_max, 'inertia_off' : self.inertia_offset, 'inertia_max' : self.inertia_max, \
                'friction_off' : self.friction_offset, 'friction_max' : self.friction_max
              })

    def save(self, pkl_file):
        ''' Saves normalization info object to a specified .pkl file. '''
        with open(pkl_file, 'wb') as f:
            pickle.dump(self, f)

    def load_from(self, pkl_file):
        ''' Load normalization info into this object from a specified .pkl file. '''
        with open(pkl_file, 'rb') as f:
            norm_info = pickle.load(f)
            self.copy_from(norm_info)

    def copy_from(self, norm_info):
        '''
        Takes values from the given normalization info object and copies them to this one
        '''
        self.max_lin_vel = norm_info.max_lin_vel
        self.max_ang_vel = norm_info.max_ang_vel
        self.max_pos = norm_info.max_pos
        self.max_rot = norm_info.max_rot
        try:
            self.max_delta_rot = norm_info.max_delta_rot
        except:
            # old versions of data doesn't have max delta rot
            pass
        self.force_vec_max = norm_info.force_vec_max
        self.pc_max = norm_info.pc_max
        self.density_offset = norm_info.density_offset
        self.density_max = norm_info.density_max
        self.mass_offset = norm_info.mass_offset
        self.mass_max = norm_info.mass_max
        self.inertia_offset = norm_info.inertia_offset
        self.inertia_max = norm_info.inertia_max
        try:
            self.friction_offset = norm_info.friction_offset
            self.friction_max = norm_info.friction_max
        except:
            # old version doesn't have this
            pass

class ToppleBatch(object):
    '''
    Structure to hold a single batch of data.
    '''

    def __init__(self, size, seq_len, num_pts):
        self.size = size
        self.num_steps = seq_len
        self.num_pts = num_pts

        self.point_cloud = np.zeros((self.size, self.num_pts, 3))
        self.lin_vel = np.zeros((self.size, self.num_steps, 3))
        self.ang_vel = np.zeros((self.size, self.num_steps, 3))
        self.pos = np.zeros((self.size, self.num_steps, 3))
        # cummulative euler angles
        self.rot = np.zeros((self.size, self.num_steps, 3))
        # change in rotation in quaternion rep (w, x, y, z)
        self.delta_quat = np.zeros((self.size, self.num_steps, 4))
        # change in rotation between steps in axis-angle rep (scaled 3 vec)
        self.delta_rot = np.zeros((self.size, self.num_steps, 3))
        # change in rotation between steps in split axis-angle rep (4-vec)
        self.delta_rot_split = np.zeros((self.size, self.num_steps, 4))

        # 0 if before topple idx, 1 if after
        self.topple_label = np.zeros((self.size, self.num_steps), dtype=int)

        # other meta-data not directly used in network
        self.toppled = []
        self.shape_name = []
        self.body_friction = np.zeros((self.size))
        self.mass = np.zeros((self.size))
        self.scale = np.zeros((self.size, 3))
        self.rot_euler = np.zeros((self.size, self.num_steps, 3))

class ToppleDataset(object):
    '''
    Loads toppling data and provides batches for training and model evaluation.
    '''

    def __init__(self, roots, norm_info_file, batch_size=32, num_steps=15, shuffle=False, num_pts=None, perturb_pts=0.0):
        '''
        - roots : list of directories containing data to load for this dataset
        - norm_info_file : .pkl file containing normalization information
        - batch_size : number of sequences to return in each batch
        - num_steps : number of timesteps to return in each sequence
        - shuffle : randomly shuffles the returned sequence ordering
        - num_pts : the number of points to use in the returned point cloud. If None uses all points in the data.
        - perturb_pts : the stdev to randomly perturb point clouds with. If None no perturbation is performed.
        - 
        '''
        # settings
        self.batch_size = batch_size
        self.steps_per_seq = num_steps
        self.shuffle = shuffle
        self.perturb_std = perturb_pts
        self.num_pts = num_pts

        # load in data
        for root in roots:
            if not exists(root):
                print('Could not find dataset at ' + root)
                return
        data_loader = ToppleDataLoader()
        self.data = data_loader.load_data(roots)

        if num_pts is None:
            # use all the points in the point cloud
            self.num_pts = self.data.point_cloud.shape[1]

        # load in normalization info
        if not exists(norm_info_file):
            print('Could not find normalization info at ' + norm_info_file)
            return
        self.norm_info = ToppleNormalizationInfo()
        self.norm_info.load_from(norm_info_file)
        print('Loaded normalization info!')

        # see if we have axis-angle info (for backwards compat)
        self.use_aa = False
        self.use_aa_split = False
        self.use_topple_idx = False
        self.use_delta_quat = False
        if len(self.data.delta_rot) > 0:
            self.use_aa = True
        if len(self.data.delta_rot_split) > 0:
            self.use_aa_split = True
        if len(self.data.topple_idx) > 0:
            self.use_topple_idx = True
        if len(self.data.body_friction) > 0:
            self.use_body_friction = True
        if len(self.data.delta_quat) > 0:
            self.use_delta_quat = True

        # normalize the data
        print('Normalizing data...')
        self.normalize_data(self.data, self.norm_info)
        print('Finished normalizing!')

        # order to iterate through data when returning batches (in order by default)
        self.iter_inds = range(0, self.data.size)

        # prepare to iterate through
        self.reset()

    def normalize_data(self, data, norm_info):
        '''
        Normalizes (in place) the given ToppleData using the ToppleNormalizationInfo.
        '''
        # point clouds -> [-1, 1]
        data.point_cloud /= norm_info.pc_max
        # force pos -> [-1, 1]
        data.force_pos /= norm_info.pc_max
        # force vec -> [-1, 1]
        data.force_vec /= norm_info.force_vec_max
        # density -> [0, 1]
        data.density = (data.density - norm_info.density_offset) / norm_info.density_max
        # mass -> [0, 1]
        data.mass = (data.mass - norm_info.mass_offset) / norm_info.mass_max 
        # inertia -> [0, 1]
        data.inertia = (data.inertia - norm_info.inertia_offset) / norm_info.inertia_max
        # friction -> [0, 1]
        if norm_info.friction_offset is not None:
            data.body_friction = (data.body_friction - norm_info.friction_offset) / norm_info.friction_max

        # now time sequence data
        # velocities -> [-1, 1]
        for i, lin_vel_steps in enumerate(data.lin_vel):
            data.lin_vel[i] = [(x / norm_info.max_lin_vel) for x in lin_vel_steps]
        for i, ang_vel_steps in enumerate(data.ang_vel):
            data.ang_vel[i] = [(x / norm_info.max_ang_vel) for x in ang_vel_steps]
        # delta position -> [-1, 1]
        for i, pos_steps in enumerate(data.pos):
            data.pos[i] = [(x / norm_info.max_pos) for x in pos_steps]
        # delta rotation -> [-1, 1]
        for i, rot_steps in enumerate(data.total_rot):
            data.total_rot[i] = [(x / norm_info.max_rot) for x in rot_steps]
        # delta rot axis-angle -> [-1, 1] norm
        if self.use_aa:
            for i, delta_rot_steps in enumerate(data.delta_rot):
                data.delta_rot[i] = [(x / norm_info.max_delta_rot) for x in delta_rot_steps]
        # make axes unit and and normalize angle -> [-1, 1]
        if self.use_aa_split:
            for i, delta_rot_split_steps in enumerate(data.delta_rot_split):
                data.delta_rot_split[i] = [np.append(x[:3] / np.linalg.norm(x[:3]), x[3] / norm_info.max_delta_rot) for x in delta_rot_split_steps]
                
    def reset(self):
        '''
        Prepares to iterate through dataset.
        '''
        if self.shuffle:
            np.random.shuffle(self.iter_inds)
        # we consider an epoch as returning one sequence from every single simulation
        # ( though if the sequence length is shorter than sim length the unique sequences contained
        #  in the dataset will be much more than an epoch length )
        self.num_batches = (self.data.size + self.batch_size - 1) // self.batch_size
        self.batch_idx = 0

    def has_next_batch(self):
        '''
        Returns false if done with the current "epoch" (seen each sim once).
        '''
        return self.batch_idx < self.num_batches

    def next_batch(self, random_window=True, focus_toppling=False):
        '''
        Returns the next batch of data. if random_window=True will get a random sequence of correct length (otherwise
        starts at 0). If focus_toppling=True, will make sure this sequence includes the part of the sequence where toppling occurs.
        '''
        # size is either batch_size, or shorter if we're at the end of the data
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx + 1) * self.batch_size, self.data.size)
        batch_size = end_idx - start_idx

        # get batch data
        batch = ToppleBatch(self.batch_size, self.steps_per_seq, self.num_pts)
        for i in range(batch_size):
            pc, lin_vel, ang_vel, pos, rot, delta_quat, delta_rot, delta_rot_split, topple_label, meta_info = \
                            self.get_seq(self.iter_inds[start_idx + i], self.steps_per_seq, random_window, focus_toppling)
            batch.point_cloud[i] = pc
            batch.lin_vel[i] = lin_vel
            batch.ang_vel[i] = ang_vel
            batch.pos[i] = pos
            batch.rot[i] = rot
            if self.use_delta_quat:
                batch.delta_quat[i] = delta_quat
            if self.use_aa:
                batch.delta_rot[i] = delta_rot
            if self.use_aa_split:
                batch.delta_rot_split[i] = delta_rot_split
            if self.use_topple_idx:
                batch.topple_label[i] = topple_label
            batch.toppled.append(meta_info[0])
            batch.shape_name.append(meta_info[1])
            batch.scale[i] = meta_info[2]
            batch.rot_euler[i] = meta_info[3]
            if self.use_body_friction:
                batch.body_friction[i] = meta_info[4]
            batch.mass[i] = meta_info[5]

        if batch_size != self.batch_size:
            # need to pad the end with repeat of data
            for i in range(self.batch_size - batch_size):
                batch.point_cloud[batch_size + i] = batch.point_cloud[i]
                batch.lin_vel[batch_size + i] = batch.lin_vel[i]
                batch.ang_vel[batch_size + i] = batch.ang_vel[i]
                batch.pos[batch_size + i] = batch.pos[i]
                batch.rot[batch_size + i] = batch.rot[i]
                if self.use_delta_quat:
                    batch.delta_quat[batch_size + i] = batch.delta_quat[i]
                batch.toppled.append(batch.toppled[i])
                batch.shape_name.append(batch.shape_name[i])
                batch.scale[batch_size + i] = batch.scale[i]
                batch.rot_euler[batch_size + i] = batch.rot_euler[i]
                batch.mass[batch_size + i] = batch.mass[i]
                if self.use_aa:
                    batch.delta_rot[batch_size + i] = batch.delta_rot[i]
                if self.use_aa_split:
                    batch.delta_rot_split[batch_size + i] = batch.delta_rot_split[i]
                if self.use_topple_idx:
                    batch.topple_label[batch_size + i] = batch.topple_label[i]
                if self.use_body_friction:
                    batch.body_friction[batch_size + i] = batch.body_friction[i]

        self.batch_idx += 1

        return batch

    def get_seq(self, idx, num_steps, random_window=True, focus_toppling=False):
        '''
        Returns a random contiguous sequence from the simulation at the given idx and length num_steps.
        If num_steps > sim_length the final (sim_length-num_steps) steps are padded with the value at
        sim[sim_length]. 
        '''
        # get the normalized canonical point cloud for this simulation
        pc = np.copy(self.data.point_cloud[self.data.shape_idx[idx]])
        scale = self.data.scale[idx]
        # scale accordingly
        pc *= np.reshape(scale, (1, -1))
        # randomly perturb point cloud
        pc += np.random.normal(0.0, self.perturb_std, pc.shape)

        # randomly draw a subset of points if desired
        if self.num_pts < pc.shape[0]:
            pc_inds = np.random.choice(pc.shape[0], self.num_pts, replace=False)
            pc = pc[pc_inds, :]

        # randomly choose a size num_steps sequence from the simulation to return time-series data
        total_steps = len(self.data.lin_vel[idx])
        max_start_step = total_steps - num_steps
        start_step = 0
        if max_start_step < 0:
            # simulation is shorter than desired sequence length
            pad_len = abs(max_start_step)
            lin_vel_list = self.data.lin_vel[idx]
            lin_vel_out = np.array(lin_vel_list + [lin_vel_list[-1]]*pad_len)
            ang_vel_list = self.data.ang_vel[idx]
            ang_vel_out = np.array(ang_vel_list + [ang_vel_list[-1]]*pad_len)
            pos_list = self.data.pos[idx]
            pos_out = np.array(pos_list + [pos_list[-1]]*pad_len)
            rot_list = self.data.total_rot[idx]
            rot_out = np.array(rot_list + [rot_list[-1]]*pad_len)
            if self.use_delta_quat:
                delta_quat_list = self.data.delta_quat[idx]
                delta_quat_out = np.array(delta_quat_list + [delta_quat_list[-1]]*pad_len)
            euler_rot_list = self.data.rot_euler[idx]
            euler_rot_out = np.array(euler_rot_list + [euler_rot_list[-1]]*pad_len)
            if self.use_aa:
                delta_rot_list = self.data.delta_rot[idx]
                delta_rot_out = np.array(delta_rot_list + [delta_rot_list[-1]]*pad_len)
            if self.use_aa_split:
                delta_rot_split_list = self.data.delta_rot_split[idx]
                delta_rot_split_out = np.array(delta_rot_split_list + [delta_rot_split_list[-1]]*pad_len)
            if self.use_topple_idx:
                topple_label_out = np.zeros((total_steps + pad_len), dtype=int)
                seq_topple_idx = self.data.topple_idx[idx]
                if seq_topple_idx > 0:
                    topple_label_out[seq_topple_idx:] = 1
        else:
            start_step = 0
            if random_window:
                if focus_toppling and self.data.toppled[idx]:
                    # choose window around the index where it topples
                    topple_idx = self.data.topple_idx[idx]
                    min_idx = max([topple_idx - num_steps + 1, 0])
                    if min_idx >= max_start_step:
                        # just pick the max index
                        start_step = max_start_step
                    else:
                        # our window is guaranteed to see some part of toppling
                        start_step = np.random.randint(min_idx, max_start_step+1)
                else:
                    start_step = np.random.randint(0, max_start_step+1)
            end_step = start_step + num_steps
            # print('Range: %d, %d' % (start_step, end_step))
            lin_vel_out = np.array(self.data.lin_vel[idx][start_step:end_step])
            ang_vel_out = np.array(self.data.ang_vel[idx][start_step:end_step])
            pos_out = np.array(self.data.pos[idx][start_step:end_step])
            rot_out = np.array(self.data.total_rot[idx][start_step:end_step])
            if self.use_delta_quat:
                delta_quat_out = np.array(self.data.delta_quat[idx][start_step:end_step])
            euler_rot_out = np.array(self.data.rot_euler[idx][start_step:end_step])
            if self.use_aa:
                delta_rot_out = np.array(self.data.delta_rot[idx][start_step:end_step])
            if self.use_aa_split:
                delta_rot_split_out = np.array(self.data.delta_rot_split[idx][start_step:end_step])
            if self.use_topple_idx:
                topple_label_out = np.zeros((num_steps), dtype=int)
                seq_topple_idx = self.data.topple_idx[idx]
                if seq_topple_idx > 0:
                    if seq_topple_idx <= start_step:
                        topple_label_out[:] = 1
                    elif seq_topple_idx < end_step:
                        topple_label_out[seq_topple_idx-start_step:] = 1

        # rotate point cloud to align with first frame of sequence
        init_rot = self.data.rot_euler[idx][start_step]
        xrot, yrot, zrot = np.radians(init_rot)
        R = transforms3d.euler.euler2mat(zrot, xrot, yrot, axes='szxy') # unity applies euler angles in z, x, y ordering
        pc = np.dot(pc, R.T)

        toppled = self.data.toppled[idx]
        shape_name = self.data.shape_name[idx]
        mass = self.data.mass[idx]
        body_fric = -1.0
        if self.use_body_friction:
            body_fric = self.data.body_friction[idx]

        meta_info = (toppled, shape_name, scale, euler_rot_out, body_fric, mass)

        if not self.use_aa:
            delta_rot_out = None
        if not self.use_aa_split:
            delta_rot_split_out = None
        if not self.use_topple_idx:
            topple_label_out = None
        if not self.use_delta_quat:
            delta_quat_out = None
        
        return pc, lin_vel_out, ang_vel_out, pos_out, rot_out, delta_quat_out, delta_rot_out, delta_rot_split_out, topple_label_out, meta_info
        

    def get_norm_info(self):
        return self.norm_info


if __name__=='__main__':
    # norm_info = ToppleNormalizationInfo()
    # norm_info.load_from('../../data/sim/normalization_info/cube_train.pkl')
    # norm_info.print_out()

    topple_data = ToppleDataset(roots=['./data/sim/Cube/Cube30k_ObjSplit/Cube30kVal'], norm_info_file='./data/sim/normalization_info/cube_30k.pkl', \
                                batch_size=5, num_steps=10, shuffle=True, num_pts=None, perturb_pts=0.01)

    count = 0
    while topple_data.has_next_batch():
        batch = topple_data.next_batch(random_window=True, focus_toppling=False)
        count += 1
        # print(batch.lin_vel[0])
        # print(batch.toppled[0])
        # print(batch.delta_rot_split[0])
        # print(batch.delta_rot[0])
        # print(batch.topple_label[0])
        # print(batch.pos)
        # print(batch.body_friction)
        # print(batch.delta_quat[0])
        # print(np.degrees(2*np.arccos(batch.delta_quat[0, :, 0])))

    print('Total num batches: ' + str(count))

    topple_data.reset()
    count = 0
    while topple_data.has_next_batch():
        batch = topple_data.next_batch()
        count += 1
        print(batch.size)

    print('Total num batches: ' + str(count))
