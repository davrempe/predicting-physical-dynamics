import argparse
from datetime import datetime
import time
import random
import json
import numpy as np
import importlib
from os.path import join
import transforms3d
import csv

import os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cwd = os.getcwd()
sys.path.append(join(cwd, 'scripts'))
sys.path.append(join(cwd, 'scripts/data'))

from utils import tf_util
from data.topple_dataset import ToppleDataset
import data.data_list_loader

class ToppleTest(object):
    
    def __init__(self, args):
        flags = self.parse_args(args)
        self.setup(flags)

    def parse_args(self, args):
        parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
        parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
        parser.add_argument('--model', default='topple_aa_rnn_classify', help='Model name')
        parser.add_argument('--saved_model', default='log/dev_train_aa/topple_aa_rnn_best_model.ckpt', help='model checkpoint file path')
        parser.add_argument('--data_list', default='./data/sim/dataset_lists/Cube5k', help='Root of the dataset lists path for the dataset to train and validate on.')
        parser.add_argument('--norm_info', default='./data/sim/normalization_info/cube_5k.pkl', help='Validation root')
        parser.add_argument('--log', default='', help='Log directory [default: log/timestamp]')
        parser.add_argument('--num_pts', type=int, default=1024, help='Number to use in Point Cloud')
        parser.add_argument('--seq_len', type=int, default=15, help='Length of sequences to test')
        parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training')
        parser.add_argument('--warm_start', type=int, default=0, help='Number of frames to warm start rollout with')

        # the tests to run
        # single-step error (given gt input, error of single step prediction using hidden state propagated through whole sequence)
        parser.add_argument('--test_single_step', dest='test_single_step', action='store_true')
        parser.set_defaults(test_single_step=False)
        # roll-out error (given initial input, roll out sequence just from predictions, error from gt at every step)
        parser.add_argument('--test_roll_out', dest='test_roll_out', action='store_true')
        parser.set_defaults(test_roll_out=False)
        parser.add_argument('--output_pred', dest='output_pred', action='store_true') # save model predictions for each sequence
        parser.set_defaults(output_pred=False)
        # topple classification error
        parser.add_argument('--test_topple_classify', dest='test_topple_classify', action='store_true')
        parser.set_defaults(test_topple_classify=False)
        parser.add_argument('--topple_thresh', type=float, default=45.0, help='Degrees rotation in x or z to be considered as toppling')

        # network properties
        parser.add_argument('--num_units', type=int, default=128, help='Number of units to use in each RNN unit [default: 128]')
        parser.add_argument('--cell_type', default='lstm', help='Cell type to use (rnn, gru, or lstm) [default: lstm]')
        parser.add_argument('--num_cells', type=int, default=3, help='Number of cells to stack to form the RNN module [default: 3]')

        flags = parser.parse_args(args)

        # print(flags)

        return flags

    # all file paths assume this is script being run from the top level directory

    def setup(self, flags):
        global tf, tf_util # have to do these in specific ordering

        # must set the self.gpu we want to use before importing tf
        self.gpu = flags.gpu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

        import tensorflow as tf
        from utils import tf_util
        
        # get the toppling model
        self.model = importlib.import_module('models.' + flags.model)
        self.model_restore_path = flags.saved_model
        
        # testing settings
        self.batch_size = flags.batch_size
        dataset_list = flags.data_list
        self.norm_path = flags.norm_info
        self.num_pts = flags.num_pts
        self.seq_len = flags.seq_len
        self.test_single_step = flags.test_single_step
        self.test_roll_out = flags.test_roll_out
        self.output_pred = flags.output_pred
        self.test_topple_classify = flags.test_topple_classify
        self.topple_thresh = flags.topple_thresh
        self.warm_start_idx = flags.warm_start

        # architecture settings
        self.num_units = flags.num_units
        self.cell_type = flags.cell_type
        self.num_cells = flags.num_cells

        # set up logging directories
        default_log = (flags.log == '')
        self.log_dir = flags.log
        if default_log:
            self.log_dir = 'log'
        if not os.path.exists(self.log_dir): 
            os.mkdir(self.log_dir)
        if default_log:
            self.log_dir = join(self.log_dir, 'log_' + str(int(time.time())))
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)
        # backup scripts
        log_scripts = join(self.log_dir, 'scripts') + '/'
        if not os.path.exists(log_scripts):
            os.mkdir(log_scripts)
        model_file = join('./scripts/models', flags.model) + '.py'
        os.system('cp %s %s' % (model_file, log_scripts)) # bkp of model def
        os.system('cp %s %s' % (os.path.realpath(__file__), log_scripts)) # bkp of test procedure
        # plots directory
        self.log_plots = join(self.log_dir, 'plots')
        if not os.path.exists(self.log_plots):
            os.mkdir(self.log_plots)
        # predicted sequences directory
        self.pred_out_path = os.path.join(self.log_dir, 'pred_out')
        if self.output_pred and not os.path.exists(self.pred_out_path):
            os.mkdir(self.pred_out_path)

        self.log_fout = open(os.path.join(self.log_dir, 'log_test.txt'), 'w')
        # log flags used
        self.log_fout.write(str(flags)+'\n')

        _, _, test_paths = data.data_list_loader.load_dataset(dataset_list)

        # load datasets
        print('===================================================================')
        print('TESTING DATA: \n===================================================================')
        self.test_data = ToppleDataset(roots=test_paths, norm_info_file=self.norm_path, batch_size=self.batch_size, \
                                num_steps=self.seq_len, shuffle=False, num_pts=self.num_pts, perturb_pts=0.0)

    def log_string(self, out_str):
            self.log_fout.write(out_str+'\n')
            self.log_fout.flush()
            print(out_str)

    #
    # testing functions
    #

    def test(self):
        self.log_string('pid: %s'%(str(os.getpid())))
        # build model graph
        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(self.gpu)): 
                # get placeholders for input
                pcl_pl, pcl_feat_pl, lin_vel_pl, ang_vel_pl, pos_pl, delta_rot_pl, topple_label_pl = \
                                    self.model.placeholder_inputs(self.batch_size, self.num_pts, None) # seq_len different for single_step vs rollout eval
                is_training_pl = tf.placeholder(tf.bool, shape=())
                steps_in_pl = tf.placeholder(tf.int32, shape=())

                # geometry feature used at every step
                pcl_feat_out = self.model.get_geom_model(pcl_pl, is_training_pl)

                # make prediction based on pt feature input
                # only need hidden_state_pl if we're doing single step i.e. taking entire sequence in at once
                lin_vel_in = tf.cond(tf.equal(steps_in_pl, tf.constant(1)), lambda: lin_vel_pl[:,:,:], lambda: lin_vel_pl[:,:(steps_in_pl - 1),:])
                ang_vel_in = tf.cond(tf.equal(steps_in_pl, tf.constant(1)), lambda: ang_vel_pl[:,:,:], lambda: ang_vel_pl[:,:(steps_in_pl - 1),:])
                use_steps = tf.cond(tf.equal(steps_in_pl, tf.constant(1)), lambda:tf.constant(1), lambda:(steps_in_pl - 1))
                pred_dynamics_out, hidden_state_out, init_state = self.model.get_dynamics_model(pcl_feat_pl, lin_vel_in, ang_vel_in, \
                                                                    self.cell_type, self.num_cells, self.num_units, 1.0, use_steps, \
                                                                    is_training_pl)

                # loss of state predictions for training
                loss, error_tup = self.model.get_loss(pred_dynamics_out, lin_vel_pl, ang_vel_pl, pos_pl, delta_rot_pl, topple_label_pl, steps_in_pl, is_training=is_training_pl)
                # unnormalize errors
                lin_vel_err = tf.reshape(self.test_data.norm_info.max_lin_vel * error_tup[0], [1, 3])
                ang_vel_err = tf.reshape(self.test_data.norm_info.max_ang_vel * error_tup[1], [1, 3])
                pos_err = tf.reshape(self.test_data.norm_info.max_pos * error_tup[2], [1, 3])
                angle_err = tf.reshape(tf.tile(self.test_data.norm_info.max_delta_rot * tf.expand_dims(error_tup[3], axis=0), [3]), [1, 3]) # repeat for ease of data transfer
                axis_err = tf.reshape(tf.tile(tf.expand_dims(error_tup[4], axis=0), [3]), [1, 3]) # cos sim doesn't need unnorm
                classify_err = tf.reshape(tf.tile(tf.expand_dims(error_tup[5], axis=0), [3]), [1, 3])
                errors = tf.concat([lin_vel_err, ang_vel_err, pos_err, angle_err, axis_err, classify_err], axis=0)

                # Add ops to save and restore all the variables.
                saver = tf.train.Saver()

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)

            # inputs/outputs
            ops = {'pcl_pl': pcl_pl,
                'pcl_feat_pl' : pcl_feat_pl,
                'lin_vel_pl' : lin_vel_pl,
                'ang_vel_pl' : ang_vel_pl,
                'pos_pl' : pos_pl,
                'delta_rot_pl' : delta_rot_pl,
                'topple_label_pl' : topple_label_pl,
                'is_training_pl': is_training_pl,
                'init_state' : init_state,
                'pcl_feat_out' : pcl_feat_out,
                'hidden_state_out' : hidden_state_out,
                'pred_dynamics_out' : pred_dynamics_out,
                'steps_in_pl' : steps_in_pl,
                'loss': loss,
                'errors' : errors}

            # restore the model
            saver.restore(sess, self.model_restore_path)
            self.log_string("Test model restored.")

            # test
            accuracy_tuple = self.one_epoch(sess, ops, self.test_data)

        self.log_fout.close()

        return accuracy_tuple

    def one_epoch(self, sess, ops, dataset):
        '''
        Execute the model graph on all data in the given dataset. Returns loss and error averaged over the epoch.
        '''
        self.log_string(str(datetime.now()))

        num_batches = 0
        num_seq = 0
        num_topple_seq = 0
        num_no_topple_seq = 0
        loss_sum = 0.
        errors_sum = np.zeros((6, 3), dtype=np.float)
        errors_norm_sum = np.zeros((6), dtype=np.float)
        final_roll_err_sum = np.zeros((5), dtype=np.float) # error at final step of sequence
        final_topple_err_sum = np.zeros((5), dtype=np.float)
        final_no_topple_err_sum = np.zeros((5), dtype=np.float)
        all_step_err_sum = np.zeros((self.seq_len, 5), dtype=np.float) # error averaged over batch dimension
        topple_step_err_sum = np.zeros((self.seq_len, 5), dtype=np.float)
        no_topple_step_err_sum = np.zeros((self.seq_len, 5), dtype=np.float)
        topple_roll_err_sum = np.zeros((5), dtype=np.float) # over averaged over all timesteps for toppling examples
        no_topple_roll_err_sum = np.zeros((5), dtype=np.float)
        rollout_err_list = []
        confusion_matrix_sum = np.zeros((4), dtype=np.float) # TP, FP, FN, TN
        while dataset.has_next_batch():
            # get the next batch of data
            cur_batch = dataset.next_batch(random_window=False)
            num_batches += 1
            num_seq += cur_batch.size    

            # run graph once to get geometry feature (needed for all tests) and initial LSTM state
            feed_dict = {ops['pcl_pl']: cur_batch.point_cloud,
                        ops['is_training_pl']: False }
            geom_feat, init_state = sess.run([ops['pcl_feat_out'], ops['init_state']], feed_dict=feed_dict)

            # error with gt input compared to ground truth at every step
            if self.test_single_step:
                loss, errors = self.single_step_seq(sess, ops, cur_batch, geom_feat)
                loss_sum += loss
                errors_sum += errors
                errors_norm_sum += np.linalg.norm(errors, axis=1)
                if num_batches % 100 == 0:
                    self.log_single_step_stats(num_batches+1, loss_sum / num_batches, errors_sum / num_batches, errors_norm_sum / num_batches)

            # error w/ only init input, rolled out over time
            obj_state_seq = None
            up_vec_seq = None
            if self.test_roll_out or self.test_topple_classify:
                rollout_errors, obj_state_seq, axis_angle_seq, up_vec_seq, topple_classify_seq = \
                            self.roll_out_seq(sess, ops, cur_batch, geom_feat, init_state, self.output_pred)
                rollout_err_list.append(rollout_errors)
                # errors at last step (B, 4)
                final_rollout_error = rollout_errors[:, -1, :] 
                if self.test_roll_out:
                    topple_arr = np.array(cur_batch.toppled)
                    topple_inds = np.nonzero(topple_arr)[0] # toppling indices
                    no_topple_inds = np.where(topple_arr == False)[0]

                    num_topple_seq += topple_inds.shape[0]
                    num_no_topple_seq += no_topple_inds.shape[0]

                    final_roll_err_sum += np.sum(final_rollout_error, axis=0)
                    # mean over batch for each timestep
                    if topple_inds.shape[0] > 0:
                        final_topple_err_sum += np.sum(final_rollout_error[topple_inds], axis=0)
                        topple_rollout_errors = rollout_errors[topple_inds]
                        topple_step_err_sum += np.sum(topple_rollout_errors, axis=0)
                        topple_roll_err_sum += np.sum(np.mean(topple_rollout_errors, axis=1), axis=0) # mean over all timesteps first, then sum over batch
                    if no_topple_inds.shape[0] > 0:
                        final_no_topple_err_sum += np.sum(final_rollout_error[no_topple_inds], axis=0)
                        no_topple_rollout_errors = rollout_errors[no_topple_inds]
                        no_topple_step_err_sum += np.sum(no_topple_rollout_errors, axis=0)
                        no_topple_roll_err_sum += np.sum(np.mean(no_topple_rollout_errors, axis=1), axis=0) # mean over all timesteps, then sum batch

                    all_step_err_sum += np.sum(rollout_errors, axis=0)

                    if num_batches % 10 == 0:
                        topple_divide = (num_topple_seq if num_topple_seq > 0 else 1)
                        no_topple_divide = (num_no_topple_seq if num_no_topple_seq > 0 else 1)
                        self.log_roll_out_stats(num_batches+1, final_roll_err_sum / num_seq, final_topple_err_sum / topple_divide, final_no_topple_err_sum / no_topple_divide, \
                                topple_roll_err_sum / topple_divide, no_topple_roll_err_sum / no_topple_divide)
                
                # save sequence of predictions fo viz later
                if self.output_pred:
                    self.save_sequence(obj_state_seq, axis_angle_seq, cur_batch, (num_batches-1)*cur_batch.size, final_rollout_error)

            # whether able to classify that it has toppled from state prediction
            if self.test_topple_classify:
                if obj_state_seq != None and up_vec_seq != None:
                    classify_results = self.classify_topple_seq(cur_batch, obj_state_seq, up_vec_seq, topple_classify_seq)
                    confusion_matrix_sum += np.array(classify_results)
                else:
                    print('Could not record predicted state sequence for some reason!! This should never happen')

        # find means
        loss_sum /= num_batches
        errors_sum /= num_batches
        errors_norm_sum /= num_batches
        final_roll_err_sum /= num_seq
        final_topple_err_sum /= num_topple_seq
        final_no_topple_err_sum /= num_no_topple_seq
        all_step_err_sum /= num_seq
        topple_step_err_sum /= num_topple_seq
        no_topple_step_err_sum /= num_no_topple_seq
        topple_roll_err_sum /= num_topple_seq
        no_topple_roll_err_sum /= num_no_topple_seq
        if self.test_topple_classify:
            total_classify_accuracy = float(confusion_matrix_sum[0] + confusion_matrix_sum[3]) / num_seq
            precision = confusion_matrix_sum[0] / (confusion_matrix_sum[0] + confusion_matrix_sum[1])
            confusion_matrix_sum /= np.array([num_topple_seq, num_no_topple_seq, num_topple_seq, num_no_topple_seq], dtype=np.float)
            recall = confusion_matrix_sum[0]
            f_score = 2 * ((precision*recall) / (precision + recall))
            topple_classify_accuracy = confusion_matrix_sum[0]
        # final log
        if self.test_single_step:
            print('================== SINGLE STEP RESULTS =====================')
            self.log_single_step_stats(num_batches+1, loss_sum, errors_sum, errors_norm_sum)
        if self.test_roll_out:
            print('================== ROLL OUT RESULTS =====================')
            self.log_roll_out_stats(num_batches+1, final_roll_err_sum, final_topple_err_sum, final_no_topple_err_sum, \
                        topple_roll_err_sum, no_topple_roll_err_sum)
            # plot mean error over all batches
            # self.plot_rollout_results(all_step_err_sum, topple_step_err_sum, no_topple_step_err_sum)
        if self.test_topple_classify:
            print('================== TOPPLE CLASSIFICATION RESULTS =================')
            self.log_classify_stats(total_classify_accuracy, topple_classify_accuracy, confusion_matrix_sum, f_score)

        # save data for further analysis later
        if self.test_roll_out:
            all_rollout_errors = np.concatenate(rollout_err_list, axis=0)
            save_path = os.path.join(self.log_dir, 'eval_data')
            if not os.path.exists(save_path): os.mkdir(save_path)
            # entire split errors
            np.savez_compressed(os.path.join(save_path, 'rollout_err_data'), rollout_errors=all_rollout_errors, \
                                all_step_err_sum=all_step_err_sum, topple_step_err_sum=topple_step_err_sum, no_topple_step_err_sum=no_topple_step_err_sum)

        # output data we need for evaluation table in a nice csv
        if self.test_single_step and self.test_roll_out and self.test_topple_classify:
            with open(os.path.join(self.log_dir, 'eval_results.csv'), 'w') as csvfile:
                writer = csv.writer(csvfile, dialect='excel')
                writer.writerow(['Single Step Error'] + ['']*4 + ['Non-topple Rollout Error'] + ['']*4 + ['Toppling Classification'] + ['']*4)
                writer.writerow(['Lin Vel (m/s)', 'Ang Vel (rad/s)', 'Pos (m)', 'Angle (deg)', 'Axis']*2 + ['TP', 'FP', 'FN', 'TN', 'Accuracy'])
                errors_out =  errors_norm_sum.tolist()[:5]
                errors_out[3] = errors_sum[3, 0] # don't want norm of angle
                errors_out[4] = errors_sum[4, 0] # or axis error
                outrow = errors_out + no_topple_roll_err_sum.tolist() + confusion_matrix_sum.tolist() + [total_classify_accuracy]
                writer.writerow(outrow)

        dataset.reset()

        if self.test_topple_classify:
            # mean error for no toppling examples for pos and rot
            return total_classify_accuracy, topple_classify_accuracy, no_topple_roll_err_sum[2], no_topple_roll_err_sum[3]

    def single_step_seq(self, sess, ops, cur_batch, geom_feat):
        # given data for a single sequence
        # run graph again with geom feat as input, entire gt sequence input
        # can use loss and errors from graph execution since same as training
        feed_dict = {ops['pcl_feat_pl']: geom_feat,
                    ops['lin_vel_pl'] : cur_batch.lin_vel,
                    ops['ang_vel_pl'] : cur_batch.ang_vel,
                    ops['pos_pl'] : cur_batch.pos,
                    ops['delta_rot_pl'] : cur_batch.delta_rot,
                    ops['topple_label_pl'] : cur_batch.topple_label,
                    ops['steps_in_pl'] : self.seq_len,
                    ops['is_training_pl']: False}

        loss, errors, pred = sess.run([ops['loss'], ops['errors'], ops['pred_dynamics_out']], feed_dict=feed_dict)
        # axis errors may be nan if batch size is one and all angles are < 1 degree
        errors[4, np.nonzero(np.isnan(errors[4, :]))[0]] = 0.0

        return loss, errors

    def roll_out_seq(self, sess, ops, cur_batch, geom_feat, init_state, save_pred):
        # structure to hold rcurrent object state
        # (batch_size, 1, 12) where axis=2 is [v, w, p, r]
        # start with initial gt state
        obj_state = np.expand_dims(np.concatenate([cur_batch.lin_vel[:,0,:], cur_batch.ang_vel[:,0,:], \
                                    cur_batch.pos[:,0,:], cur_batch.rot_euler[:,0,:]], axis=1), axis=1)
        # array to unnormalize with during roll-out
        unnorm_arr = 3*[self.test_data.norm_info.max_lin_vel] + 3*[self.test_data.norm_info.max_ang_vel] + \
                        3*[self.test_data.norm_info.max_pos] + 3*[1.0] # will do rotation unnormalization separately
        unnorm_arr = np.array([unnorm_arr])
        # sequence of object state (will always only be 1 step so we just save [B, 12])
        obj_state_seq = [np.copy(obj_state[:,0,:])*unnorm_arr]
        angle_pred_seq = np.zeros((cur_batch.size, cur_batch.num_steps))
        # print(unnorm_arr)

        # sequence of toppling classificaitons
        topple_classify_seq = np.zeros((cur_batch.size, cur_batch.num_steps))

        # also want to track how our sequence transforms the up vector to classify toppling
        up_vec_state = np.concatenate([np.zeros((cur_batch.size, 1)), \
                                    np.ones((cur_batch.size, 1)), \
                                    np.zeros((cur_batch.size, 1))], axis=1)
        up_vec_state_seq = [np.copy(up_vec_state)]

        axis_angle_seq = np.zeros((cur_batch.size, cur_batch.num_steps, 3))
        axis_errors = np.zeros((cur_batch.size, cur_batch.num_steps))

        hidden_state = init_state
        for i in range(cur_batch.num_steps - 1):
            feed_dict = {ops['pcl_feat_pl']: geom_feat,
                        ops['lin_vel_pl'] : obj_state[:, :, 0:3],
                        ops['ang_vel_pl'] : obj_state[:, :, 3:6],
                        ops['pos_pl'] : obj_state[:, :, 6:9],
                        ops['init_state'] : hidden_state,
                        ops['steps_in_pl'] : 1, # only uses first step in graph
                        ops['is_training_pl']: False }

            hidden_state, test_init_state, pred = sess.run([ops['hidden_state_out'], ops['init_state'], ops['pred_dynamics_out']], feed_dict=feed_dict)

            # get delta rotaion (axis-angle) predictions and convert to euler so can update state and output rot sequence
            axis_pred = pred[:, 0, 9:12] * self.test_data.norm_info.max_delta_rot
            axis_angle_seq[:, i+1, :] = axis_pred
            angle_pred = np.linalg.norm(axis_pred, axis=1)
            angle_pred_seq[:, i+1] = angle_pred
            axis_pred /= np.reshape(angle_pred, (-1, 1))
            angle_pred = np.radians(angle_pred)
            for j in range(axis_pred.shape[0]):
                # must be in zxy ordering to visualize in unity
                pred_euler_zxy = np.array(transforms3d.euler.axangle2euler(axis_pred[j].tolist(), angle_pred[j], axes='szxy'))
                pred[j,0,9:12] = np.degrees(pred_euler_zxy[[1, 2, 0]])
                # update up-vec state
                R = transforms3d.axangles.axangle2mat(axis_pred[j].tolist(), angle_pred[j])
                up_vec_state[j,:] = np.dot(R, up_vec_state[j,:])

            axis_gt = cur_batch.delta_rot[:, i+1, :] * self.test_data.norm_info.max_delta_rot
            dot_prod = np.sum(axis_angle_seq[:,i+1,:] * axis_gt, axis=1)
            denom = (np.degrees(angle_pred) * np.linalg.norm(axis_gt, axis=1)) + 1e-6 # for stability
            axis_errors[:, i] = np.ones((cur_batch.size)) - (dot_prod / denom)

            topple_classify_seq[:, i+1] = 1.0 / (1.0 + np.exp(-pred[:, 0, 12])) # we get raw logits from output

            if i < self.warm_start_idx:
                # warm start
                obj_state = np.expand_dims(np.concatenate([cur_batch.lin_vel[:,i+1,:], cur_batch.ang_vel[:,i+1,:], \
                                    cur_batch.pos[:,i+1,:], cur_batch.rot_euler[:,i+1,:]], axis=1), axis=1) 
            else:
                # update and save unnormalized state
                obj_state += pred[:,:,:12]

            # update and save unnormalized state
            obj_state_seq.append(np.copy(obj_state[:,0,:])*unnorm_arr)
            # save up_vec_state
            up_vec_state_seq.append(np.copy(up_vec_state))

        # axis error blows up when angle is very small (since the direction doesn't matter at this point)
        cur_batch_angles = np.linalg.norm(cur_batch.delta_rot[:,:,:] * self.test_data.norm_info.max_delta_rot, axis=2)
        axis_errors[np.nonzero(cur_batch_angles < 1.0)] = 0.0

        # calculate errors
        # difference for all batches and steps for v, w, p
        errors = np.zeros((cur_batch.size, cur_batch.num_steps, 3, 3))
        angle_errors = np.zeros((cur_batch.size, cur_batch.num_steps))
        for i in range(len(obj_state_seq)):
            # do axis angle errors separately
            if i > 0: # first one doesn't matter since given
                cur_batch_angles = np.linalg.norm(cur_batch.delta_rot[:,i,:] * self.test_data.norm_info.max_delta_rot, axis=1)
                angle_errors[:, i] = np.abs(cur_batch_angles - angle_pred_seq[:, i])
                # now everything else
                gt_diff = np.stack([cur_batch.lin_vel[:,i,:] - cur_batch.lin_vel[:,i-1,:], cur_batch.ang_vel[:,i,:] - cur_batch.ang_vel[:,i-1,:], \
                                        cur_batch.pos[:,i,:] - cur_batch.pos[:,i-1,:]], axis=1) * np.reshape(unnorm_arr[:, 0:9], (1, 3, 3))
                pred = np.reshape(obj_state_seq[i][:,:9], (cur_batch.size, 3, 3))
                pred_m1 = np.reshape(obj_state_seq[i-1][:,:9], (cur_batch.size, 3, 3))
                pred_diff = pred - pred_m1
                errors[:,i,:,:] = pred_diff - gt_diff
        # now take norm over each state vector which leaves (B, T, 3), norm of each value at every time step
        roll_out_errors = np.linalg.norm(errors, axis=3)
        # stack on angle errors for (B, T, 5)
        roll_out_errors = np.concatenate([roll_out_errors, np.expand_dims(angle_errors, axis=2), np.expand_dims(axis_errors, axis=2)], axis=2)

        return roll_out_errors, obj_state_seq, axis_angle_seq, up_vec_state_seq, topple_classify_seq

    def save_sequence(self, obj_state_seq, axis_angle_seq, batch_data, start_sim_idx, final_errors):
        '''
        Takes in a sequence of predicted object states of shape (B, 12) where axis 1 is [v, w, p, r] vectors along with
        batch data to save for visualization later.
        Must also input the index of the first sim in the batch - the predictions will be saved with this label.
        '''
        if len(obj_state_seq) == 0:
            return
        
        # save every sim in the batch
        batch_size = obj_state_seq[0].shape[0]
        num_steps = len(obj_state_seq)
        for i in range(batch_size):
            file_out = os.path.join(self.pred_out_path, 'eval_sim_' + str(start_sim_idx + i) + '.json')
            with open(file_out, 'w') as f:
                # put info into dictionaries or strings for json dump
                shape_out = batch_data.shape_name[i]
                toppled_out = batch_data.toppled[i]
                cur_scale = batch_data.scale[i]
                scale_out = {'x' : cur_scale[0], 'y' : cur_scale[1], 'z' : cur_scale[2]}
                # ground truth pos
                cur_gt_pos = batch_data.pos[i]
                cur_gt_pos *= self.test_data.norm_info.max_pos # unnormalize
                gt_pos_out = self.create_json_vec_list(cur_gt_pos)
                # ground truth cummulative rotation
                cur_gt_total_rot = batch_data.rot[i]
                cur_gt_total_rot *= self.test_data.norm_info.max_rot
                gt_total_rot_out = self.create_json_vec_list(cur_gt_total_rot)
                # ground truth euler rot (actual pose from simulation)
                cur_gt_euler_rot = batch_data.rot_euler[i] # don't need to unnormalize
                gt_euler_rot_out = self.create_json_vec_list(cur_gt_euler_rot)
                # ground truth angle-axis delta rot
                cur_gt_delta_rot = batch_data.delta_rot[i]
                cur_gt_delta_rot *= self.test_data.norm_info.max_delta_rot
                gt_delta_rot_out = self.create_json_vec_list(cur_gt_delta_rot)
                # predicted angle-axis delta rot
                cur_pred_delta_rot = axis_angle_seq[i, :, :]
                pred_delta_rot_out = self.create_json_vec_list(cur_pred_delta_rot)
                # predicted pos (already unnormalized)
                cur_pred_pos = np.array([obj_state_seq[j][i, 6:9] for j in range(num_steps)])
                pred_pos_out = self.create_json_vec_list(cur_pred_pos)
                # predicted rot (already unnormalized)
                cur_pred_rot = np.array([obj_state_seq[j][i, 9:12] for j in range(num_steps)])
                pred_rot_out = self.create_json_vec_list(cur_pred_rot)
                # final step errors (norm across all x, y, z)
                pos_err_out = float(final_errors[i, 2])
                rot_err_out = float(final_errors[i, 3])

                # create dict and output
                json_dict = {'shape' : shape_out, \
                                'scale' : scale_out, \
                                'gt_pos' : gt_pos_out, \
                                'gt_total_rot' : gt_total_rot_out, \
                                'gt_euler_rot' : gt_euler_rot_out, \
                                'gt_delta_rot' : gt_delta_rot_out, \
                                'pred_pos' : pred_pos_out, \
                                'pred_rot' : pred_rot_out, \
                                'pred_delta_rot' : pred_delta_rot_out, \
                                'pos_err' : pos_err_out, \
                                'rot_err' : rot_err_out, \
                                'toppled' : toppled_out}
                json_string = json.dumps(json_dict, sort_keys=True, separators=(',', ':'))
                # print(json_string)
                f.write(json_string)


    def create_json_vec(self, vec):
            ''' creates a json dict vec from a np array of size 3 or 4'''
            json_dict = {}
            if len(vec) == 3:
                json_dict = {'x' : float(vec[0]), 'y' : float(vec[1]), 'z' : float(vec[2])}
            elif len(vec) == 4:
                json_dict = {'x' : float(vec[0]), 'y' : float(vec[1]), 'z' : float(vec[2]), 'w' : float(vec[3])}
            return json_dict

    def create_json_vec_list(self, vec_list):
        return [self.create_json_vec(vec) for vec in vec_list]


    def classify_topple_seq(self, cur_batch, obj_state_seq, up_vec_seq, classification_seq):
        # classify based on if there is a 1 in the predicted classification sequence
        topple_classification = (np.sum(np.round(classification_seq), axis=1) > 0)

        # compare to gt
        gt_topple = np.array(cur_batch.toppled)
        topple_inds = np.nonzero(gt_topple)[0]
        if topple_inds.shape[0] > 0:
            pred_topple = topple_classification[topple_inds]
            true_pos = len(np.nonzero(pred_topple)[0])
            false_neg = pred_topple.shape[0] - true_pos
        else:
            true_pos = 0
            false_neg = 0

        non_topple_inds = np.where(gt_topple == False)[0]
        if non_topple_inds.shape[0] > 0:
            pred_non_topple = topple_classification[non_topple_inds]
            true_neg = len(np.where(pred_non_topple == False)[0])
            false_pos = len(np.nonzero(pred_non_topple)[0])
        else:
            true_neg = 0
            false_pos = 0

        return true_pos, false_pos, false_neg, true_neg

    def log_single_step_stats(self, batch_num, loss, errors, error_norms):
        self.log_string(' ---- SINGLE STEP after batch: %03d ----' % (batch_num))
        self.log_string('mean loss: %f' % (loss))
        self.log_string('mean lin vel err: (%f, %f, %f) m/s, norm = %f' % (errors[0, 0], errors[0, 1], errors[0, 2], error_norms[0]))
        self.log_string('mean ang vel err: (%f, %f, %f) rad/s, norm = %f' % (errors[1, 0], errors[1, 1], errors[1, 2], error_norms[1]))
        self.log_string('mean pos err: (%f, %f, %f) m, norm = %f' % (errors[2, 0], errors[2, 1], errors[2, 2], error_norms[2]))
        self.log_string('mean angle err: %f deg' % (errors[3, 0]))
        self.log_string('mean axis err: %f' % (errors[4, 0]))
        self.log_string('mean classification loss: %f' % (errors[5, 0]))

    def log_roll_out_stats(self, batch_num, mean_final_err, mean_final_topple_err, mean_final_no_topple_err, mean_topple_err, mean_no_topple_err):
        self.log_string(' ---- ROLLOUT after batch: %03d ----' % (batch_num))
        self.log_string('mean final step err: (%f m/s, %f rad/s, %f m, %f deg, %f)' % (mean_final_err[0], mean_final_err[1], mean_final_err[2], mean_final_err[3], mean_final_err[4]))
        self.log_string('mean final step err TOPPLING: (%f m/s, %f rad/s, %f m, %f deg, %f)' % (mean_final_topple_err[0], mean_final_topple_err[1], mean_final_topple_err[2], mean_final_topple_err[3], mean_final_topple_err[4]))
        self.log_string('mean final step err NO TOPPLING: (%f m/s, %f rad/s, %f m, %f deg, %f)' % (mean_final_no_topple_err[0], mean_final_no_topple_err[1], mean_final_no_topple_err[2], mean_final_no_topple_err[3], mean_final_no_topple_err[4]))
        self.log_string('mean all step err TOPPLING: (%f m/s, %f rad/s, %f m, %f deg, %f)' % (mean_topple_err[0], mean_topple_err[1], mean_topple_err[2], mean_topple_err[3], mean_topple_err[4]))
        self.log_string('mean all step err NO TOPPLING: (%f m/s, %f rad/s, %f m, %f deg, %f)' % (mean_no_topple_err[0], mean_no_topple_err[1], mean_no_topple_err[2], mean_no_topple_err[3], mean_no_topple_err[4]))

    def log_classify_stats(self, total_accuracy, topple_accuracy, confusion_mat, f_score):
        self.log_string('total classification accuracy: %f' % (total_accuracy))
        self.log_string('TOPPLE classification accuracy: %f' % (topple_accuracy))
        self.log_string('true pos: %f' % (confusion_mat[0]))
        self.log_string('false pos: %f' % (confusion_mat[1]))
        self.log_string('false neg: %f' % (confusion_mat[2]))
        self.log_string('true neg: %f' % (confusion_mat[3]))
        self.log_string('F1 Score: %f' % (f_score))

    def plot_rollout_results(self, mean_rollout_err, topple_rollout_err, no_topple_rollout_err):
        self.plot_rollout_errors(mean_rollout_err, 'all_rollout_err.png')
        self.plot_rollout_errors(topple_rollout_err, 'topple_rollout_err.png')
        self.plot_rollout_errors(no_topple_rollout_err, 'no_topple_rollout_err.png')

    def plot_rollout_errors(self, rollout_err, file_name):
        num_steps = rollout_err.shape[0]
        verr = rollout_err[:, 0]
        werr = rollout_err[:, 1]
        perr = rollout_err[:, 2]
        rerr = rollout_err[:, 3]
        axerr = rollout_err[:, 4]

        f, axarr = plt.subplots(5, 1, figsize=(10, 17), dpi=300)
        # vel
        axarr[0].plot(np.arange(0, num_steps), verr, '-r')
        axarr[0].set(xlabel='Time Steps', ylabel='Vel Err (m/s)')
        axarr[0].set_title('Mean Roll-out Error Over Time')
        # ang vel
        axarr[1].plot(np.arange(0, num_steps), werr, '-r')
        axarr[1].set(xlabel='Time Steps', ylabel='Ang Vel Err (rad/s)')
        # pos
        axarr[2].plot(np.arange(0, num_steps), perr, '-r')
        axarr[2].set(xlabel='Time Steps', ylabel='Pos Err (m)')
        # rot
        axarr[3].plot(np.arange(0, num_steps), rerr, '-r')
        axarr[3].set(xlabel='Time Steps', ylabel='Rot Err (deg)')
        axarr[4].plot(np.arange(0, num_steps), axerr, '-r')
        axarr[4].set(xlabel='Time Steps', ylabel='Axis Err')
        for ax in axarr.flat:
            ax.label_outer()
            ax.grid(True)

        plt.savefig(os.path.join(self.log_plots, file_name))

def main(args):
    tester = ToppleTest(args)
    return tester.test()

if __name__ == "__main__":
    # print(sys.argv[1:])
    main(sys.argv[1:])