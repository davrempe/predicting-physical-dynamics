import argparse
import os, sys
from datetime import datetime
import time
import random
import numpy as np
import importlib
import sys
from os.path import join

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cwd = os.getcwd()
sys.path.append(join(cwd, 'scripts'))
sys.path.append(join(cwd, 'scripts/data'))

from data.topple_dataset import ToppleDataset
import data.data_list_loader

class ToppleTrain(object):
    
    def __init__(self, args):
        flags = self.parse_args(args)
        self.setup(flags)

    def parse_args(self, args):
        parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
        parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
        parser.add_argument('--model', default='topple_aa_rnn_classify', help='Model name')
        parser.add_argument('--data_list', default='./data/sim/dataset_lists/Cube5k', help='Root of the dataset lists path for the dataset to train and validate on.')
        parser.add_argument('--norm_info', default='./data/sim/normalization_info/cube_5k.pkl', help='Normalization info pickle file')
        parser.add_argument('--log', default='', help='Log directory [default: log/timestamp]')
        parser.add_argument('--num_pts', type=int, default=1024, help='Number to use in Point Cloud')
        parser.add_argument('--seq_len', type=int, default=15, help='Length of sequences to train on')
        parser.add_argument('--epochs', type=int, default=1001, help='num epochs to run')
        parser.add_argument('--validate_every', type=int, default=100, help='Number of epochs between each early stopping validation epoch [default: 5]')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training')
        parser.add_argument('--lr', type=float, default=0.001, help='Starting learning rate')
        parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay')
        parser.add_argument('--decay_rate', type=float, default=0.75, help='Decay rate for lr decay')

        # loss weights
        parser.add_argument('--lin_vel_loss', type=float, default=1.0, help='Amount to weight loss')
        parser.add_argument('--ang_vel_loss', type=float, default=1.0, help='Amount to weight loss')
        parser.add_argument('--pos_loss', type=float, default=1.0, help='Amount to weight loss')
        parser.add_argument('--angle_loss', type=float, default=1.0, help='Amount to weight loss')
        parser.add_argument('--axis_loss', type=float, default=1.0, help='Amount to weight axis loss')
        parser.add_argument('--classify_loss', type=float, default=2.0, help='Amount to weight classification loss')

        # network properties
        parser.add_argument('--num_units', type=int, default=1024, help='Number of units to use in each RNN unit [default: 128]')
        parser.add_argument('--cell_type', default='lstm', help='Cell type to use (rnn, gru, or lstm) [default: lstm]')
        parser.add_argument('--num_cells', type=int, default=3, help='Number of cells to stack to form the RNN module [default: 3]')
        parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Probability of keeping nodes between RNN cells [default:1.0 (no dropout)]')

        parser.add_argument('--no_ptnet', dest='no_ptnet', action='store_true')
        parser.set_defaults(no_ptnet=False)

        flags = parser.parse_args(args)

        # print(flags)

        return flags

    # all file paths assume this is script being run from the top level directory

    def setup(self, flags):
        global tf, tf_util # have to do these in a specific order

        # must set the GPU we want to use before importing tf
        self.gpu = flags.gpu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        
        import tensorflow as tf
        from utils import tf_util
        
        # get the toppling model
        self.model = importlib.import_module('models.' + flags.model)
        # path for pointnet model
        self.ptnet_weights = './pretrained/pointnet_cls_basic_model.ckpt'
        self.ptnet_path = './scripts/models/pointnet_cls_basic.py'

        # training settings
        self.batch_size = flags.batch_size
        dataset_list = flags.data_list
        norm_path = flags.norm_info
        self.num_pts = flags.num_pts
        self.seq_len = flags.seq_len 
        self.num_epochs = flags.epochs
        self.val_rate = flags.validate_every
        self.base_lr = flags.lr
        self.decay_step = flags.decay_step
        self.decay_rate = flags.decay_rate
        self.no_ptnet = flags.no_ptnet

        self.loss_weights = (flags.lin_vel_loss, flags.ang_vel_loss, flags.pos_loss, flags.angle_loss, flags.axis_loss, flags.classify_loss)

        # architecture settings
        self.num_units = flags.num_units
        self.cell_type = flags.cell_type
        self.num_cells = flags.num_cells
        self.drop_prob = flags.dropout_keep_prob

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
        os.system('cp %s %s' % (self.ptnet_path, log_scripts)) # bkp of pointnet model def
        os.system('cp %s %s' % (os.path.realpath(__file__), log_scripts)) # bkp of train procedure
        # plots directory
        self.log_plots = join(self.log_dir, 'plots')
        if not os.path.exists(self.log_plots):
            os.mkdir(self.log_plots)

        self.log_fout = open(os.path.join(self.log_dir, 'log_train.txt'), 'w')
        # log flags used
        self.log_fout.write(str(flags)+'\n')

        # load datasets
        train_paths, val_paths, _ = data.data_list_loader.load_dataset(dataset_list)

        print('===================================================================')
        print('TRAINING DATA: \n===================================================================')
        self.train_data = ToppleDataset(roots=train_paths, norm_info_file=norm_path, batch_size=self.batch_size, \
                                num_steps=self.seq_len, shuffle=True, num_pts=self.num_pts, perturb_pts=0.0)
        print('===================================================================')
        print('VALIDATION DATA: \n===================================================================')
        self.val_data = ToppleDataset(roots=val_paths, norm_info_file=norm_path, batch_size=self.batch_size, \
                                num_steps=self.seq_len, shuffle=False, num_pts=self.num_pts, perturb_pts=0.0)


    def log_string(self, out_str):
            self.log_fout.write(out_str+'\n')
            self.log_fout.flush()
            print(out_str)

    #
    # training functions
    #

    def get_learning_rate(self, batch):
        '''
        Get the decayed learning rate for the current training step.
        '''
        learning_rate = tf.train.exponential_decay(
                            self.base_lr,  # Base learning rate.
                            batch * self.batch_size,  # Current index into the dataset.
                            self.decay_step,          # Decay step.
                            self.decay_rate,          # Decay rate.
                            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
        # learning_rate = tf.constant(self.base_lr)
        return learning_rate

    def train(self):
        self.log_string('pid: %s'%(str(os.getpid())))
        # build model graph
        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(self.gpu)): 
                # get placeholders for input
                pcl_pl, pcl_feat_pl, lin_vel_pl, ang_vel_pl, pos_pl, delta_rot_pl, topple_label_pl = \
                                    self.model.placeholder_inputs(self.batch_size, self.num_pts, self.seq_len)            
                is_training_pl = tf.placeholder(tf.bool, shape=())

                batch = tf.get_variable('batch', [],
                    initializer=tf.constant_initializer(0), trainable=False)

                # get model
                pcl_feat_out = self.model.get_geom_model(pcl_pl, is_training_pl)
                pred_geom_out, _, _ = self.model.get_dynamics_model(pcl_feat_out, lin_vel_pl[:,:(self.seq_len - 1),:], ang_vel_pl[:,:(self.seq_len - 1),:], \
                                            self.cell_type, self.num_cells, self.num_units, self.drop_prob, self.seq_len - 1, \
                                                is_training_pl)

                # loss of state predictions for training
                loss, error_tup = self.model.get_loss(pred_geom_out, lin_vel_pl, ang_vel_pl, pos_pl, delta_rot_pl, topple_label_pl, self.seq_len, self.loss_weights, is_training_pl)
                # unnormalize errors
                lin_vel_err = tf.reshape(self.train_data.norm_info.max_lin_vel * error_tup[0], [1, 3])
                ang_vel_err = tf.reshape(self.train_data.norm_info.max_ang_vel * error_tup[1], [1, 3])
                pos_err = tf.reshape(self.train_data.norm_info.max_pos * error_tup[2], [1, 3])
                angle_err = tf.reshape(tf.tile(self.train_data.norm_info.max_delta_rot * tf.expand_dims(error_tup[3], axis=0), [3]), [1, 3]) # repeat for ease of data transfer
                axis_err = tf.reshape(tf.tile(tf.expand_dims(error_tup[4], axis=0), [3]), [1, 3]) # cos sim doesn't need unnorm
                classify_err = tf.reshape(tf.tile(tf.expand_dims(error_tup[5], axis=0), [3]), [1, 3])
                errors = tf.concat([lin_vel_err, ang_vel_err, pos_err, angle_err, axis_err, classify_err], axis=0)

                # get training operator
                learning_rate = self.get_learning_rate(batch)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(loss, global_step=batch)

                # Add ops to save and restore all the variables.
                saver = tf.train.Saver()
                # pointnet saver to load pretrained weights
                if not self.no_ptnet:
                    ptnet_variables = tf.contrib.framework.get_variables_to_restore()
                    ptnet_variables = [v for v in ptnet_variables if v.name.split('/')[0] in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'maxpool', 'fc1', 'fc2', 'fc3', 'dp1']]
                    ptnet_saver = tf.train.Saver(ptnet_variables)

                # count number of params
                # print(tf.trainable_variables())
                total_parameters = 0
                for variable in tf.trainable_variables():
                    shape = variable.get_shape()
                    variable_parameters = 1
                    for dim in shape:
                        variable_parameters *= dim.value
                    total_parameters += variable_parameters
                self.log_string('TOTAL PARAMS: ' + str(total_parameters))

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)

            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)
            # restore ptnet weights
            if not self.no_ptnet:
                ptnet_saver.restore(sess, self.ptnet_weights)
                self.log_string("PointNet model restored.")

            # inputs/outputs
            ops = {'pcl_pl': pcl_pl,
                'pcl_feat_pl' : pcl_feat_pl,
                'lin_vel_pl' : lin_vel_pl,
                'ang_vel_pl' : ang_vel_pl,
                'pos_pl' : pos_pl,
                'delta_rot_pl' : delta_rot_pl,
                'topple_label_pl' : topple_label_pl,
                'is_training_pl': is_training_pl,
                'pcl_feat_out' : pcl_feat_out,
                'pred_geom_out' : pred_geom_out,
                'loss': loss,
                'errors' : errors,
                'lr' : learning_rate,
                'train_op': train_op,
                'step': batch}

            #
            # main training loop
            #

            # training stats for plotting
            train_losses = []
            train_errors = []
            eval_losses = []
            eval_errors = []
            lrs = []

            min_eval_loss = float('inf')
            epoch_cnt = 0
            for epoch in range(self.num_epochs):
                # train
                self.log_string('**** EPOCH %03d TRAINING ****' % (epoch))
                train_loss, train_err, lr = self.one_epoch(sess, ops, True, self.train_data)
                train_losses.append(train_loss)
                train_errors.append(train_err)
                lrs.append(lr)

                # check if we need to eval for early stopping
                if epoch % self.val_rate == 0:
                    self.log_string('**** EPOCH %03d VALIDATION ****' % (epoch))
                    eval_loss, eval_err, _ = self.one_epoch(sess, ops, False, self.val_data)
                    eval_losses.append(eval_loss)
                    eval_errors.append(eval_err)
                    # also update plots
                    # self.plot_curves(epoch_cnt+1, train_losses, train_errors, eval_losses, eval_errors, lrs)

                    # save model if best so far
                    if eval_loss < min_eval_loss:
                        min_eval_loss = eval_loss
                        save_path = saver.save(sess, os.path.join(self.log_dir, "topple_aa_rnn_best_model.ckpt"))
                        self.log_string("BEST EVAL Model saved in file: %s" % save_path)


                # always save every val rate even if not best
                if epoch % self.val_rate == 0:
                    save_path = saver.save(sess, os.path.join(self.log_dir, "topple_aa_rnn_model.ckpt"))
                    self.log_string("Model saved in file: %s" % save_path)

                epoch_cnt += 1

            # save loss and error data for replotting later
            # use np.load(file) to get them later
            np.savez_compressed(os.path.join(self.log_dir, 'train_curve_data'), train_losses=np.array(train_losses), \
                            train_errors=np.array(train_errors), eval_losses=np.array(eval_losses), eval_errors=np.array(eval_errors), \
                            lers=np.array(lrs))


        self.log_fout.close()

    def one_epoch(self, sess, ops, is_training, dataset):
        '''
        Execute the model graph on all data in the given dataset. Returns loss and error averaged over the epoch.
        '''
        self.log_string(str(datetime.now()))

        num_batches = 0
        loss_sum = 0.
        lr_sum = 0.
        errors_sum = np.zeros((6, 3), dtype=np.float)
        while dataset.has_next_batch():
            # get the next batch of data
            cur_batch = dataset.next_batch(random_window=True)
            # build graph input
            feed_dict = {ops['pcl_pl']: cur_batch.point_cloud,
                            ops['lin_vel_pl'] : cur_batch.lin_vel,
                            ops['ang_vel_pl'] : cur_batch.ang_vel,
                            ops['pos_pl'] : cur_batch.pos,
                            ops['delta_rot_pl'] : cur_batch.delta_rot,
                            ops['topple_label_pl'] : cur_batch.topple_label,
                            ops['is_training_pl']: is_training }
            # execute graph
            if is_training:
                step, _, loss, errors, lr, pred = sess.run([ops['step'], ops['train_op'], ops['loss'], \
                                            ops['errors'], ops['lr'], ops['pred_geom_out']], feed_dict=feed_dict)
            else:
                # don't need train_op or lr
                step, loss, errors, pred = sess.run([ops['step'], ops['loss'], \
                                            ops['errors'], ops['pred_geom_out']], feed_dict=feed_dict)
                lr = 0

            # record stats
            loss_sum += loss
            errors_sum += errors
            lr_sum += lr
            num_batches += 1

        # log after epoch
        self.log_stats(num_batches+1, loss_sum / num_batches, errors_sum / num_batches)

        # find means
        loss_sum /= num_batches
        lr_sum /= num_batches
        errors_sum /= num_batches
        # final log
        self.log_stats(num_batches+1, loss_sum, errors_sum)

        # reset for next time
        dataset.reset()

        return loss_sum, errors_sum, lr_sum

    def log_stats(self, batch_num, loss, errors):
        self.log_string(' ---- after batch: %03d ----' % (batch_num))
        self.log_string('mean loss: %f' % (loss))
        self.log_string('mean lin vel err: (%f, %f, %f) m/s' % (errors[0, 0], errors[0, 1], errors[0, 2]))
        self.log_string('mean ang vel err: (%f, %f, %f) rad/s' % (errors[1, 0], errors[1, 1], errors[1, 2]))
        self.log_string('mean pos err: (%f, %f, %f) m' % (errors[2, 0], errors[2, 1], errors[2, 2]))
        self.log_string('mean angle err: %f deg' % (errors[3, 0]))
        self.log_string('mean axis err: %f ' % (errors[4, 0]))
        self.log_string('mean classification loss: %f ' % (errors[5, 0]))

    def plot_curves(self, num_epochs, train_losses, train_errors, eval_losses, eval_errors, lrs):
        '''
        Plot training curves given the loss and error statistics (and optionally learning rate).
        '''
        train_x = np.arange(1, num_epochs+1)
        eval_x = np.arange(1, num_epochs+1, self.val_rate)

        # loss
        fig = plt.figure()
        plt.plot(train_x, np.array(train_losses), 'r-', label='Train')
        plt.plot(eval_x, np.array(eval_losses), 'b-', label='Val')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(self.train_data.data.roots[0] + ' Loss')
        plt.savefig(os.path.join(self.log_plots, 'loss.png'))
        plt.close(fig)

        # learning rate
        fig = plt.figure()
        plt.plot(train_x, np.array(lrs), 'r-')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(self.train_data.data.roots[0] + ' Learning Rate')
        plt.savefig(os.path.join(self.log_plots, 'lr.png'))
        plt.close(fig)

        train_err_arr = np.array(train_errors)
        val_err_arr = np.array(eval_errors)

        # linear vel error
        train_lin_vel = np.linalg.norm(train_err_arr[:, 0, :], axis=1)
        eval_lin_vel = np.linalg.norm(val_err_arr[:, 0, :], axis=1)
        fig = plt.figure()
        plt.plot(train_x, train_lin_vel, 'r-', label='Train')
        plt.plot(eval_x, eval_lin_vel, 'b-', label='Val')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Lin Vel Err Norm (m/s)')
        plt.title(self.train_data.data.roots[0] + ' Lin Vel Err')
        plt.savefig(os.path.join(self.log_plots, 'lin_vel_err.png'))
        plt.close(fig)

        # angular vel error
        train_ang_vel = np.linalg.norm(train_err_arr[:, 1, :], axis=1)
        eval_ang_vel = np.linalg.norm(val_err_arr[:, 1, :], axis=1)
        fig = plt.figure()
        plt.plot(train_x, train_ang_vel, 'r-', label='Train')
        plt.plot(eval_x, eval_ang_vel, 'b-', label='Val')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Ang Vel Err Norm (rad/s)')
        plt.title(self.train_data.data.roots[0] + ' Ang Vel Err')
        plt.savefig(os.path.join(self.log_plots, 'ang_vel_err.png'))
        plt.close(fig)

        # position error
        train_pos = np.linalg.norm(train_err_arr[:, 2, :], axis=1)
        eval_pos = np.linalg.norm(val_err_arr[:, 2, :], axis=1)
        fig = plt.figure()
        plt.plot(train_x, train_pos, 'r-', label='Train')
        plt.plot(eval_x, eval_pos, 'b-', label='Val')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Pos Err Norm (m)')
        plt.title(self.train_data.data.roots[0] + ' Pos Err')
        plt.savefig(os.path.join(self.log_plots, 'pos_err.png'))
        plt.close(fig)

        # rotation angle error
        train_rot = train_err_arr[:, 3, 0]
        eval_rot = val_err_arr[:, 3, 0]
        fig = plt.figure()
        plt.plot(train_x, train_rot, 'r-', label='Train')
        plt.plot(eval_x, eval_rot, 'b-', label='Val')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Angle Err (deg)')
        plt.title(self.train_data.data.roots[0] + ' Rot Angle Err')
        plt.savefig(os.path.join(self.log_plots, 'angle_err.png'))
        plt.close(fig)

        # rotation axis error
        train_rot = train_err_arr[:, 4, 0]
        eval_rot = val_err_arr[:, 4, 0]
        fig = plt.figure()
        plt.plot(train_x, train_rot, 'r-', label='Train')
        plt.plot(eval_x, eval_rot, 'b-', label='Val')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Axis Err (1 - cos_sim)')
        plt.title(self.train_data.data.roots[0] + ' Rot Axis Err')
        plt.savefig(os.path.join(self.log_plots, 'axis_err.png'))
        plt.close(fig)

def main(args):
    trainer = ToppleTrain(args)
    trainer.train()

if __name__ == "__main__":
    # print(sys.argv[1:])
    main(sys.argv[1:])