import tensorflow as tf
import pointnet_cls_basic as pointnet
import utils.tf_util as tf_util

pcl_feat_size = 16
bn_decay = 0.9
weight_decay = 0.005

def placeholder_inputs(batch_size, num_points, num_steps):
    '''
    Returns placeholders for both geometry and state prediction modules.
    '''
    pcl_pl = tf.placeholder(tf.float32, shape=(batch_size, num_points, 3))
    pcl_feat_pl = tf.placeholder(tf.float32, shape=(batch_size, pcl_feat_size)) # for test-time
    lin_vel_pl = tf.placeholder(tf.float32, shape=(batch_size, num_steps, 3))
    ang_vel_pl = tf.placeholder(tf.float32, shape=(batch_size, num_steps, 3))
    pos_pl = tf.placeholder(tf.float32, shape=(batch_size, num_steps, 3))
    delta_rot_pl = tf.placeholder(tf.float32, shape=(batch_size, num_steps, 3))
    topple_label_pl = tf.placeholder(tf.float32, shape=(batch_size, num_steps))

    return pcl_pl, pcl_feat_pl, lin_vel_pl, ang_vel_pl, pos_pl, delta_rot_pl, topple_label_pl

def get_geom_model(pcl, is_training):
    '''
    Build the graph for the shape processing branch
    '''
    # first get shape feature
    pointnet_feat = get_pointnet_model(pcl, is_training, bn_decay=bn_decay)  
    # process pointnet output
    pt_vec = tf_util.fully_connected(pointnet_feat, 1024, weight_decay=weight_decay, bn=True, \
                           is_training=is_training, scope='geom_fc1', bn_decay=bn_decay)
    pt_vec = tf_util.fully_connected(pt_vec, 512,  weight_decay=weight_decay, bn=True, \
                           is_training=is_training, scope='geom_fc2', bn_decay=bn_decay)
    pt_vec = tf_util.fully_connected(pt_vec, 128,  weight_decay=weight_decay, bn=True, \
                           is_training=is_training, scope='geom_fc3', bn_decay=bn_decay)
    pt_vec = tf_util.fully_connected(pt_vec, 32,  weight_decay=weight_decay, bn=True, \
                           is_training=is_training, scope='geom_fc4', bn_decay=bn_decay)
    shape_feat = tf_util.fully_connected(pt_vec, pcl_feat_size,  weight_decay=weight_decay, bn=True, \
                           is_training=is_training, scope='geom_fc5', bn_decay=bn_decay)

    return shape_feat

def get_pointnet_model(pcl, is_training, bn_decay=None):
    '''
    PointNet classifier model. Returns only global feature.
    '''
    _, _, global_feat = pointnet.get_model(pcl, is_training, bn_decay=bn_decay)
    return global_feat


def get_dynamics_model(shape_feat, lin_vel, ang_vel, cell_type, num_cells, hidden_size, dropout_keep_prob, time_steps, is_training):
    '''
    Build the graph for the state prediction module
    '''
    batch_size = shape_feat.get_shape()[0].value

    # inputs are a 22-vec [lin_vel, ang_vel, shape_feat]
    tile_arg = tf.stack([tf.constant(1), time_steps, tf.constant(1)])
    step_shape = tf.tile(tf.expand_dims(shape_feat, 1), tile_arg)
    inputs = tf.concat([lin_vel, ang_vel, step_shape], axis=2)

    # ouputs are size 23, 4 size-3 vectors representing change in state: dv, dw, dp, d\theta, one topply classify logit
    num_params = 13
    W_hy = tf.get_variable('W_hy', shape=(hidden_size, num_params), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    l2_normalization = tf.multiply(tf.nn.l2_loss(W_hy), weight_decay, name='weight_loss')
    tf.add_to_collection('losses', l2_normalization)
    b_hy = tf.get_variable('b_hy', shape=(1, num_params), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

    if cell_type=='fc':
        # need to do it differently
        # num_cells used as number of FC layers, each with hidden_size nodes
        # inputs is B, num_steps, 12
        input_feat_size = inputs.get_shape()[2].value
        inputs = tf.reshape(inputs, [batch_size*time_steps, input_feat_size])
        cur_input = inputs
        for j in range(num_cells):
            cell_name = 'cell_fc' + str(j)
            # NOTE: batch norm causes some issues that really hinders training here - don't use it
            cur_input = tf_util.fully_connected(cur_input, hidden_size, weight_decay=weight_decay, bn=False, \
                        is_training=is_training, scope=cell_name, activation_fn=tf.nn.tanh, bn_decay=bn_decay)
        # final output
        y = tf.matmul(cur_input, W_hy) + b_hy
        y = tf.reshape(y, [batch_size, time_steps, num_params])

        init_state = tf.constant(0)

        return y, init_state, init_state # no state to return
 
    # then feed to RNN with velocites
    if cell_type=='rnn':
        rnn_cell = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicRNNCell(hidden_size), output_keep_prob=dropout_keep_prob) for i in range(0, num_cells)]
    if cell_type=='gru':
        rnn_cell = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer()), output_keep_prob=dropout_keep_prob) for i in range(0, num_cells)]
    if cell_type=='lstm':
        rnn_cell = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=tf.contrib.layers.xavier_initializer()), output_keep_prob=dropout_keep_prob) for i in range(0, num_cells)]
    
    if num_cells > 1:
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cell)
    else:
        rnn_cell = rnn_cell[0]

    
    init_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

    # feed through RNN
    # outputs are [batch, time_steps, hidden_size]
    outputs, state = tf.nn.dynamic_rnn(rnn_cell, inputs, initial_state=init_state, dtype=tf.float32)
    
    y = tf.matmul(tf.reshape(outputs, [batch_size*time_steps, hidden_size]), W_hy) + b_hy
    y = tf.reshape(y, [batch_size, time_steps, num_params])

    return y, state, init_state

def get_loss(pred, gt_lin_vel, gt_ang_vel, gt_pos, gt_delta_rot, topple_label, num_steps, loss_weights=None, is_training=None):
    '''
    Calculate loss for the given prediction and ground truth values.
    Input pred is size 13: 4 size-3 vectors representing change in state: dv, dw, dp, dtheta and topple logit
        (batch_size, time_steps, 13)
    '''
    lin_vel_weight, ang_vel_weight, pos_weight, angle_weight, axis_weight, classify_weight = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    if loss_weights != None:
        lin_vel_weight, ang_vel_weight, pos_weight, angle_weight, axis_weight, classify_weight = loss_weights

    # calculate change in linear vel for gt
    vel_t = gt_lin_vel[:,0:(num_steps-1)]
    vel_tp1 = gt_lin_vel[:, 1:]
    vel_diff = vel_tp1 - vel_t
    # calclate change in ang vel for gt
    angvel_t = gt_ang_vel[:, 0:(num_steps-1)]
    angvel_tp1 = gt_ang_vel[:, 1:]
    angvel_diff = angvel_tp1 - angvel_t
    # calculate change in pos for gt
    pos_t = gt_pos[:, 0:(num_steps-1)]
    pos_tp1 = gt_pos[:, 1:]
    pos_diff = pos_tp1 - pos_t
    # already have change in rot for gt (in axis-angle rep) - first entry is useless all zeros
    rot_diff = gt_delta_rot[:, 1:, :]
    
    # linear velocity
    gt_diff_lin_vel = tf.norm(pred[:,:,:3] - vel_diff, axis=2)
    lin_vel_rel = tf.norm(vel_diff, axis=2) + tf.norm(pred[:,:,:3], axis=2)
    lin_vel_loss = tf.reduce_mean(tf.reduce_mean(gt_diff_lin_vel / lin_vel_rel, axis=1))
    # angular velocity
    gt_diff_ang_vel = tf.norm(pred[:,:,3:6] - angvel_diff, axis=2)
    ang_vel_rel = tf.norm(angvel_diff, axis=2) + tf.norm(pred[:,:,3:6], axis=2)
    ang_vel_loss = tf.reduce_mean(tf.reduce_mean(gt_diff_ang_vel / ang_vel_rel, axis=1))
    # position
    gt_diff_pos = tf.norm(pred[:,:,6:9] - pos_diff, axis=2)
    pos_rel = tf.norm(pos_diff, axis=2) + tf.norm(pred[:,:,6:9], axis=2)
    pos_loss = tf.reduce_mean(tf.reduce_mean(gt_diff_pos / pos_rel, axis=1))
    # rotation
    gt_diff_rot = tf.norm(pred[:,:,9:12] - rot_diff, ord=1, axis=2)
    rot_rel = tf.norm(pred[:,:,9:12], ord=1, axis=2) + tf.norm(rot_diff, ord=1, axis=2)
    rot_loss = tf.reduce_mean(tf.reduce_mean(gt_diff_rot / rot_rel, axis=1))

    # topple classification
    topple_logits = pred[:, :, 12]
    ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=topple_label[:,1:], logits=topple_logits) # starts at second step since this is first prediction step
    classify_loss = tf.reduce_mean(tf.reduce_mean(ce_loss, axis=1))

    # final loss
    loss = tf.constant(lin_vel_weight)*lin_vel_loss + tf.constant(ang_vel_weight)*ang_vel_loss + \
            tf.constant(pos_weight)*pos_loss + tf.constant(angle_weight)*rot_loss + tf.constant(classify_weight)*classify_loss

    # now calculate meaningful errors
    # absolute error averaged over all timesteps for each sequence then the entire batch
    lin_vel_err = tf.reduce_mean(tf.reduce_mean(tf.abs(pred[:,:,:3] - vel_diff), axis=1), axis=0)
    ang_vel_err = tf.reduce_mean(tf.reduce_mean(tf.abs(pred[:,:,3:6] - angvel_diff), axis=1), axis=0)
    pos_err = tf.reduce_mean(tf.reduce_mean(tf.abs(pred[:,:,6:9] - pos_diff), axis=1), axis=0)

    pred_angle = tf.norm(pred[:,:,9:12], axis=2)
    gt_angle = tf.norm(rot_diff, axis=2)
    angle_err = tf.reduce_mean(tf.reduce_mean(tf.abs(gt_angle - pred_angle), axis=1), axis=0)
    #axis error
    axis_prod = pred[:,:,9:12] * rot_diff
    dot_prod = tf.reduce_sum(axis_prod, axis=2)
    denom =  pred_angle * gt_angle + tf.constant(1e-6) # for stability
    cos_sim = tf.constant(1.0) - (dot_prod / denom)
    # pay no attention when angle is extremely small < 1 degree
    zero = tf.constant(7.5e-3, dtype=tf.float32)
    mask = tf.math.greater(gt_angle, 7.5e-3)
    cos_sim = tf.boolean_mask(cos_sim, mask)
    axis_loss = tf.reduce_mean(cos_sim)

    errors = (lin_vel_err, ang_vel_err, pos_err, angle_err, axis_loss, classify_loss)

    return loss, errors

