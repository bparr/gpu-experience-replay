#!/usr/bin/env python


import gpu_replay_memory
import tensorflow as tf


# Example from melee-ai repository.
def create_model(input_shape, num_actions, model_name, create_network_fn,
                 learning_rate, batch_size, gamma, target_network=None,
                 replay_memory_sample_ops=None):
    """Create the Q-network model."""
    is_target_model = (target_network is None)
    with tf.name_scope(model_name):
        if replay_memory_sample_ops is not None:
            input_frames = replay_memory_sample_ops[3 if is_target_model else 0]
        else:
            input_frames = tf.placeholder(tf.float32, [None, input_shape],
                                          name ='input_frames')

        q_network = create_network_fn(
            input_frames, input_shape, num_actions,
            trainable=(not is_target_model))

        mean_max_Q =tf.reduce_mean(
            tf.reduce_max(q_network, axis=[1]), name='mean_max_Q')

        train_step = None
        if replay_memory_sample_ops is not None and not is_target_model:
            _, rewards_op, actions_op, __, is_terminal_op = (
                replay_memory_sample_ops)
            y = rewards_op + tf.where(
                is_terminal_op,
                tf.zeros(shape=[batch_size], dtype=tf.float32),
                tf.scalar_mul(gamma, tf.reduce_max(target_network, axis=1)))
            # From https://github.com/tensorflow/tensorflow/issues/206.
            enumerate_mask = tf.range(0, batch_size * int(q_network.shape[1]),
                                      q_network.shape[1])
            gathered_outputs = tf.gather(tf.reshape(q_network, [-1]),
                                         enumerate_mask + actions_op,
                                         name='gathered_outputs')
            loss = mean_huber_loss(y, gathered_outputs)
            train_step = tf.train.RMSPropOptimizer(learning_rate,
                decay=RMSP_DECAY, momentum=RMSP_MOMENTUM,
                epsilon=RMSP_EPSILON).minimize(loss)

    model = {
        'batch_size': batch_size,
        'q_network' : q_network,
        'input_frames' : input_frames,
        'train_step': train_step,
        'mean_max_Q' : mean_max_Q,
    }
    return model

