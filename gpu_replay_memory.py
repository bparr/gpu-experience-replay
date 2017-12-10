"""Replay Memory stored in GPU memory."""

import numpy as np
import replay_memory
import tensorflow as tf

_DEFAULT_UPDATE_SIZE = 100

def createGpuReplayMemory(sess, size, state_size, sample_size,
                          update_size=_DEFAULT_UPDATE_SIZE,
                          variable_name='replay_memory',
                          random_sample_op=None):
  """Create a GpuReplayMemory.

  Args:
    sess: A tf.Session.
    size: Number of memories the replay memory can hold.
    state_size: Number. The amount of tf.float32 in a single env state.
    sample_size: Number of memories to sample.
    update_size: Number of memories to update at a time. The GpuReplayMemory
        will delay updating until it has at least this many new memories.
    variable_name: Name of replay memory tf.Variable.
    random_sample_op: TensorFlow operation that returns sample_size many
        indexes, where each index 0 <= index < size. Defaults to using
        tf.random_uniform, which could include duplicate indexes.
  Returns:
    A (GpuReplayMemory instance, sample ops) tuples, where the sample ops is
    a tuple of (sampled_old_states, sampled_rewards, sampled_actions,
                sampled_new_states, sampled_is_terminals)
  """
  if size % update_size != 0:
    raise Exception('Size must be multiple of update size')

  # "memory" = action, reward, is_terminal, old_state, new_state
  single_memory_size = 2 * state_size + 3

  memories = tf.get_variable(
      name=variable_name,
      initializer=tf.zeros_initializer,
      dtype=tf.float32,
      shape=[size, single_memory_size],
      trainable=False)

  update_memories_ph = tf.placeholder(
      dtype=memories.dtype, shape=[update_size, single_memory_size])
  update_ops = []
  for i in range(0, size, update_size):
    update_ops.append(tf.scatter_update(
        memories, list(range(i, i + update_size)), update_memories_ph))

  current_size = tf.get_variable(
      name=variable_name + '_current_size',
      initializer=tf.zeros_initializer,
      dtype=tf.int32,
      shape=(),
      trainable=False)
  assign_next_current_size_op = tf.assign_add(current_size, update_size)

  replay_memory = GpuReplayMemory(
      sess, memories, update_ops, update_size, update_memories_ph,
      assign_next_current_size_op)

  if random_sample_op is None:
    random_sample_op = tf.random_uniform(shape=[sample_size], dtype=tf.int32,
                                         maxval=current_size)
  sampled_memories = tf.gather(memories, random_sample_op)
  sampled_actions = tf.cast(sampled_memories[:, 0], dtype=tf.int32)
  sampled_rewards = sampled_memories[:, 1]
  sampled_is_terminals = tf.cast(sampled_memories[:, 2], dtype=tf.bool)
  sampled_old_states = sampled_memories[:, 3:(3 + state_size)]
  sampled_new_states = sampled_memories[:, (3 + state_size):]

  return replay_memory, (sampled_old_states, sampled_rewards, sampled_actions,
                         sampled_new_states, sampled_is_terminals)


class GpuReplayMemory(replay_memory.ReplayMemory):
  """Store and replay (sample) memories.

  Clips reward to -1.0 <= reward <= 1.0 before storing in GPU memory.
  """
  def __init__(self, sess, memories, update_ops, update_size,
               update_memories_ph, assign_next_current_size_op):
    self._sess = sess
    self._memmories = memories
    self._update_ops = update_ops
    self._update_size = update_size
    self._update_memories_ph = update_memories_ph
    self._next_update = []  # Mutable.
    self._next_update_index = 0  # Mutable.
    self._max_size = len(self._update_ops) * self._update_size
    self._current_size = 0  # Mutable.
    # Only run when an update op is used for the first time.
    # Used to tell the GPU it can now sample from the next update slice.
    self._assign_next_current_size_op = assign_next_current_size_op

  def append(self, old_state, reward, action, new_state, is_terminal, q_values):
    return self.append_all(
        [(old_state, reward, action, new_state, is_terminal, q_values)])

  def append_all(self, memories):
    self._next_update.extend(memories)
    if len(self._next_update) < self._update_size:
      return False

    while len(self._next_update) >= self._update_size:
      memory_parts, self._next_update = (self._next_update[:self._update_size],
                                         self._next_update[self._update_size:])
      update_memories = []
      for old_state, reward, action, new_state, is_terminal, _ in memory_parts:
        # Improve network stability by clipping to -1.0 <= reward <= 1.
        reward = max(-1.0, min(1.0, float(reward)))
        # TODO remove this by having the worker just output lists?
        old_state = list(np.reshape(old_state, (-1, )))
        new_state = list(np.reshape(new_state, (-1, )))
        update_memories.append(
            [float(action), reward, float(is_terminal)] + old_state + new_state)

      self._sess.run(self._update_ops[self._next_update_index],
               feed_dict={self._update_memories_ph: update_memories})
      self._next_update_index += 1
      self._next_update_index %= len(self._update_ops)
      if self._current_size < self._max_size:
        self._current_size += self._update_size
        self._sess.run(self._assign_next_current_size_op)

    return True

  def sample(self, batch_size):
    raise Exception('Not implemented.')

  def save_to_file(self, filepath):
    raise Exception('Not implemented.')

  def _get_all_memories_for_test(self):
    return self._sess.run(self._memmories)

