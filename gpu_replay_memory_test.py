#!/usr/bin/env python3

from gpu_replay_memory import createGpuReplayMemory
import numpy as np
import numpy.testing as npt
import tempfile
import tensorflow as tf

ID = 0  # Mutable.
STATE_SIZE = 2  # So a whole memory is 7 floats.
SAMPLE_SIZE = 3


def get_next_variable_name():
  global ID
  next_variable_name = 'test_memory_' + str(ID)
  ID += 1
  return next_variable_name


def get_test_reward(i):
  return (i + 2) / 16384.0


def create_test_memory(i, reward=None):
  is_terminal = (i % 2)
  i = 10.0 * (i + 1)
  if reward is None:
    reward = get_test_reward(i)
  # ((old_state[0], old_state[1]), reward, action,
  #  (new_state[0], new_state[1]), is_terminal, q_values)
  return [[i, i + 1], reward, i + 3, [i + 4, i + 5], is_terminal, None]


def create_expected_memory(i, reward=None):
  # TODO remove this internal reordering of a memory?
  # TODO remove q_values entirely, since not actually storing it?
  is_terminal = float(i % 2)
  i = 10.0 * (i + 1)
  if reward is None:
    reward = get_test_reward(i)
  return [i + 3, reward, is_terminal, i, i + 1, i + 4, i + 5]


class TestGpuReplayMemory(tf.test.TestCase):

  def setUp(self):
    self.sess = tf.Session()
    self.sample_ph = tf.placeholder(dtype=tf.int32, shape=[SAMPLE_SIZE])
    self.memory, self.sample_ops = createGpuReplayMemory(
        self.sess, size=6, state_size=STATE_SIZE, sample_size=SAMPLE_SIZE,
        update_size=2, variable_name=get_next_variable_name(),
        random_sample_op=self.sample_ph)

  # Returns samples in the create_test_memory format.
  def assert_sample_correct(self, expected_memory_ids, sampled):
    for sampled_part in sampled:
      self.assertEqual(SAMPLE_SIZE, sampled_part.shape[0])

    # Check the states are the correct size.
    self.assertEqual(STATE_SIZE, sampled[0].shape[1])
    self.assertEqual(STATE_SIZE, sampled[3].shape[1])

    sampled = list(zip(*sampled))
    expected_sampled = [create_test_memory(i) for i in expected_memory_ids]
    for sample, expected_sample in zip(sampled, expected_sampled):
      # Use err since tf.float32 storage losses precision.
      self.assertArrayNear(expected_sample[0], sample[0], err=1e-7)
      self.assertNear(expected_sample[1], sample[1], err=1e-7)
      self.assertNear(expected_sample[2], sample[2], err=1e-7)
      self.assertArrayNear(expected_sample[3], sample[3], err=1e-7)
      self.assertNear(expected_sample[4], sample[4], err=1e-7)

  def test_initial_memory_correct(self):
    with self.sess.as_default():
      self.sess.run(tf.global_variables_initializer())
      npt.assert_equal(np.zeros((6, 7)),
                       self.memory._get_all_memories_for_test())

  def test_single_append_does_not_change_memory(self):
    with self.sess.as_default():
      self.sess.run(tf.global_variables_initializer())
      # Single append does not change memory since update_size > 1.
      self.assertFalse(self.memory.append(*create_test_memory(0)))
      npt.assert_equal(np.zeros((6, 7)),
                       self.memory._get_all_memories_for_test())

  def test_multiple_appends_does_change_memory(self):
    with self.sess.as_default():
      self.sess.run(tf.global_variables_initializer())
      self.assertFalse(self.memory.append(*create_test_memory(0)))
      self.assertTrue(self.memory.append(*create_test_memory(1)))
      npt.assert_equal([create_expected_memory(0),
                        create_expected_memory(1),
                        np.zeros(7),
                        np.zeros(7),
                        np.zeros(7),
                        np.zeros(7)],
                        self.memory._get_all_memories_for_test())

  def test_append_all(self):
    with self.sess.as_default():
      self.sess.run(tf.global_variables_initializer())
      self.assertTrue(self.memory.append_all(
          [create_test_memory(i) for i in range(6)]))
      npt.assert_equal([create_expected_memory(i) for i in range(6)],
                       self.memory._get_all_memories_for_test())

  def test_append_stores_non_updated_memories_to_update_later(self):
    with self.sess.as_default():
      self.sess.run(tf.global_variables_initializer())
      self.assertTrue(self.memory.append_all([
          create_test_memory(0), create_test_memory(1), create_test_memory(2)]))
      self.assertTrue(self.memory.append_all([
          create_test_memory(3), create_test_memory(4), create_test_memory(5)]))
      npt.assert_equal([create_expected_memory(i) for i in range(6)],
                       self.memory._get_all_memories_for_test())

  def test_append_wraps_around_when_updating(self):
    with self.sess.as_default():
      self.sess.run(tf.global_variables_initializer())
      self.assertTrue(self.memory.append_all([
          create_test_memory(0), create_test_memory(1), create_test_memory(2)]))
      self.assertTrue(self.memory.append_all([
          create_test_memory(3), create_test_memory(4), create_test_memory(5)]))
      self.assertTrue(self.memory.append_all([
          create_test_memory(6), create_test_memory(7), create_test_memory(8)]))
      # Note that the "8" memory is not updated yet.
      npt.assert_equal([create_expected_memory(i) for i in [6, 7, 2, 3, 4, 5]],
                       self.memory._get_all_memories_for_test())
      self.assertTrue(self.memory.append_all([
          create_test_memory(9), create_test_memory(10)]))
      # Note that the "8" memory is now updated, but the "10" memory is not.
      npt.assert_equal([create_expected_memory(i) for i in [6, 7, 8, 9, 4, 5]],
                       self.memory._get_all_memories_for_test())

  def test_append_a_lot(self):
      self.sess.run(tf.global_variables_initializer())
      self.assertTrue(self.memory.append_all(
          [create_test_memory(i) for i in range(0, 5)]))
      self.assertTrue(self.memory.append_all(
          [create_test_memory(i) for i in range(5, 10)]))
      self.assertTrue(self.memory.append_all(
          [create_test_memory(i) for i in range(10, 15)]))
      self.assertTrue(self.memory.append_all(
          [create_test_memory(i) for i in range(15, 20)]))
      self.assertTrue(self.memory.append_all(
          [create_test_memory(i) for i in range(20, 25)]))
      self.assertTrue(self.memory.append_all(
          [create_test_memory(i) for i in range(25, 30)]))
      self.assertTrue(self.memory.append_all(
          [create_test_memory(i) for i in range(30, 35)]))
      npt.assert_equal(
          [create_expected_memory(i) for i in [30, 31, 32, 33, 28, 29]],
          self.memory._get_all_memories_for_test())

  def test_reward_is_clipped(self):
    with self.sess.as_default():
      self.sess.run(tf.global_variables_initializer())
      orig = [-1.5, -0.5, 0.0, 0.5, 1.0, 10.0]
      exp =  [-1.0, -0.5, 0.0, 0.5, 1.0, 1.0]
      self.assertTrue(self.memory.append_all(
          [create_test_memory(i, reward=r) for i, r in zip(range(6), orig)]))
      npt.assert_equal(
          [create_expected_memory(i, reward=r) for i, r in zip(range(6), exp)],
          self.memory._get_all_memories_for_test())

  def test_sample_raises_exception(self):
    with self.assertRaises(Exception):
      memory.sample(32)

  def test_save_to_file_raises_exception(self):
    _, filepath = tempfile.mkstemp()
    with self.assertRaises(Exception):
      memory.save_to_file(filepath)

  def test_sample_op_raises_exception_when_memory_contains_no_samples(self):
    self.memory, self.sample_ops = createGpuReplayMemory(
        self.sess, size=6, state_size=2, sample_size=SAMPLE_SIZE, update_size=2,
        variable_name=get_next_variable_name())
    with self.sess.as_default():
      self.sess.run(tf.global_variables_initializer())
      with self.assertRaises(Exception):
        self.sess.run(self.sample_ops, feed_dict={self.sample_ph: [0, 0, 0]})

  def test_sample_op(self):
    with self.sess.as_default():
      self.sess.run(tf.global_variables_initializer())
      self.assertTrue(self.memory.append_all(
          [create_test_memory(i) for i in range(6)]))
      self.assertEqual(6, self.memory._current_size)
      self.assert_sample_correct([2, 5, 1], self.sess.run(
          self.sample_ops, feed_dict={self.sample_ph: [2, 5, 1]}))

  def test_sample_op_handles_duplicate_indexes(self):
    with self.sess.as_default():
      self.sess.run(tf.global_variables_initializer())
      self.assertTrue(self.memory.append_all(
          [create_test_memory(i) for i in range(6)]))
      self.assert_sample_correct([2, 2, 2], self.sess.run(
          self.sample_ops, feed_dict={self.sample_ph: [2, 2, 2]}))

  def test_current_size_is_max_size_after_appending_a_bunch(self):
    with self.sess.as_default():
      self.sess.run(tf.global_variables_initializer())
      self.assertTrue(self.memory.append_all(
          [create_test_memory(i) for i in range(0, 5)]))
      self.assertTrue(self.memory.append_all(
          [create_test_memory(i) for i in range(5, 10)]))
      self.assertTrue(self.memory.append_all(
          [create_test_memory(i) for i in range(10, 15)]))
      self.assertTrue(self.memory.append_all(
          [create_test_memory(i) for i in range(15, 20)]))
      self.assertEqual(6, self.memory._current_size)

  def test_sample_op_with_default_random_sample_op(self):
    self.memory, self.sample_ops = createGpuReplayMemory(
        self.sess, size=6, state_size=2, sample_size=SAMPLE_SIZE, update_size=2,
        variable_name=get_next_variable_name())
    with self.sess.as_default():
      self.sess.run(tf.global_variables_initializer())
      self.assertTrue(self.memory.append_all(
          [create_test_memory(i) for i in range(6)]))
      actual_sample = self.sess.run(self.sample_ops)
      for actual_sample_part in actual_sample:
        self.assertEqual(SAMPLE_SIZE, actual_sample_part.shape[0])
      possible_actions = [create_expected_memory(i)[0] for i in range(6)]
      for i in range(SAMPLE_SIZE):
        self.assertIn(actual_sample[2][i], possible_actions)

  def test_sample_op_with_default_random_sample_op_with_half_full_memory(self):
    self.memory, self.sample_ops = createGpuReplayMemory(
        self.sess, size=6, state_size=2, sample_size=SAMPLE_SIZE, update_size=2,
        variable_name=get_next_variable_name())
    with self.sess.as_default():
      self.sess.run(tf.global_variables_initializer())
      self.assertTrue(self.memory.append_all(
          [create_test_memory(i) for i in range(3)]))

      self.assertEqual(2, self.memory._current_size)
      all_actions = set()
      for i in range(250):
        actual_sample = self.sess.run(self.sample_ops)
        for actual_sample_part in actual_sample:
          self.assertEqual(SAMPLE_SIZE, actual_sample_part.shape[0])
          all_actions.update(actual_sample[2])

      possible_actions = [int(create_expected_memory(i)[0]) for i in range(2)]
      self.assertEqual(sorted(possible_actions), sorted(all_actions))


      self.assertTrue(self.memory.append(*create_test_memory(3)))
      self.assertEqual(4, self.memory._current_size)
      all_actions = set()
      for i in range(250):
        actual_sample = self.sess.run(self.sample_ops)
        for actual_sample_part in actual_sample:
          self.assertEqual(SAMPLE_SIZE, actual_sample_part.shape[0])
          all_actions.update(actual_sample[2])

      possible_actions = [int(create_expected_memory(i)[0]) for i in range(4)]
      self.assertEqual(sorted(possible_actions), sorted(all_actions))


if __name__ == '__main__':
  tf.test.main()

