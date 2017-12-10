#!/usr/bin/env python3

import numpy as np
import pickle
from ram_replay_memory import RamReplayMemory
import tempfile
import tensorflow as tf


def get_test_reward(i):
  return (i + 2) / 16384.0

# Argumnet is the test state "id".
def generate_state(i, reward=None):
  s = str(i)
  reward = reward if reward is not None else get_test_reward(i)
  return 's' + s, reward, 10 * i, 's' + str(i + 1), i % 50 == 0, 'q_values' + s

class TestRamReplayMemory(tf.test.TestCase):

  def assert_sample_correct(self, result, expected_ids, expected_rewards=None):
    for result_part in result:
      self.assertEqual(len(expected_ids), len(result_part))

    old_states, rewards, actions, new_states, is_terminals = result
    expected_states = [generate_state(i) for i in expected_ids]
    for i, expected_state in enumerate(expected_states):
      self.assertEqual(expected_state[0], old_states[i])
      expected_reward = expected_state[1]
      if expected_rewards is not None:
        expected_reward = expected_rewards[i]
      self.assertEqual(expected_reward, rewards[i])
      self.assertEqual(expected_state[2], actions[i])
      self.assertEqual(expected_state[3], new_states[i])
      self.assertEqual(expected_state[4], is_terminals[i])

  def generate_sample_fn(self, expected_memory_ids, expected_sample_size,
                         return_indexes, expected_rewards=None):
    def sample_fn(memory, sample_size):
      self.assert_sample_correct(list(zip(*memory)), expected_memory_ids,
                                 expected_rewards=expected_rewards)
      self.assertEqual(expected_sample_size, sample_size)
      return [memory[i] for i in return_indexes]
    return sample_fn

  def test_append_errors_if_full_and_error_if_full_set_to_true(self):
    memory = RamReplayMemory(max_size=2, error_if_full=True)
    self.assertTrue(memory.append(*generate_state(0)))
    self.assertTrue(memory.append(*generate_state(1)))
    with self.assertRaises(Exception):
      memory.append(*generate_state(2))

  def test_sample_negative_amount_errors(self):
    memory = RamReplayMemory(max_size=2, error_if_full=False)
    self.assertTrue(memory.append(*generate_state(0)))
    with self.assertRaises(Exception):
      memory.sample(-1)

  def test_sample_from_empty_memory_errors(self):
    memory = RamReplayMemory(max_size=2, error_if_full=False)
    with self.assertRaises(Exception):
      memory.sample(32)

  def test_sample_from_half_filled_memory(self):
    memory = RamReplayMemory(max_size=2, error_if_full=False)
    self.assertTrue(memory.append(*generate_state(0)))
    self.assert_sample_correct(memory.sample(32), [0])

  def test_sample_from_full_filled_memory(self):
    memory = RamReplayMemory(max_size=5, error_if_full=False)
    self.assertTrue(memory.append(*generate_state(0)))
    self.assertTrue(memory.append(*generate_state(1)))
    self.assertTrue(memory.append(*generate_state(2)))
    self.assertTrue(memory.append(*generate_state(3)))
    self.assertTrue(memory.append(*generate_state(4)))
    sample_fn = self.generate_sample_fn([0, 1, 2, 3, 4], 2, [4, 1])
    self.assert_sample_correct(memory.sample(2, sample_fn=sample_fn), [4, 1])

  def test_append_reuses_space(self):
    memory = RamReplayMemory(max_size=2, error_if_full=False)
    self.assertTrue(memory.append(*generate_state(0)))
    self.assertTrue(memory.append(*generate_state(1)))
    self.assertTrue(memory.append(*generate_state(2)))
    self.assertTrue(memory.append(*generate_state(3)))
    self.assertTrue(memory.append(*generate_state(4)))
    sample_fn = self.generate_sample_fn([4, 3], 2, [0, 1])
    self.assert_sample_correct(memory.sample(2, sample_fn=sample_fn), [4, 3])

  def test_append_all(self):
    memory = RamReplayMemory(max_size=2, error_if_full=False)
    memory.append_all([generate_state(i) for i in range(5)])
    sample_fn = self.generate_sample_fn([4, 3], 2, [0, 1])
    self.assert_sample_correct(memory.sample(2, sample_fn=sample_fn), [4, 3])

  def test_reward_is_clipped(self):
    memory = RamReplayMemory(max_size=6, error_if_full=False)
    orig = [-1.5, -0.5, 0.0, 0.5, 1.0, 10.0]
    exp =  [-1.0, -0.5, 0.0, 0.5, 1.0, 1.0]
    memory.append_all([generate_state(i, reward=orig[i]) for i in range(6)])
    sample_fn = self.generate_sample_fn(list(range(6)), 6, list(range(6)),
                                        expected_rewards=exp)
    self.assert_sample_correct(memory.sample(6, sample_fn=sample_fn),
                               list(range(6)), expected_rewards=exp)

  def test_save_to_file(self):
    memory = RamReplayMemory(max_size=5, error_if_full=False)
    memory.append_all([generate_state(i) for i in range(5)])
    _, filepath = tempfile.mkstemp()
    memory.save_to_file(filepath)

    with open(filepath, 'rb') as f:
      result = pickle.load(f)
    self.assert_sample_correct(list(zip(*result)), [0, 1, 2, 3, 4])


if __name__ == '__main__':
  tf.test.main()

