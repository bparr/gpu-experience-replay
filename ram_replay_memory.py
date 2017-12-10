"""Replay Memory stored in RAM."""
import numpy as np
import pickle
import random
import replay_memory


class RamReplayMemory(replay_memory.ReplayMemory):
  """Store and replay (sample) memories.

  Clips reward to -1.0 <= reward <= 1.0 before storing in GPU memory.
  """
  def __init__(self, max_size, error_if_full):
    """Setup memory.

    You should specify the maximum size o the memory. Once the
    memory fills up oldest values are removed.
    """
    self._max_size = max_size
    self._error_if_full = error_if_full

    # Mutable
    self._memory = []
    self._replace_at = 0


  def append(self, old_state, reward, action, new_state, is_terminal, q_values):
    """Add a sample to the replay memory."""
    reward = max(-1.0, min(1.0, float(reward)))
    # TODO add back storing q_values? It was removed to make GPU version easier
    #     to implement.
    sample = (old_state, reward, action, new_state, is_terminal)
    if len(self._memory) >= self._max_size:
      if self._error_if_full:
        raise Exception('Replay memory unexpectedly full.')
      self._memory[self._replace_at] = sample
      self._replace_at = (self._replace_at + 1) % self._max_size
    else:
      self._memory.append(sample)
    return True


  def append_all(self, memories):
    for memory in memories:
      self.append(*memory)
    return True


  def sample(self, batch_size, sample_fn=random.sample):
    """Return samples from the memory.

    Returns
    --------
    (old_state_list, reward_list, action_list, new_state_list, is_terminal_list)
    """
    sample_size = min(batch_size, len(self._memory))
    if sample_size <= 0:
      raise Exception('Can not sample <= 0 samples.')
    zipped = list(zip(*sample_fn(self._memory, sample_size)))
    zipped[0] = np.reshape(zipped[0], (sample_size, -1))
    zipped[3] = np.reshape(zipped[3], (sample_size, -1))
    return zipped


  def save_to_file(self, filepath):
    with open(filepath, 'wb') as f:
      pickle.dump(self._memory, f)

