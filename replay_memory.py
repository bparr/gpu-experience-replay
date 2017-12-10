"""Replay Memory Interface."""


class ReplayMemory:
  """Store and replay (sample) memories."""

  def append(self, old_state, reward, action, new_state, is_terminal, q_values):
    # Returns True if the memory changed. Sometimes an append is delayed.
    raise Exception('Not implemented.')

  def append_all(self, memories):
    # Returns True if the memory changed. Sometimes an append is delayed.
    raise Exception('Not implemented.')

  def sample(self, batch_size):
    raise Exception('Not implemented.')

  def save_to_file(self, filepath):
    raise Exception('Not implemented.')

