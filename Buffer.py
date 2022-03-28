import numpy as np

class ReplayBuffer():
	def __init__(self, obs_size, n_action, max_size):

		self.max_size = max_size
		self.state = np.empty((self.max_size, *obs_size), dtype=np.float32)
		self.action = np.empty((self.max_size, n_action), dtype=np.float32)
		self.reward = np.empty(self.max_size, dtype=np.float32)
		self.done = np.empty(self.max_size, dtype=np.bool)
		self.next_state = np.empty((self.max_size, *obs_size), dtype=np.float32)

		self.cntr = 0

	def store_transition(self, s, a, r, d, n_s):
		index = self.cntr % self.max_size

		self.state[index] = s
		self.action[index] = a
		self.reward[index] = r
		self.done[index] = d
		self.next_state[index] = n_s

		self.cntr += 1

	def getMiniBatch(self, batch_size):

		mem_size = min(self.cntr, self.max_size)
		batch = np.random.choice(mem_size, size=batch_size)

		s = self.state[batch]
		a = self.action[batch]
		r = self.reward[batch]
		d = self.done[batch]
		s_  = self.next_state[batch]

		return s, a, r, d, s_