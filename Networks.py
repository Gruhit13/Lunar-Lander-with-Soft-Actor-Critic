import tensorflow as tf
from tensorflow.keras.layers import Dense

import os

class CriticNetwork(tf.keras.Model):
	def __init__(self, fc1=256, fc2=256):

		super(CriticNetwork, self).__init__()
		self.fc1 = fc1
		self.fc2 = fc2

		self.layer1 = Dense(self.fc1, activation='relu')
		self.layer2 = Dense(self.fc2, activation='relu')
		self.q = Dense(1, activation=None)

	def call(self, state, action):
		action_value = self.layer1(tf.concat([state, action], axis=1))
		action_value = self.layer2(action_value)
		q = self.q(action_value)
		return q

class ValueNetwork(tf.keras.Model):
	def __init__(self, fc1=256, fc2=256):
		super(ValueNetwork, self).__init__()

		self.fc1 = fc1
		self.fc2 = fc2

		self.layer1 = Dense(self.fc1, activation='relu')
		self.layer2 = Dense(self.fc2, activation='relu')
		
		self.v = Dense(1, activation=None)

	def call(self, state):
		state_value = self.layer1(state)
		state_value = self.layer2(state_value)

		v = self.v(state_value)
		return v

class ActorNetwork(tf.keras.Model):
	def __init__(self, max_action, n_action=2, fc1=256, fc2=256):

		super(ActorNetwork, self).__init__()
		self.max_action = max_action
		self.n_action = n_action

		self.fc1 = fc1
		self.fc2 = fc2

		self.noise = 1e-6

		self.layer1 = Dense(self.fc1, activation='relu')
		self.layer2 = Dense(self.fc2, activation='relu')
		self.mu = Dense(self.n_action, activation='linear')
		self.sigma = Dense(self.n_action, activation='linear')

	def call(self, state):
		prob = self.layer1(state)
		prob = self.layer2(prob)

		mu = self.mu(prob)
		sigma = self.sigma(prob)

		sigma = tf.clip_by_value(sigma, -20, 2)
		sigma = tf.exp(sigma)

		return mu, sigma