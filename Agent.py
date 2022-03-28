import tensorflow as tf

from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import os

from Buffer import ReplayBuffer
from Networks import CriticNetwork, ValueNetwork, ActorNetwork

import numpy as np

class Agent():
	def __init__(self, input_dims, n_actions, env, 
		alpha=0.0003, beta=0.0003, gamma=0.99, tau=0.005, max_size=10_00_000, 
		fc1=256, fc2=256, batch_size=256, reward_scale=2, chkpt_dir="./Models"):

			
		self.gamma = gamma
		self.tau = tau
		self.buffer = ReplayBuffer(input_dims, n_actions, max_size)
		self.n_actions = n_actions
		self.batch_size = batch_size
		self.chkpt_dir = chkpt_dir
		self.noise = 1e-6

		self.min_action = env.action_space.low[0]
		self.max_action = env.action_space.high[0]
		if not os.path.isdir(self.chkpt_dir):
			os.mkdir(self.chkpt_dir)

		self.actor = ActorNetwork(max_action=env.action_space.high, n_action=n_actions)
		self.critic_1 = CriticNetwork()
		self.critic_2 = CriticNetwork()
		self.value = ValueNetwork()
		self.target_value = ValueNetwork()

		self.actor.compile(optimizer=Adam(learning_rate=alpha))
		self.critic_1.compile(optimizer=Adam(learning_rate=beta))
		self.critic_2.compile(optimizer=Adam(learning_rate=beta)) 
		self.value.compile(optimizer=Adam(learning_rate=beta))
		self.target_value.compile(optimizer=Adam(learning_rate=beta))


		self.scale_reward = reward_scale
		self.update_target_parameter(tau=1)

	def update_target_parameter(self, tau=None):

		if tau is None:
			tau = self.tau

		target_weights = self.target_value.get_weights()
		weights = []
		for i, weight in enumerate(self.value.get_weights()):
			weights.append(weight*tau + target_weights[i]*(1 - tau))

		self.target_value.set_weights(weights)

	def save(self):
		if self.buffer.cntr > self.batch_size:
			print("......Model Saved......")
			self.actor.save(os.path.join(self.chkpt_dir, "actor"), save_format="tf")
			self.critic_1.save(os.path.join(self.chkpt_dir, "critic1"), save_format="tf")
			self.critic_2.save(os.path.join(self.chkpt_dir, "critic2"), save_format="tf")
			self.value.save(os.path.join(self.chkpt_dir, "value"), save_format="tf")
			self.target_value.save(os.path.join(self.chkpt_dir, "target_value"), save_format="tf")

	def load(self):
		print("......Model Loaded......")
		self.actor = tf.keras.models.load_model(os.path.join(self.chkpt_dir, "actor"))
		self.critic_1 = tf.keras.models.load_model(os.path.join(self.chkpt_dir, "critic1"))
		self.critic_2 = tf.keras.models.load_model(os.path.join(self.chkpt_dir, "critic2"))
		self.value = tf.keras.models.load_model(os.path.join(self.chkpt_dir, "value"))
		self.target_value = tf.keras.models.load_model(os.path.join(self.chkpt_dir, "target_value"))

	def sample_normal(self, state):

		mu, sigma = self.actor(state)

		probabilities = tfp.distributions.Normal(mu, sigma, allow_nan_stats=False)

		actions = probabilities.sample()
		log_prob = probabilities.log_prob(actions)
		
		actions = tf.math.tanh(actions)
		
		log_prob -= tf.math.log(1-tf.math.pow(actions, 2)+self.noise)
		log_prob = tf.math.reduce_sum(log_prob, axis=1, keepdims=True)

		return actions, log_prob
			

	def get_action(self, state):
		state = tf.convert_to_tensor([state], dtype=tf.float32)

		actions, log_prob = self.sample_normal(state)
		# actions = tf.clip_by_value(actions, self.min_action, self.max_action)
		return actions[0].numpy(), log_prob.numpy()

	def remember(self, s, a, r, d, n_s):
		self.buffer.store_transition(s, a, r, d, n_s)


	def learn(self):
		if self.buffer.cntr < self.batch_size:
			return

		s, a, r, d, s_ = self.buffer.getMiniBatch(self.batch_size)
		s = tf.convert_to_tensor(s, dtype=tf.float32)
		a = tf.convert_to_tensor(a, dtype=tf.float32)
		r = tf.convert_to_tensor(r, dtype=tf.float32)
		s_ = tf.convert_to_tensor(s_, dtype=tf.float32)

		#	Train Value network
		with tf.GradientTape() as tape:
			value = tf.squeeze(self.value(s), axis=1)
			current_act_policy, log_prob = self.sample_normal(s)

			log_prob = tf.squeeze(log_prob, axis=1)
			q1_new_pi = self.critic_1(s, current_act_policy)
			q2_new_pi = self.critic_2(s, current_act_policy)

			critic_value = tf.squeeze(tf.math.minimum(q1_new_pi, q2_new_pi), axis=1)

			target_value = critic_value - log_prob
			target_value_loss = 0.5 * tf.keras.losses.MSE(target_value, value)


		if tf.math.is_nan(target_value_loss):
			print("Target Value loss is None")
			print("Value:- ", value)

		value_gradient = tape.gradient(target_value_loss, self.value.trainable_variables)
		self.value.optimizer.apply_gradients(zip(value_gradient, self.value.trainable_variables))

		#	Train Actor Network
		with tf.GradientTape() as tape:
			new_policy_action, log_prob = self.sample_normal(s)

			log_prob = tf.squeeze(log_prob, axis=1)

			q1_new_pi = self.critic_1(s, new_policy_action)
			q2_new_pi = self.critic_2(s, new_policy_action)
			critic_value = tf.squeeze(tf.math.minimum(q1_new_pi, q2_new_pi), axis=1)
			actor_loss = log_prob - critic_value
			actor_loss = tf.math.reduce_mean(actor_loss)

		if tf.math.is_nan(actor_loss):
			print("This mf is nan:- ", actor_loss)
			print("Log_prob:- ", log_prob)
			print("critic value:- ", critic_value)
		gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
		self.actor.optimizer.apply_gradients(zip(gradient, self.actor.trainable_variables))

		#	Train Critic Network
		with tf.GradientTape(persistent=True) as tape:
			value_ = tf.squeeze(self.target_value(s_), axis=1)
			q_hat = self.scale_reward*r + (self.gamma * value_ * (1-d))
			q1_old_pi = tf.squeeze(self.critic_1(s, a), axis=1)
			q2_old_pi = tf.squeeze(self.critic_2(s, a), axis=1)
			
			critic_1_loss = 0.5 * tf.keras.losses.MSE(q_hat, q1_old_pi)
			critic_2_loss = 0.5 * tf.keras.losses.MSE(q_hat, q2_old_pi)

		if tf.math.is_nan(critic_1_loss):
			print("Critic 1 loss is nan")
			print("Q_Hat:- ", q_hat)
			print("Value_ = ", value_)

		if tf.math.is_nan(critic_2_loss):
			print("Critic 2 loss is nan")
			print("Q_Hat:- ", q_hat)
			print("Value_ = ", value_)

		critic_1_grad = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
		self.critic_1.optimizer.apply_gradients(zip(critic_1_grad, self.critic_1.trainable_variables))

		critic_2_grad = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
		self.critic_2.optimizer.apply_gradients(zip(critic_2_grad, self.critic_2.trainable_variables))

		self.update_target_parameter()