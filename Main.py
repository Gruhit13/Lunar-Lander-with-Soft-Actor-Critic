import os
import numpy as np

import gym

from Agent import Agent

def main():
	print("Starting Environment")
	
	env = gym.make('LunarLanderContinuous-v2')

	print("Action Spec:- ", env.action_space)
	print("Observation Spec:- ", env.observation_space)
	agent = Agent(env.observation_space.shape, env.action_space.shape[0], env)
	agent.load()
	EPISODE = 400
	MAX_FRAME = 1000
	SAVE_MODEL = 25
	RENDER_AT = 10

	score_history = []

	best_score = env.reward_range[0]

	is_nan = False
	frame_number = 0
	for e in range(EPISODE):
		obs = env.reset()
		episodic_reward = 0

		if (e+1) % RENDER_AT == 0:
			do_render = False
		else:
			do_render = False

		for i in range(MAX_FRAME):
			action, log_prob = agent.get_action(obs)

			if np.isnan(log_prob[0]):
				print("Obtained nan")
				is_nan = True
				break
			obs_, reward, done, _ = env.step(action)
			frame_number += 1
			
			if do_render:
				env.render()

			agent.remember(obs, action, reward, done, obs_)
			episodic_reward += reward
			obs = obs_

			agent.learn()
			# if frame_number % UPDATE_NET == 0:
			# 	agent.learn(BATCH_SIZE)
			if done: break

		if do_render:
			env.close()

		score_history.append(episodic_reward)
		avg_reward = np.mean(score_history[-100:])

		if avg_reward > best_score:
			agent.save()
			best_score = avg_reward

		print("%3d || Score = %.2f || Avg Score = %.2f || Best Score = %.2f || Frame = %3d"%(
				e+1, episodic_reward, avg_reward, best_score, i
			), end=" || ")

if __name__ == "__main__":
	main()