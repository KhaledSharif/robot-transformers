import imageio
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from dm_control.rl.control import PhysicsError
import gym_aloha

env = gym.make("gym_aloha/AlohaInsertion-v0")
observation, info = env.reset()
frames = []

for _ in tqdm(range(1000)):
    action = env.action_space.sample()
    try:
        observation, reward, terminated, truncated, info = env.step(action)
    except PhysicsError as pe:
        print("Warning! Physics Error")
        continue
    
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave("example.mp4", np.stack(frames), fps=25)
