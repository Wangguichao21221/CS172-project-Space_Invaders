# opencv_only_version.py
import gymnasium as gym
import cv2
import numpy as np
from gymnasium.wrappers import ResizeObservation
from ultralytics import YOLO
import ale_py
import pygame
import random
import algo 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n",'--num',type=int,default=10)

args = parser.parse_args()
# 初始化
pygame.init()

# 设置PyGame窗口（用于捕获键盘事件）

# 加载模型
model = YOLO('./yolo.pt')

# 创建环境
env = gym.make(
    'ALE/SpaceInvaders-v5',
    render_mode='human',
    frameskip=1,  
    repeat_action_probability=0.0  
)

env = ResizeObservation(env, shape=(618, 838))

seed = random.randint(0,12523151241)
obs, info = env.reset(seed = seed)


display_size = (838,618)
num_rollouts = args.num
print(f"Test begin, {num_rollouts} runs in total.")



ACTION_NOOP = 0      # 不动
ACTION_FIRE = 1      # 发射
ACTION_RIGHT = 2     # 向右
ACTION_LEFT = 3      # 向左
ACTION_RIGHTFIRE = 4 # 向右并发射
ACTION_LEFTFIRE = 5  # 向左并发射

running = True
frame_index = 0
num_lives = 3
agent = algo.algorithm()
total_reward = 0
rewards_all_runs = 0
rollout_index = 1
while running and rollout_index <= num_rollouts:
    action =agent.step()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward+=reward
    if info['lives'] != num_lives:
        num_lives = info['lives'] 
        frame_index-=200 
    frame_index+=1
    obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    shelter_status = algo.shelter(obs_bgr) 

    results = model(obs_bgr, conf=0.25, verbose=False)
    agent.analyse(results, frame_index, shelter_mask=shelter_status)
    # 游戏重置
    if terminated or truncated:
        print(f"{rollout_index}-th run:score:{total_reward},seed: {seed}")
        rewards_all_runs+=total_reward
        total_reward = 0
        rollout_index+=1
        seed = random.randint(0,12523151241)
        obs, info = env.reset(seed = seed)


# 清理
env.close()
pygame.quit()
print(f"Test end, average score:{rewards_all_runs/num_rollouts}")