# opencv_only_version.py
import gymnasium as gym
import cv2
import numpy as np
from gymnasium.wrappers import ResizeObservation
from ultralytics import YOLO
import ale_py
import pygame
import random
from algo import algorithm
# 初始化
pygame.init()

# 设置PyGame窗口（用于捕获键盘事件）

# 加载模型
model = YOLO('./best.pt')

# 创建环境
env = gym.make(
    'ALE/SpaceInvaders-v5',
    render_mode='human',
    frameskip=1,  
    repeat_action_probability=0.0  # 禁用随机重复动作（可选）
)
# 再封装 ResizeObservation
env = ResizeObservation(env, shape=(618, 838))
obs, info = env.reset()

# 设置显示大小

display_size = (838,618)

# 创建OpenCV窗口
cv2.namedWindow('Space Invader Detection')

# 调试工具2：鼠标点击标点功能
clicked_points = []  # 存储点击的点
def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数：点击时在画面上标点并打印坐标"""
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"[调试工具2] 点击位置: X={x}, Y={y}")

cv2.setMouseCallback('Space Invader Detection', mouse_callback)

print("游戏开始! 使用 ← →移动，↑射击，ESC退出")
print("[调试工具2] 在游戏窗口上点击鼠标左键可以标点并查看坐标")

# 按键状态



ACTION_NOOP = 0      # 不动
ACTION_FIRE = 1      # 发射
ACTION_RIGHT = 2     # 向右
ACTION_LEFT = 3      # 向左
ACTION_RIGHTFIRE = 4 # 向右并发射
ACTION_LEFTFIRE = 5  # 向左并发射

fps = 30  # 视频帧率（根据你的游戏循环速度调整，通常25/30）
frame_width = None  # 自动从第一帧获取分辨率
frame_height = None
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式（mp4格式）
video_writer = None  # 视频写入器对象

running = True
frame_index = 0
num_lives = 3
algo = algorithm()
while running:
    # 处理PyGame事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                observation, info = env.reset()
                print("游戏重置")
            elif event.key == pygame.K_h:
                print(frame_index,algo.enemy_record)
            elif event.key == pygame.K_c:
                # 清除所有标点
                clicked_points.clear()
                print("[调试工具2] 已清除所有标点")

    # 获取按键状态（持续按下）
    keys = pygame.key.get_pressed()
    
    # 组合按键逻辑
    moving_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    moving_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    firing = keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w]
    

    action =algo.step()

    obs, reward, terminated, truncated, info = env.step(action)
    if info['lives'] != num_lives:
        num_lives = info['lives'] 
        frame_index-=200 

    frame_index+=1
    obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    # --- 掩体检测逻辑 ---
    # 提取定义的 ROI (y: 460-517, x: 211-602)
    roi_y1, roi_y2, roi_x1, roi_x2 = 460, 517, 211, 602
    shelter_roi = obs_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # 转换为 HSV 并提取黄色
    hsv_roi = cv2.cvtColor(shelter_roi, cv2.COLOR_BGR2HSV)
    # 根据实测值 [9, 199, 181] 精准调整范围：
    # H: 5-15 (捕捉实测的 9), S: 150-255 (实测 199, 过滤背景), V: 100-255 (实测 181)
    lower_yellow = np.array([5, 150, 100])
    upper_yellow = np.array([15, 255, 255])
    mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
    
    # 转换为 0/1 状态矩阵
    shelter_status = (mask > 0).astype(np.uint8)

    results = model(obs_bgr, conf=0.25, verbose=False)
    if results[0].boxes is not None:
        annotated = results[0].plot(labels=True, conf=True, line_width=1, pil=False)
    else:
        annotated = obs
    algo.analyse(results, frame_index, shelter_mask=shelter_status)

    # 检测统计
    if results[0].boxes is not None:
        cv2.putText(annotated, f"frame: {frame_index}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
        # 在屏幕上显示玩家位置（实时显示）
        if algo.player_pos and algo.player_pos != (0, 0):
            player_x, player_y = algo.player_pos
            cv2.putText(annotated, f"Player: ({player_x:.1f}, {player_y:.1f})", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # 绘制 bonus 预测位置
        if algo.predicted_bonus_pos:
            px, py = algo.predicted_bonus_pos
            # 画一个十字或者圆圈表示预测命中点
            cv2.drawMarker(annotated, (int(px), int(py)), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(annotated, f"Bonus Predict: {px:.1f}", (int(px) + 10, int(py) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            # 画出 bonus 的速度
            cv2.putText(annotated, f"Bonus Speed: {algo.bonus_speed:.2f}", (10, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # 绘制敌人预测位置
        if hasattr(algo, 'predicted_enemy_pos') and algo.predicted_enemy_pos:
            ex, ey = algo.predicted_enemy_pos
            cv2.drawMarker(annotated, (int(ex), int(ey)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 15, 2)
            cv2.putText(annotated, "Enemy Predict", (int(ex) + 10, int(ey) + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 显示周期信息
        if algo.movement_pattern['is_calibrated']:
            p = algo.movement_pattern
            cv2.putText(annotated, f"Period: {p['frames_per_step']}f, Step: {p['pixels_per_step']:.1f}px", 
                       (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 调试工具2：在画面上绘制所有标点
    for px, py in clicked_points:
        # 画一个红色的圆圈标记
        cv2.circle(annotated, (px, py), 8, (0, 0, 255), 2)
        # 在标记旁边显示X坐标值
        cv2.putText(annotated, f"X:{px}", (px + 12, py - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # cv2.imwrite('./first_frame.png',annotated)
    # exit()
    # 显示
    cv2.imshow('Space Invader Detection', annotated)
    if video_writer is None:
        frame_height, frame_width = annotated.shape[:2]
        video_writer = cv2.VideoWriter(
            'space_invader_detection.mp4',  # 输出视频文件名
            fourcc,
            fps,
            (frame_width, frame_height)     # 帧分辨率（宽, 高）
        )
    
    # 3. 将当前帧写入视频
    video_writer.write(annotated)
    # 游戏重置
    if terminated or truncated:
        obs, info = env.reset()
        print("游戏重置")

# 清理
env.close()
cv2.destroyAllWindows()
pygame.quit()
print("游戏结束")