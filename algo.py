import numpy as np
import cv2
ACTION_NOOP = 0      # 不动
ACTION_FIRE = 1      # 发射
ACTION_RIGHT = 2     # 向右
ACTION_LEFT = 3      # 向左
ACTION_RIGHTFIRE = 4 # 向右并发射
ACTION_LEFTFIRE = 5  # 向左并发射
def shelter(observe):
   # --- 掩体检测逻辑 ---
    # 提取定义的 ROI (y: 460-517, x: 211-602)
    roi_y1, roi_y2, roi_x1, roi_x2 = 460, 517, 211, 602
    shelter_roi = observe[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # 转换为 HSV 并提取黄色
    hsv_roi = cv2.cvtColor(shelter_roi, cv2.COLOR_BGR2HSV)
    # 根据实测值 [9, 199, 181] 精准调整范围：
    # H: 5-15 (捕捉实测的 9), S: 150-255 (实测 199, 过滤背景), V: 100-255 (实测 181)
    lower_yellow = np.array([5, 150, 100])
    upper_yellow = np.array([15, 255, 255])
    mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
    
    # 转换为 0/1 状态矩阵
    shelter_status = (mask > 0).astype(np.uint8)
    return shelter_status
class algorithm():
  def __init__(self):
    self.player_pos = (0,0)
    self.enemy_bullet_speed = 3.4
    self.player_bullet_speed =5.5
    self.player_speed = 2.6
    self.bonus_speed = -1.07
    self.bonus_history = []  # 存储 bonus 的历史位置
    self.bonus_detected_frames = 0
    self.bonus_start_pos = None
    self.positions = None
    self.target = None
    self.danger = False
    self.bullets = []
    self.enemy_record = (0,0,0)
    self.bonus_exist = False
    self.bonus_pos = (0,0)
    self.shelter_mask = None # 用于存储掩体像素状态
    self.bullet_width = 8    # 默认子弹宽度
    self.roi_x_offset = 211  # ROI 在原图中的 X 偏移
    
    # 周期性移动追踪相关
    self.movement_pattern = {
        'pixels_per_step': 0,
        'frames_per_step': 0,
        'last_jump_frame': 0,
        'last_x': None,
        'is_calibrated': False,
        'consecutive_stable_frames': 0
    }
    self.reference_invader_id = None
    self.current_frame = 0
  def get_record(self):
     return self.enemy_record
  def detect_bonus(self,frame):
      if 'bonus' in self.positions:
        pos = self.positions['bonus'][0]
        
        if not self.bonus_exist:
            # 刚出现，初始化
            self.bonus_start_pos = pos
            self.bonus_detected_frames = 1
            # 初始速度保持默认或设为0，直到有足够帧数计算
        else:
            self.bonus_detected_frames += 1
            if self.bonus_detected_frames <= 20:
                # 计算平均速度: (当前x - 起始x) / (经过的帧数 - 1)
                # 注意：第一帧时 self.bonus_detected_frames 为 1，dx 为 0
                if self.bonus_detected_frames > 1:
                    self.bonus_speed = (pos[0] - self.bonus_start_pos[0]) / (self.bonus_detected_frames - 1)
        
        self.bonus_pos = pos
        self.bonus_exist = True
      else:
        self.bonus_exist = False
        self.bonus_detected_frames = 0
        self.bonus_start_pos = None

  def update_movement_pattern(self, frame_index):
      # 找到一个参考物（最下方、最左边的敌人）
      current_x = None
      for i in range(6, 0, -1):
          key = f"invader{i}"
          if key in self.positions and len(self.positions[key]) > 0:
              # positions[key] 已经在 analyse 中按 x 排序了
              current_x = self.positions[key][0][0]
              break
      
      if current_x is None:
          return

      p = self.movement_pattern
      if p['last_x'] is None:
          p['last_x'] = current_x
          p['last_jump_frame'] = frame_index
          return

      dx = current_x - p['last_x']
      
      # 检测是否发生了移动（跳跃）
      # 这里使用 abs(dx) > 0.5 过滤掉 YOLO 检测的小幅抖动
      # 同时如果 abs(dx) 过大（比如 > 15），可能是参考物（最左边的敌人）被消灭了，换了一个参考物，此时不应视为跳跃
      if 0.5 < abs(dx) < 15.0:
          frames_passed = frame_index - p['last_jump_frame']
          
          # 如果已经校准，且检测到的跳跃与规律不符，说明规律变了（比如敌人变少速度加快）
          if p['is_calibrated']:
              # 如果跳跃像素或间隔帧数发生明显变化
              if abs(abs(dx) - abs(p['pixels_per_step'])) > 1.0 or abs(frames_passed - p['frames_per_step']) > 2:
                  # print(f"检测到周期变化: dx={dx}, frames={frames_passed}")
                  p['is_calibrated'] = False

          # 更新规律
          p['pixels_per_step'] = dx
          p['frames_per_step'] = frames_passed
          p['last_jump_frame'] = frame_index
          p['last_x'] = current_x
          
          # 一旦捕捉到一次有效的移动，就认为是进入了新周期
          p['is_calibrated'] = True
      elif abs(dx) >= 15.0:
          # 参考点发生大范围偏移，通常是原参考敌人被击杀，重置参考点
          p['last_x'] = current_x
          p['last_jump_frame'] = frame_index
      else:
          # 没有移动或抖动过小，保持现状
          pass

  def step(self):
      if self.positions == None:
         return ACTION_FIRE
      action = self.avoid_bullet()
      if not self.danger:
        if not self.bonus_exist:
          action = self.move_shoot()
        else:
           action = self.shoot_bonus()
      return action
  def analyse(self,results,frame_index, shelter_mask=None):
    self.current_frame = frame_index
    self.shelter_mask = shelter_mask
    class_positions = {}
    for result in results:
        class_names = result.names
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                xyxy = box.xyxy[0].cpu().numpy()  # 存储xyxy格式为例
                x_mid = (xyxy[0]+xyxy[2])/2
                y_mid = (xyxy[1]+xyxy[3])/2
                
                # 如果是子弹，更新子弹宽度
                if class_names[int(box.cls[0])] == "bullet":
                    self.bullet_width = max(self.bullet_width, xyxy[2] - xyxy[0])

                # 按类别名分组
                if class_name not in class_positions:
                    class_positions[class_name] = []
                class_positions[class_name].append((x_mid,y_mid))
    
    for cls_name in class_positions:
        class_positions[cls_name].sort(key=lambda p: p[0])

    self.positions = class_positions.copy()
    self.update_movement_pattern(frame_index)
    self.detect_bonus(frame_index)
    
    # 计算 bonus 预测位置用于显示
    if self.bonus_exist:
        target_x = self.bonus_pos[0]
        target_y = self.bonus_pos[1]
        dy = self.player_pos[1] - target_y
        t = dy / self.player_bullet_speed
        shoot_x = target_x + t * self.bonus_speed
        self.predicted_bonus_pos = (shoot_x, target_y)
    else:
        self.predicted_bonus_pos = None
    
    # 计算敌人预测位置用于显示（新逻辑）
    self.predicted_enemy_pos = None
    if self.movement_pattern['is_calibrated'] and self.target:
        tx, ty = self.target
        dy = self.player_pos[1] - ty
        t = dy / self.player_bullet_speed
        p = self.movement_pattern
        
        # 预测 t 帧后的位置
        # 计算从当前帧到子弹命中帧之间会发生多少次跳跃
        # 注意：frame_index 是当前检测到的帧
        future_frame = frame_index + t
        # 计算从上一次跳跃到现在过了多少帧
        # 预测未来会发生的跳跃次数
        if p['frames_per_step'] > 0:
            num_jumps = int((future_frame - p['last_jump_frame']) // p['frames_per_step'])
            predicted_x = tx + num_jumps * p['pixels_per_step']
            self.predicted_enemy_pos = (predicted_x, ty)
        else:
            self.predicted_enemy_pos = (tx, ty)

    # print(class_positions)
    # 打印按类别分组的结果
    # print("按类别分组的目标位置：")
    for cls_name, positions in class_positions.items():
        if cls_name == "bullet":
          for pos in positions:
            self.update_bullets(pos,frame_index)
        else:
          self.update_bullets(None,frame_index)
        if cls_name == "player":
           self.player_pos = class_positions['player'][0]
          #  print(self.player_pos)
  def count_total_enemies(self):
    """统计所有敌人的总数"""
    if self.positions is None:
      return 0
    total = 0
    for i in range(1, 7):  # invader1 到 invader6
      invader_key = f"invader{i}"
      if invader_key in self.positions:
        total += len(self.positions[invader_key])
    return total

  def is_path_clear(self, x):
    # 如果敌人总数小于 10，无需考虑掩体限制，自由开火
    if self.count_total_enemies() < 10:
      return True
    
    if self.shelter_mask is None:
      return True
    
    # 子弹通道范围 [x-m, x+m]
    m = self.bullet_width / 2 + 1 # 加上 1 像素的容错边距
    x_start = int(x - m - self.roi_x_offset)
    x_end = int(x + m - self.roi_x_offset)
    
    # 如果完全在 ROI 左右两侧，则没有遮挡
    if x_end < 0 or x_start >= self.shelter_mask.shape[1]:
      return True
    
    # 限制在 mask 范围内
    x_start = max(0, x_start)
    x_end = min(self.shelter_mask.shape[1] - 1, x_end)
    
    # 检测该通道内是否有黄色像素 (1)
    # 如果和为 0，说明没有黄色像素，路径通畅
    return np.sum(self.shelter_mask[:, x_start:x_end+1]) == 0

  def move_shoot(self):
    # 1. 收集所有潜在目标（优先最后两排）
    potential_targets = []
    existing_rows = []
    for i in range(6, 0, -1):
        if f"invader{i}" in self.positions:
            existing_rows.append(i)
            if len(existing_rows) == 2:
                break
    
    for r in existing_rows:
        potential_targets.extend(self.positions[f"invader{r}"])
    
    if not potential_targets:
        # 如果前两排没找到，找更上面的
        for i in range(1, 7):
            if i not in existing_rows and f"invader{i}" in self.positions:
                potential_targets.extend(self.positions[f"invader{i}"])
                break

    if not potential_targets:
        return ACTION_NOOP

    # 2. 评估每个目标的预测位置，并计算与玩家的距离
    best_target = None
    min_dx = float('inf')
    best_shoot_x = 0
    p = self.movement_pattern

    for target in potential_targets:
        tx, ty = target
        dy = self.player_pos[1] - ty
        t = dy / self.player_bullet_speed
        
        # 计算预测位置
        if p['is_calibrated'] and p['frames_per_step'] > 0:
            future_frame = self.current_frame + t
            num_jumps = int((future_frame - p['last_jump_frame']) // p['frames_per_step'])
            shoot_x = tx + num_jumps * p['pixels_per_step']
        else:
            shoot_x = tx
        
        # 检查预测位置的路径是否畅通
        if self.is_path_clear(shoot_x):
            dist = abs(shoot_x - self.player_pos[0])
            if dist < min_dx:
                min_dx = dist
                best_target = target
                best_shoot_x = shoot_x

    # 3. 如果没有路径畅通的目标，退而求其次选一个最近的（哪怕被挡住）
    if best_target is None:
        for target in potential_targets:
            tx, ty = target
            dy = self.player_pos[1] - ty
            t = dy / self.player_bullet_speed
            if p['is_calibrated'] and p['frames_per_step'] > 0:
                future_frame = self.current_frame + t
                num_jumps = int((future_frame - p['last_jump_frame']) // p['frames_per_step'])
                shoot_x = tx + num_jumps * p['pixels_per_step']
            else:
                shoot_x = tx
            
            dist = abs(shoot_x - self.player_pos[0])
            if dist < min_dx:
                min_dx = dist
                best_target = target
                best_shoot_x = shoot_x

    if best_target is None:
        return ACTION_NOOP

    self.target = best_target
    dx = best_shoot_x - self.player_pos[0]
    path_clear = self.is_path_clear(self.player_pos[0])

    if -3 < dx < 3:
      if path_clear:
        return ACTION_FIRE
      else:
        return ACTION_NOOP
    elif dx < -3:
       return ACTION_LEFTFIRE
    else:
       return ACTION_RIGHTFIRE
  def shoot_bonus(self):
    if 'bonus' not in self.positions:
       return ACTION_NOOP
  
    pos = self.positions['bonus'][0]
    target_x = pos[0]
    target_y = pos[1]
    dy = self.player_pos[1]-target_y
    t = dy/self.player_bullet_speed
    shoot_x = target_x+t*self.bonus_speed
    dx = shoot_x-self.player_pos[0]
    # print(f"target:{self.target},player:{self.player_pos},dy:{dy},t:{t},shoot_x:{shoot_x},dx:{dx}")
    if -3<dx<3:
      return ACTION_FIRE
    elif dx<-3:
       return ACTION_LEFTFIRE
    else:
       return ACTION_RIGHTFIRE

  def avoid_bullet(self):
    for idx,bullet in enumerate(self.bullets):
       if bullet[2]==1:
          # 计算中心距离
          dx = bullet[0] - self.player_pos[0]
          dy = bullet[1] - self.player_pos[1]
          
          # y 范围增加到 180 像素，提供更早的预警，x 范围设定为覆盖玩家身位（约 60 像素）
          if abs(dx) < 50 and abs(dy) < 100:
              self.danger = True
              
              # 边界判定逻辑
              player_x = self.player_pos[0]
              if player_x < 220:
                  # 靠近左边界，强行右移躲避
                  return ACTION_RIGHT
              elif player_x > 650:
                  # 靠近右边界，强行左移躲避
                  return ACTION_LEFT
              else:
                  # 增强躲避逻辑：向离子弹更远的方向移动
                  if dx > 0:
                    return ACTION_LEFT
                  else:
                     return ACTION_RIGHT
    self.danger = False
    self.predicted_bonus_pos = None
    
  def update_bullets(self,pos, frame):
    if pos == None:
      for idx,bullet in enumerate(self.bullets):
        self.bullets[idx][3] -=1
      for idx,bullet in enumerate(self.bullets):
       if bullet[1]>600 or bullet[3]<=0:
          self.bullets.pop(idx)
      return
    exist = False
    for idx,bullet in enumerate(self.bullets):
        if abs(bullet[0]-pos[0])<20 and abs(bullet[1]-pos[1])<20 :
          exist = True
          if pos[1]>bullet[1]:
            self.bullets[idx][2] = 1
            self.bullets[idx][3] =25
          else:
            self.bullets[idx][2] = -1
            self.bullets[idx][3] =25
        self.bullets[idx][3] -=1
    if not exist:
      self.bullets.append([pos[0],pos[1],0,20])
      return
    for idx,bullet in enumerate(self.bullets):
       if bullet[1]>580 or bullet[3]<=0:
          self.bullets.pop(idx)
    # print(frame,self.bullets)

