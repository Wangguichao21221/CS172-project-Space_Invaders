import numpy as np

ACTION_NOOP = 0      # 不动
ACTION_FIRE = 1      # 发射
ACTION_RIGHT = 2     # 向右
ACTION_LEFT = 3      # 向左
ACTION_RIGHTFIRE = 4 # 向右并发射
ACTION_LEFTFIRE = 5  # 向左并发射

class algorithm():
  def __init__(self):
    self.player_pos = (0,0)
    self.enemy_bullet_speed = 3.4
    self.player_bullet_speed =5.5
    self.player_speed = 2.6
    self.enemy_speed = 3/32
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
    self.shelter_mask = None # 用于存储掩体的像素状态
    self.bullet_width = 4    # 默认子弹宽度
    self.roi_x_offset = 211  # ROI 在原图中的 X 偏移
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
  def update_enemy_speed(self,frame):
      if frame<980:
        self.enemy_speed = 0.16
        return
      if frame<1500:
        self.enemy_speed = -0.27
        return
      if frame<1800:
        self.enemy_speed = 0.46
        return
      if frame<2100:
        self.enemy_speed = -0.46
        return
      if frame<2400:
        self.enemy_speed = 0.46
        return
      else:
          if ((frame-2400)//150) %2 == 0:
            self.enemy_speed = 0.93
          else :
            self.enemy_speed = -0.93
  def step(self):
      if self.positions == None:
         return ACTION_FIRE
      action = self.avoid_bullet()
      if not self.danger:
        self.get_target()
        if not self.bonus_exist:
          action = self.move_shoot()
        else:
           action = self.shoot_bonus()
      return action
  def analyse(self,results,frame_index, shelter_mask=None):
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
    
    # 将每个类别的坐标按 x 轴从左到右排序
    for cls_name in class_positions:
        class_positions[cls_name].sort(key=lambda p: p[0])

    self.positions = class_positions.copy()
    self.update_enemy_speed(frame_index)
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
    # 如果敌人总数小于10，无需考虑掩体限制，自由开火
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

  def get_target(self):
    # 找到最后两排存在的敌人
    existing_rows = []
    for i in range(6, 0, -1):
        if f"invader{i}" in self.positions:
            existing_rows.append(i)
            if len(existing_rows) == 2:
                break
    
    if not existing_rows:
        # print("no invaders found!")
        return

    # 合并这两排的所有敌人位置
    pos = []
    for r in existing_rows:
        pos.extend(self.positions[f"invader{r}"])
    
    # 优先选择：1. 在范围内 2. 且前方路径畅通的敌人
    in_range = [p for p in pos if 196 <= p[0] <= 620]
    
    # 在范围内挑选路径不被遮挡的敌人
    clear_targets = [p for p in in_range if self.is_path_clear(p[0])]
    
    if clear_targets:
        # 挑选距离玩家最近的通畅目标
        left_invader = min(clear_targets, key=lambda x: abs(x[0] - self.player_pos[0]))
    elif in_range:
        # 如果范围内全被遮挡，退而求其次选范围内最近的
        left_invader = min(in_range, key=lambda x: abs(x[0] - self.player_pos[0]))
    else:
        # 兜底逻辑：选范围外的，继续往上面排找
        n = existing_rows[-1] - 1
        while n >= 1:
            if f"invader{n}" in self.positions:
                pos = self.positions[f"invader{n}"]
                left_invader = min(pos, key=lambda x: abs(x[0] - self.player_pos[0]))
                break
            n -= 1
        else:
            return
    
    self.target = left_invader

  def move_shoot(self):
    if self.target == None:
       return ACTION_NOOP
    target_x = self.target[0]
    target_y = self.target[1]
    dy = self.player_pos[1]-target_y
    t = dy/self.player_bullet_speed
    shoot_x = target_x+t*self.enemy_speed
    dx = shoot_x-self.player_pos[0]
    
    # 核心判断：如果目标路径被掩体挡住了，我们只移动到目标下方，但不触发 FIRE
    path_clear = self.is_path_clear(self.player_pos[0])

    if -3<dx<3:
      if path_clear:
        return ACTION_FIRE
      else:
        # 路径被挡，可以考虑在这里加入“微调寻找缝隙”的逻辑
        # 目前先保持不动，等待敌人移动出掩体或掩体消失
        return ACTION_NOOP
    elif dx<-3:
       return ACTION_LEFT
    else:
       return ACTION_RIGHT
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
          
          # y 范围增加到 100 像素，x 范围设定为覆盖玩家身位（约 45 像素）
          if abs(dx) < 60 and abs(dy) < 100:
              self.danger = True
              
              # 边界判定逻辑
              player_x = self.player_pos[0]
              if player_x < 230:
                  # 靠近左边界，强行右移躲避
                  return ACTION_RIGHT
              elif player_x > 640:
                  # 靠近右边界，强行左移躲避
                  return ACTION_LEFT
              else:
                  # 正常躲避：先躲开区域，不带开火
                  if bullet[0] > self.player_pos[0]:
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

