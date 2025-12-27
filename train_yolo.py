
from ultralytics import YOLO

model = YOLO('yolov8n.pt') 
results = model.train(data="./invader_data/data.yaml", workers =0, epochs = 200, batch = 16,
                      perspective=0.0,     # 关闭透视变换（像素游戏是2D的）
                      shear=0.0,           # 关闭剪切变换
                      hsv_h=0.3,           # 中等色调变化
                      hsv_s=0.7,           # 较大饱和度变化（像素游戏颜色重要）
                      hsv_v=0.4,           # 亮度变化
                      degrees=5.0,         # 小角度旋转（像素游戏通常轴对齐）
                      translate=0.1,       # 小范围平移
                      scale=0.5,           # 缩放增强
                      )
