import cv2
import mediapipe as mp
import pygame
import random
import math
import numpy as np
import time
import json
import os
import sys
import pickle
from datetime import datetime
from collections import deque
from keras.models import load_model
import requests
# -*- coding: utf-8 -*-
# -------------------------- 获取资源路径（支持打包后的运行）--------------------------
def resource_path(relative_path):
    """获取资源的绝对路径，支持PyInstaller打包"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def app_base_path():
    """获取程序运行目录（打包后为可执行文件所在目录）"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def get_training_data_dir():
    """获取训练数据目录并确保存在"""
    data_dir = os.path.join(app_base_path(), "training_data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

# -------------------------- Initialization --------------------------
pygame.init()
display_info = pygame.display.Info()
LOGICAL_WIDTH, LOGICAL_HEIGHT = 1000, 700
DISPLAY_WIDTH, DISPLAY_HEIGHT = display_info.current_w, display_info.current_h
WIDTH, HEIGHT = LOGICAL_WIDTH, LOGICAL_HEIGHT
screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.FULLSCREEN)
render_surface = pygame.Surface((WIDTH, HEIGHT))
pygame.display.set_caption("认知训练游戏 - 三合一")
clock = pygame.time.Clock()
FPS = 30
url="http://129.28.37.144:8080/upload"

def get_font(size):
    """获取字体，尝试加载更美观的字体"""
    # Prefer a bundled CJK-capable font to avoid missing glyphs on new machines.
    bundled_font = os.path.join(app_base_path(), "fonts", "wqy-zenhei.ttc")
    system_fallbacks = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for font_path in [bundled_font] + system_fallbacks:
        if os.path.exists(font_path):
            try:
                return pygame.font.Font(font_path, size)
            except Exception:
                pass
    try:
        font = pygame.font.SysFont("WenQuanYiZen Hei", size)
        test_surface = font.render("测试", True, (255, 255, 255))
        if test_surface.get_width() > 0:
            return font
    except Exception:
        pass
    return pygame.font.Font(None, size)

# 漂亮的配色方案
COLORS = {
    'background': (20, 30, 40),
    'text': (255, 255, 255),
    'text_bg': (0, 0, 0, 150),
    'menu_card': (40, 55, 70),
    'menu_card_hover': (60, 85, 110),
    'reaction': (255, 100, 100),
    'memory': (100, 200, 255),
    'judgment': (150, 100, 255),
    'judgment_block': (0, 150, 255),
    'success': (100, 255, 100),
    'warning': (255, 100, 100),
    'info': (100, 200, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'left': (255, 165, 0),
    'right': (0, 200, 255),
    'number': (255, 255, 100),
}

# 彩色手部骨架颜色
HAND_COLORS = {
    'wrist': (200, 200, 200),
    'thumb': (255, 100, 100),
    'index': (255, 165, 0),
    'middle': (255, 255, 100),
    'ring': (100, 255, 100),
    'pinky': (100, 200, 255),
}

# MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Camera
cap = cv2.VideoCapture(0)
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {actual_width}x{actual_height}")

# -------------------------- DNN Left/Right Gesture Recognizer --------------------------
class DNNLeftRightRecognizer:
    def __init__(self, model_path=None, 
                 scaler_path=None, 
                 label_encoder_path=None,
                 confidence_threshold=0.85):
        """
        初始化DNN左右手势识别器
        """
        print("加载DNN左右手势识别模型...")
        try:
            model_path = model_path or resource_path("dnn_gesture_model.h5")
            scaler_path = scaler_path or resource_path("dnn_scaler.pkl")
            label_encoder_path = label_encoder_path or resource_path("dnn_label_encoder.pkl")

            self.model = load_model(model_path)
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            with open(label_encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            
            self.confidence_threshold = confidence_threshold
            print("DNN左右手势识别器初始化完成")
            print(f"支持的手势: {list(self.label_encoder.classes_)}")
            self.model_loaded = True
        except Exception as e:
            print(f"警告：无法加载DNN模型，将使用传统方法识别左右手势: {e}")
            self.model_loaded = False
    
    def extract_landmarks(self, frame):
        """
        从帧中提取手部关键点
        返回: (处理后的帧, 关键点数组, 是否检测到手部)
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        landmarks_2d = []
        hand_detected = False
        
        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                
                for lm in hand_landmarks.landmark:
                    landmarks_2d.extend([lm.x, lm.y])
        
        return frame, landmarks_2d, hand_detected
    
    def calculate_hand_orientation(self, landmarks_2d):
        """
        计算手掌方向：水平或垂直
        返回: 'horizontal', 'vertical', 'diagonal', 'unknown'
        """
        if len(landmarks_2d) != 42:
            return "unknown"
        
        points = np.array(landmarks_2d).reshape(21, 2)
        
        WRIST = 0
        MIDDLE_FINGER_MCP = 9
        
        palm_vector = points[MIDDLE_FINGER_MCP] - points[WRIST]
        palm_angle = np.degrees(np.arctan2(palm_vector[1], palm_vector[0]))
        palm_angle = palm_angle % 360
        
        is_horizontal = (abs(palm_angle) <= 30 or abs(palm_angle - 180) <= 30 or 
                        abs(palm_angle - 360) <= 30)
        is_vertical = (abs(palm_angle - 90) <= 30 or abs(palm_angle - 270) <= 30)
        
        if is_horizontal:
            return "horizontal"
        elif is_vertical:
            return "vertical"
        else:
            return "diagonal"
    
    def predict(self, landmarks_2d, hand_orientation=None):
        """
        预测左右手势
        返回: (手势名称, 置信度)
        """
        if not self.model_loaded:
            return None, 0.0
        
        if len(landmarks_2d) != 42:
            return None, 0.0
        
        if hand_orientation == "vertical":
            return None, 0.0
        
        try:
            landmarks_array = np.array(landmarks_2d).reshape(1, -1)
            landmarks_scaled = self.scaler.transform(landmarks_array)
            
            predictions = self.model.predict(landmarks_scaled, verbose=0)[0]
            pred_idx = np.argmax(predictions)
            confidence = predictions[pred_idx]
            
            if confidence > self.confidence_threshold:
                gesture = self.label_encoder.inverse_transform([pred_idx])[0]
                if gesture in ['left', 'right']:
                    return gesture, confidence
        except Exception as e:
            print(f"预测错误: {e}")
        
        return None, 0.0
    
    def predict_from_frame(self, frame, check_orientation=True):
        """
        直接从摄像头帧进行预测
        返回: (处理后的帧, 手势名称, 置信度, 手掌方向)
        """
        frame, landmarks_2d, hand_detected = self.extract_landmarks(frame)
        
        gesture = None
        confidence = 0.0
        hand_orientation = "unknown"
        
        if hand_detected and len(landmarks_2d) == 42:
            hand_orientation = self.calculate_hand_orientation(landmarks_2d)
            
            if check_orientation:
                gesture, confidence = self.predict(landmarks_2d, hand_orientation)
            else:
                gesture, confidence = self.predict(landmarks_2d)
        
        return frame, gesture, confidence, hand_orientation
    
    def close(self):
        """释放资源"""
        pass

# -------------------------- Traditional Gesture Recognition Functions --------------------------
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
FINGER_MCPS = [2, 5, 9, 13, 17]
THUMB_IP = 3
THUMB_MCP = 2
THUMB_TIP = 4

def is_finger_extended(hand_landmarks, finger_idx, hand_side):
    """判断手指是否伸直"""
    tip_idx = FINGER_TIPS[finger_idx]
    pip_idx = FINGER_PIPS[finger_idx]
    mcp_idx = FINGER_MCPS[finger_idx]
    
    tip = hand_landmarks.landmark[tip_idx]
    pip = hand_landmarks.landmark[pip_idx]
    mcp = hand_landmarks.landmark[mcp_idx]
    
    if finger_idx == 0:  # 拇指
        thumb_ip = hand_landmarks.landmark[THUMB_IP]
        
        if hand_side == 'Right':
            is_straight = (tip.x < thumb_ip.x - 0.02 and thumb_ip.x < mcp.x - 0.02)
            return is_straight
        else:
            is_straight = (tip.x > thumb_ip.x + 0.02 and thumb_ip.x > mcp.x + 0.02)
            return is_straight
    else:
        return tip.y < pip.y - 0.01

def get_hand_side(hand_landmarks):
    """判断是左手还是右手"""
    thumb_mcp = hand_landmarks.landmark[1]
    index_mcp = hand_landmarks.landmark[5]
    
    if thumb_mcp.x < index_mcp.x:
        return 'Right'
    else:
        return 'Left'

def count_extended_fingers(hand_landmarks, hand_side):
    """统计伸直的手指数量"""
    extended_count = 0
    finger_states = []
    
    for i in range(5):
        extended = is_finger_extended(hand_landmarks, i, hand_side)
        finger_states.append(extended)
        if extended:
            extended_count += 1
    
    return extended_count, finger_states

def recognize_number_gesture(hand_landmarks):
    """识别数字手势 0-5"""
    hand_side = get_hand_side(hand_landmarks)
    extended_count, finger_states = count_extended_fingers(hand_landmarks, hand_side)
    
    if extended_count == 0:
        return 0
    
    elif extended_count == 1 and finger_states[1] and not any(finger_states[i] for i in [0,2,3,4]):
        return 1
    
    elif extended_count == 2 and finger_states[1] and finger_states[2] and not any(finger_states[i] for i in [0,3,4]):
        return 2
    
    elif extended_count == 3 and finger_states[1] and finger_states[2] and finger_states[3] and not any(finger_states[i] for i in [0,4]):
        return 3
    
    elif extended_count == 4:
        if (finger_states[1] and finger_states[2] and finger_states[3] and finger_states[4] and 
            not finger_states[0]):
            thumb_tip = hand_landmarks.landmark[THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[THUMB_IP]
            
            if hand_side == 'Right':
                if thumb_tip.x > thumb_ip.x:
                    return 4
            else:
                if thumb_tip.x < thumb_ip.x:
                    return 4
    
    elif extended_count == 5:
        if all(finger_states):
            thumb_tip = hand_landmarks.landmark[THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[THUMB_IP]
            thumb_mcp = hand_landmarks.landmark[THUMB_MCP]
            
            if hand_side == 'Right':
                if thumb_tip.x < thumb_ip.x and thumb_ip.x < thumb_mcp.x:
                    return 5
            else:
                if thumb_tip.x > thumb_ip.x and thumb_ip.x > thumb_mcp.x:
                    return 5
    
    return -1

def recognize_traditional_left_right_gesture(hand_landmarks):
    """传统方法识别向左/向右手势（备用）"""
    hand_side = get_hand_side(hand_landmarks)
    extended_count, finger_states = count_extended_fingers(hand_landmarks, hand_side)
    if not (finger_states[1] and finger_states[2] and not any(finger_states[i] for i in [0,3,4])):
        return None

    wrist = hand_landmarks.landmark[0]
    middle_tip = hand_landmarks.landmark[12]
    dx = middle_tip.x - wrist.x
    dy = middle_tip.y - wrist.y
    angle = abs(math.degrees(math.atan2(dy, dx)))
    if angle < 30:
        if dx > 0:
            return 'right'
        elif dx < 0:
            return 'left'
    return None

def recognize_gesture(hand_landmarks, dnn_recognizer):
    """综合手势识别：先判断手掌方向，再决定使用DNN还是传统方法"""
    # 提取关键点数组用于计算手掌方向
    landmarks_2d = []
    for lm in hand_landmarks.landmark:
        landmarks_2d.extend([lm.x, lm.y])
    
    # 计算手掌方向
    hand_orientation = dnn_recognizer.calculate_hand_orientation(landmarks_2d)
    
    # 如果是水平方向，优先使用DNN识别左右手势
    if hand_orientation == "horizontal":
        if dnn_recognizer.model_loaded:
            gesture, confidence = dnn_recognizer.predict(landmarks_2d, hand_orientation)
            if gesture:
                return gesture
    
    # 否则使用传统方法识别数字手势
    num = recognize_number_gesture(hand_landmarks)
    if num != -1:
        return num
    
    return None

# -------------------------- Data Tracking --------------------------
class DataTracker:
    def __init__(self, user_id="user"):
        self.user_id = user_id
        self.data_dir = get_training_data_dir()
        self.session_start = time.time()
        self.session_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'duration': 0,
            'training_mode': 'reaction',
            'total_score': 0,
            'hits': 0,
            'misses': 0,
            'accuracy': 0,
            'max_combo': 0,
            'memory_rounds_completed': 0,
            'memory_accuracy': 0,
            'judgment_total': 0,
            'judgment_correct': 0,
            'performance_history': []
        }
        self.memory_performance = []
        self.judgment_results = []
    
    def record_hit(self):
        self.session_data['hits'] += 1
        
    def record_miss(self):
        self.session_data['misses'] += 1
        
    def record_memory_round(self, correct_count, total_count):
        accuracy = correct_count / total_count if total_count > 0 else 0
        self.memory_performance.append(accuracy)
        self.session_data['memory_rounds_completed'] += 1
        self.session_data['memory_accuracy'] = sum(self.memory_performance) / len(self.memory_performance)
    
    def record_judgment_result(self, correct):
        self.judgment_results.append(correct)
        if correct:
            self.session_data['judgment_correct'] += 1
        self.session_data['judgment_total'] += 1
        
    def update_score(self, score, combo):
        self.session_data['total_score'] = score
        if combo > self.session_data['max_combo']:
            self.session_data['max_combo'] = combo
            
    def update_mode(self, mode):
        self.session_data['training_mode'] = mode
        
    def calculate_metrics(self):
        total_attempts = self.session_data['hits'] + self.session_data['misses']
        if total_attempts > 0:
            self.session_data['accuracy'] = self.session_data['hits'] / total_attempts
        self.session_data['duration'] = time.time() - self.session_start
        
    def generate_report(self):
        self.calculate_metrics()
        report = f"""
╔════════════════════════════════════════════════╗
║          认知训练报告                            ║
╠════════════════════════════════════════════════╣
║ 用户: {self.user_id:<30} ║
║ 时间: {self.session_data['timestamp']} ║
║ 时长: {self.session_data['duration']:.1f}秒{''*24} ║
║ 模式: {self.session_data['training_mode']:<30} ║
║ 得分: {self.session_data['total_score']:<30} ║
║ 击中: {self.session_data['hits']:<30} ║
║ 错过: {self.session_data['misses']:<30} ║
║ 准确率: {self.session_data['accuracy']*100:.1f}%{''*26} ║
║ 最大连击: {self.session_data['max_combo']:<27} ║
"""
        if self.session_data['training_mode'] == 'memory':
            report += f"""║ 记忆轮次: {self.session_data['memory_rounds_completed']:<28} ║
║ 记忆准确率: {self.session_data['memory_accuracy']*100:.1f}%{''*25} ║
"""
        elif self.session_data['training_mode'] == 'judgment':
            total = self.session_data['judgment_total']
            correct = self.session_data['judgment_correct']
            acc = correct/total*100 if total>0 else 0
            report += f"""║ 判断次数: {total:<30} ║
║ 正确次数: {correct:<30} ║
║ 判断准确率: {acc:.1f}%{''*26} ║
"""
        report += f"""║                                          ║
║ 建议:                                     ║
{self._generate_suggestions()}
╚════════════════════════════════════════════════╝
"""
        return report
    
    def _generate_suggestions(self):
        suggestions = []
        if self.session_data['accuracy'] < 0.5:
            suggestions.append("║  准确率较低，建议放慢速度练习                    ║")
        elif self.session_data['accuracy'] > 0.9:
            suggestions.append("║  表现很好！可以尝试更高难度                      ║")
        if self.session_data['training_mode'] == 'memory' and self.session_data['memory_accuracy'] < 0.6:
            suggestions.append("║  记忆准确率需要提高，多练习记忆模式                ║")
        if self.session_data['training_mode'] == 'judgment':
            total = self.session_data['judgment_total']
            correct = self.session_data['judgment_correct']
            if total > 0 and correct/total < 0.6:
                suggestions.append("║  判断准确率较低，多练习手势识别                  ║")
        return "\n".join(suggestions) if suggestions else "║  继续保持练习！                                 ║"
    
    def save_session(self):
        filename = os.path.join(self.data_dir, f"{self.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.session_data, f, ensure_ascii=False, indent=2)
        try:
            with open(filename, 'rb') as upload_file:
                files = {'file': upload_file}
                requests.post(url, files=files, timeout=5)
        except Exception as e:
            print(f"上传失败（已忽略）：{e}")
        return filename
    
    def show_report_screen(self, screen):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        screen.blit(overlay, (0, 0))
        
        report_width = 600
        report_height = 500
        report_x = (WIDTH - report_width) // 2
        report_y = (HEIGHT - report_height) // 2
        
        pygame.draw.rect(screen, (30, 40, 50), (report_x, report_y, report_width, report_height), border_radius=20)
        pygame.draw.rect(screen, COLORS['white'], (report_x, report_y, report_width, report_height), 3, border_radius=20)
        
        title_font = get_font(36)
        title = title_font.render("训练报告", True, COLORS['white'])
        title_rect = title.get_rect(center=(WIDTH//2, report_y + 40))
        screen.blit(title, title_rect)
        
        report_font = get_font(20)
        lines = [
            f"时长: {self.session_data['duration']:.1f} 秒",
            f"模式: {self.session_data['training_mode']}",
            f"得分: {self.session_data['total_score']}",
            f"准确率: {self.session_data['accuracy']*100:.1f}%",
            f"最大连击: {self.session_data['max_combo']}"
        ]
        if self.session_data['training_mode'] == 'memory':
            lines.append(f"记忆准确率: {self.session_data['memory_accuracy']*100:.1f}%")
        elif self.session_data['training_mode'] == 'judgment':
            total = self.session_data['judgment_total']
            correct = self.session_data['judgment_correct']
            acc = correct/total*100 if total>0 else 0
            lines.append(f"判断准确率: {acc:.1f}%")
        
        y = report_y + 100
        for line in lines:
            text = report_font.render(line, True, COLORS['white'])
            text_rect = text.get_rect(center=(WIDTH//2, y))
            screen.blit(text, text_rect)
            y += 35
        
        suggest_title = report_font.render("建议:", True, COLORS['success'])
        suggest_rect = suggest_title.get_rect(center=(WIDTH//2, y + 20))
        screen.blit(suggest_title, suggest_rect)
        y += 60
        
        suggestions = self._generate_suggestions().split('\n')
        for s in suggestions:
            if s.strip():
                clean_s = s.replace('║', '').strip()
                text = report_font.render(clean_s, True, COLORS['info'])
                text_rect = text.get_rect(center=(WIDTH//2, y))
                screen.blit(text, text_rect)
                y += 30
        
        hint = report_font.render("按任意键继续", True, COLORS['white'])
        hint_rect = hint.get_rect(center=(WIDTH//2, report_y + report_height - 40))
        screen.blit(hint, hint_rect)

# -------------------------- Target Class --------------------------
class Target:
    def __init__(self, color=None, x_range=None, y_range=None):
        self.radius = 50
        if x_range is None:
            x_range = (self.radius, WIDTH - self.radius)
        if y_range is None:
            y_range = (self.radius, HEIGHT - self.radius)
        self.x = random.randint(int(x_range[0]), int(x_range[1]))
        self.y = random.randint(int(y_range[0]), int(y_range[1]))
        self.initial_radius = self.radius
        self.current_radius = self.initial_radius
        self.active = True
        self.spawn_time = time.time()
        self.lifetime = 3.0
        if color is None:
            self.color = random.choice([COLORS['reaction'], COLORS['memory'], 
                                        COLORS['judgment'], COLORS['success'], COLORS['info']])
        else:
            self.color = color
        self.index = 0
        
    def update(self):
        if not self.active:
            return None
        elapsed = time.time() - self.spawn_time
        if elapsed >= self.lifetime:
            self.active = False
            return "timeout"
        progress = elapsed / self.lifetime
        self.current_radius = int(self.initial_radius * (1 - progress * 0.7))
        if self.current_radius < 1:
            self.active = False
            return "timeout"
        return None
    
    def draw(self, surface, alpha=255, show_number=False, number=None):
        if not self.active or self.current_radius <= 0:
            return
        glow_radius = self.current_radius + 8
        glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.color[:3], 40), (glow_radius, glow_radius), glow_radius)
        surface.blit(glow_surf, (self.x - glow_radius, self.y - glow_radius))
        
        if alpha < 255:
            circle_surf = pygame.Surface((self.current_radius*2, self.current_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(circle_surf, (*self.color[:3], alpha), (self.current_radius, self.current_radius), self.current_radius)
            pygame.draw.circle(circle_surf, (*COLORS['white'][:3], alpha), (self.current_radius, self.current_radius), self.current_radius, 4)
            surface.blit(circle_surf, (self.x - self.current_radius, self.y - self.current_radius))
        else:
            pygame.draw.circle(surface, self.color, (self.x, self.y), self.current_radius)
            pygame.draw.circle(surface, COLORS['white'], (self.x, self.y), self.current_radius, 4)
        
        if show_number and number is not None:
            font = get_font(36)
            num_text = font.render(str(number), True, COLORS['white'])
            num_rect = num_text.get_rect(center=(self.x, self.y))
            bg_rect = num_rect.inflate(10,5)
            pygame.draw.rect(surface, COLORS['text_bg'], bg_rect)
            pygame.draw.rect(surface, COLORS['white'], bg_rect, 1)
            surface.blit(num_text, num_rect)
    
    def check_hit(self, finger_x, finger_y):
        if not self.active or self.current_radius <= 0:
            return False
        dist = math.hypot(finger_x - self.x, finger_y - self.y)
        return dist <= self.current_radius

# -------------------------- Memory Training --------------------------
class MemoryTraining:
    def __init__(self):
        self.sequence = []
        self.clicked_indices = []
        self.memory_phase = True
        self.memory_start_time = 0
        self.memory_duration = 3
        self.round_size = 3
        self.active = False
        self.feedback_text = ""
        self.feedback_timer = 0
        self.round_color = COLORS['memory']
        
    def start_new_round(self):
        self.sequence = []
        self.clicked_indices = []
        self.memory_phase = True
        self.memory_start_time = time.time()
        self.active = True
        min_distance = 2 * 50
        
        center_x, center_y = WIDTH/2, HEIGHT/2
        area_width = WIDTH * 0.8
        area_height = HEIGHT * 0.8
        half_w = area_width / 2
        half_h = area_height / 2
        x_min = max(center_x - half_w + 50, 50)
        x_max = min(center_x + half_w - 50, WIDTH - 50)
        y_min = max(center_y - half_h + 50, 50)
        y_max = min(center_y + half_h - 50, HEIGHT - 50)
        
        max_attempts = 1000
        for i in range(self.round_size):
            attempts = 0
            while attempts < max_attempts:
                new_target = Target(color=self.round_color, x_range=(x_min, x_max), y_range=(y_min, y_max))
                new_target.index = i
                overlap = False
                for t in self.sequence:
                    dist = math.hypot(new_target.x - t.x, new_target.y - t.y)
                    if dist < min_distance:
                        overlap = True
                        break
                if not overlap:
                    self.sequence.append(new_target)
                    break
                attempts += 1
            else:
                new_target = Target(color=self.round_color, x_range=(x_min, x_max), y_range=(y_min, y_max))
                new_target.index = i
                self.sequence.append(new_target)
        
    def update(self):
        if not self.active:
            return None
        if self.memory_phase:
            elapsed = time.time() - self.memory_start_time
            if elapsed > self.memory_duration:
                self.memory_phase = False
                return "recall_start"
        return None
    
    def handle_click(self, pos):
        if self.memory_phase or not self.active:
            return None
        clicked_target = None
        for t in self.sequence:
            if t.active and t.check_hit(pos[0], pos[1]):
                clicked_target = t
                break
        if clicked_target is None:
            return None
        next_index = len(self.clicked_indices)
        if clicked_target.index == next_index:
            self.clicked_indices.append(clicked_target.index)
            if len(self.clicked_indices) == len(self.sequence):
                self.active = False
                return "round_complete"
            return "correct"
        elif clicked_target.index < next_index:
            return None
        else:
            self.feedback_text = "wrong"
            self.feedback_timer = 20
            return "wrong"
    
    def draw(self, screen):
        if not self.active:
            return
        for t in self.sequence:
            if t.active:
                if self.memory_phase:
                    t.draw(screen, alpha=255, show_number=True, number=t.index+1)
                else:
                    if t.index in self.clicked_indices:
                        t.draw(screen, alpha=255, show_number=False)
                    else:
                        t.draw(screen, alpha=100, show_number=False)
        if self.memory_phase:
            elapsed = time.time() - self.memory_start_time
            remaining = max(0, self.memory_duration - elapsed)
            font = get_font(48)
            text = font.render(f"记忆时间: {remaining:.1f}s", True, COLORS['white'])
            rect = text.get_rect(center=(WIDTH//2, 100))
            bg_rect = rect.inflate(20,10)
            pygame.draw.rect(screen, COLORS['text_bg'], bg_rect)
            pygame.draw.rect(screen, COLORS['white'], bg_rect, 2)
            screen.blit(text, rect)
        if self.feedback_timer > 0 and self.feedback_text == "wrong":
            font = get_font(48)
            text = font.render("错误!", True, COLORS['warning'])
            rect = text.get_rect(center=(WIDTH//2, HEIGHT-100))
            bg_rect = rect.inflate(20,10)
            pygame.draw.rect(screen, COLORS['text_bg'], bg_rect)
            pygame.draw.rect(screen, COLORS['white'], bg_rect, 2)
            screen.blit(text, rect)
            self.feedback_timer -= 1
        if not self.memory_phase:
            font = get_font(36)
            prog = f"进度: {len(self.clicked_indices)}/{len(self.sequence)}"
            text = font.render(prog, True, COLORS['white'])
            rect = text.get_rect(center=(WIDTH//2, 150))
            bg_rect = rect.inflate(20,10)
            pygame.draw.rect(screen, COLORS['text_bg'], bg_rect)
            pygame.draw.rect(screen, COLORS['white'], bg_rect, 2)
            screen.blit(text, rect)

# -------------------------- Judgment Block --------------------------
class JudgmentBlock:
    def __init__(self, action):
        self.action = action
        self.width = 120
        self.height = 80
        self.x = random.randint(self.width, WIDTH - self.width)
        self.y = -self.height
        self.speed = 3.0
        self.active = True
        self.judged = False
        
        if action == 'left':
            self.color = COLORS['left']
            self.display_text = '←'
        elif action == 'right':
            self.color = COLORS['right']
            self.display_text = '→'
        else:
            self.color = COLORS['judgment_block']
            self.display_text = str(action)
            
    def update(self):
        if not self.active:
            return None
        self.y += self.speed
        
        if self.y > HEIGHT + self.height:
            self.active = False
            return "miss"
        return None
    
    def draw(self, surface):
        if not self.active:
            return
        
        rect = pygame.Rect(self.x - self.width//2, self.y - self.height//2, self.width, self.height)
        
        glow_rect = rect.inflate(10, 10)
        glow_surf = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.color[:3], 40), 
                        (0, 0, glow_rect.width, glow_rect.height), border_radius=15)
        surface.blit(glow_surf, (glow_rect.x, glow_rect.y))
        
        pygame.draw.rect(surface, self.color, rect, border_radius=10)
        pygame.draw.rect(surface, COLORS['white'], rect, 3, border_radius=10)
        
        font = get_font(48)
        text = font.render(self.display_text, True, COLORS['white'])
        text_rect = text.get_rect(center=rect.center)
        surface.blit(text, text_rect)
    
    def check_judgment(self, left_gesture, right_gesture):
        if self.judged or not self.active:
            return None
        
        judgment_line_y = HEIGHT * 0.7
        if abs(self.y - judgment_line_y) < 20:
            self.judged = True
            
            correct = False
            if left_gesture == self.action:
                correct = True
            if right_gesture == self.action:
                correct = True
            
            self.active = False
            
            if correct:
                return "correct"
            else:
                return "wrong"
        return None

# -------------------------- Judgment Training --------------------------
class JudgmentTraining:
    def __init__(self):
        self.blocks = []
        self.max_blocks = 3
        self.last_spawn_time = 0
        self.spawn_interval = 1.0
        self.total_actions = 20
        self.actions_remaining = []
        self.active = False
        self.completed = False
        self.score = 0
        self.correct_count = 0
        self.possible_actions = ['left', 'right', 0, 1, 2, 3, 4, 5]
        
        self.left_gesture = None
        self.right_gesture = None
        self.left_gesture_history = deque(maxlen=5)
        self.right_gesture_history = deque(maxlen=5)
        self.left_stable = None
        self.right_stable = None
        
        self.feedback_text = ""
        self.feedback_timer = 0
        
    def start_new_session(self):
        self.actions_remaining = [random.choice(self.possible_actions) for _ in range(self.total_actions)]
        self.blocks = []
        self.last_spawn_time = time.time()
        self.active = True
        self.completed = False
        self.score = 0
        self.correct_count = 0
        self.feedback_text = ""
        self.feedback_timer = 0
        self.left_gesture_history.clear()
        self.right_gesture_history.clear()
        self.left_stable = None
        self.right_stable = None
        print("判断模式开始:", self.actions_remaining)
        
    def update_gesture(self, left_gesture, right_gesture):
        if left_gesture is not None:
            self.left_gesture_history.append(left_gesture)
        if right_gesture is not None:
            self.right_gesture_history.append(right_gesture)
        
        if len(self.left_gesture_history) > 0:
            counter = {}
            for g in self.left_gesture_history:
                if g is not None:
                    counter[g] = counter.get(g, 0) + 1
            if counter:
                self.left_stable = max(counter.items(), key=lambda x: x[1])[0]
            else:
                self.left_stable = None
        else:
            self.left_stable = None
        
        if len(self.right_gesture_history) > 0:
            counter = {}
            for g in self.right_gesture_history:
                if g is not None:
                    counter[g] = counter.get(g, 0) + 1
            if counter:
                self.right_stable = max(counter.items(), key=lambda x: x[1])[0]
            else:
                self.right_stable = None
        else:
            self.right_stable = None
        
    def update(self):
        if not self.active or self.completed:
            return
        
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
        else:
            self.feedback_text = ""
        
        current_time = time.time()
        if (len(self.actions_remaining) > 0 and 
            len(self.blocks) < self.max_blocks and 
            current_time - self.last_spawn_time >= self.spawn_interval):
            action = self.actions_remaining.pop(0)
            new_block = JudgmentBlock(action)
            self.blocks.append(new_block)
            self.last_spawn_time = current_time
        
        for block in self.blocks[:]:
            result = block.update()
            if result == "miss":
                self.blocks.remove(block)
                self.score = max(0, self.score - 50)
                self.feedback_text = "错过!"
                self.feedback_timer = 30
            
            judgment_result = block.check_judgment(self.left_stable, self.right_stable)
            if judgment_result == "correct":
                self.blocks.remove(block)
                self.correct_count += 1
                self.score += 100
                self.feedback_text = "正确!"
                self.feedback_timer = 30
            elif judgment_result == "wrong":
                self.blocks.remove(block)
                self.score = max(0, self.score - 50)
                self.feedback_text = "错误!"
                self.feedback_timer = 30
        
        if len(self.actions_remaining) == 0 and len(self.blocks) == 0:
            self.completed = True
            self.active = False
    
    def draw(self, screen):
        if not self.active and not self.completed:
            return
        
        judgment_line_y = HEIGHT * 0.7
        pygame.draw.line(screen, COLORS['white'], (0, judgment_line_y), (WIDTH, judgment_line_y), 3)
        
        for block in self.blocks:
            block.draw(screen)
        
        font_info = get_font(24)
        remaining_text = font_info.render(f"剩余: {len(self.actions_remaining)}", True, COLORS['white'])
        remaining_rect = remaining_text.get_rect(topleft=(20, 20))
        bg_rect = remaining_rect.inflate(10,5)
        pygame.draw.rect(screen, COLORS['text_bg'], bg_rect)
        pygame.draw.rect(screen, COLORS['white'], bg_rect, 1)
        screen.blit(remaining_text, remaining_rect)
        
        if self.left_stable is not None:
            left_display = str(self.left_stable) if isinstance(self.left_stable, int) else self.left_stable.capitalize()
            left_text = font_info.render(f"左手: {left_display}", True, COLORS['success'])
            left_rect = left_text.get_rect(topleft=(20, 60))
            bg_left = left_rect.inflate(10,5)
            pygame.draw.rect(screen, COLORS['text_bg'], bg_left)
            pygame.draw.rect(screen, COLORS['white'], bg_left, 1)
            screen.blit(left_text, left_rect)
        
        if self.right_stable is not None:
            right_display = str(self.right_stable) if isinstance(self.right_stable, int) else self.right_stable.capitalize()
            right_text = font_info.render(f"右手: {right_display}", True, COLORS['info'])
            right_rect = right_text.get_rect(topleft=(20, 100))
            bg_right = right_rect.inflate(10,5)
            pygame.draw.rect(screen, COLORS['text_bg'], bg_right)
            pygame.draw.rect(screen, COLORS['white'], bg_right, 1)
            screen.blit(right_text, right_rect)
        
        if self.feedback_text:
            font_fb = get_font(48)
            color = COLORS['success'] if "正确" in self.feedback_text else COLORS['warning']
            fb = font_fb.render(self.feedback_text, True, color)
            fb_rect = fb.get_rect(center=(WIDTH//2, HEIGHT-100))
            bg_fb = fb_rect.inflate(20,10)
            pygame.draw.rect(screen, COLORS['text_bg'], bg_fb)
            pygame.draw.rect(screen, COLORS['white'], bg_fb, 2)
            screen.blit(fb, fb_rect)
        
        if self.completed:
            font_end = get_font(48)
            end_text = font_end.render("完成!", True, COLORS['success'])
            end_rect = end_text.get_rect(center=(WIDTH//2, HEIGHT//2))
            bg_end = end_rect.inflate(40,20)
            pygame.draw.rect(screen, COLORS['text_bg'], bg_end)
            pygame.draw.rect(screen, COLORS['white'], bg_end, 4)
            screen.blit(end_text, end_rect)

# -------------------------- Main Menu --------------------------
class MainMenu:
    def __init__(self):
        self.active = True
        self.selected_mode = None
        self.hover_timer = 0
        self.hover_duration = 1.5
        
        self.card_width = 250
        self.card_height = 300
        self.cards = [
            {
                'rect': pygame.Rect(150, 200, self.card_width, self.card_height),
                'mode': 'reaction',
                'title': '反应训练',
                'color': COLORS['reaction'],
                'desc': ['快速触碰', '下落的目标']
            },
            {
                'rect': pygame.Rect(400, 200, self.card_width, self.card_height),
                'mode': 'memory',
                'title': '记忆训练',
                'color': COLORS['memory'],
                'desc': ['记住数字顺序', '按顺序点击']
            },
            {
                'rect': pygame.Rect(650, 200, self.card_width, self.card_height),
                'mode': 'judgment',
                'title': '判断训练',
                'color': COLORS['judgment'],
                'desc': ['识别手势', '匹配下落方块']
            }
        ]
        
    def update(self, finger_pos, finger_visible):
        if not self.active:
            return
        
        hovered = False
        
        if finger_visible and finger_pos is not None:
            for card in self.cards:
                if card['rect'].collidepoint(finger_pos):
                    self.hover_timer += 1/FPS
                    if self.hover_timer >= self.hover_duration:
                        self.selected_mode = card['mode']
                        self.active = False
                    hovered = True
                    break
        
        if not hovered:
            self.hover_timer = 0
    
    def draw(self, screen, finger_pos, finger_visible):
        for i in range(HEIGHT):
            color_value = 20 + int(i * 30 / HEIGHT)
            pygame.draw.line(screen, (color_value, color_value+10, color_value+20), 
                           (0, i), (WIDTH, i))
        
        title_font = get_font(72)
        title = title_font.render("认知训练游戏", True, COLORS['white'])
        title_rect = title.get_rect(center=(WIDTH//2, 100))
        for offset in range(5, 0, -1):
            glow_surf = title_font.render("认知训练游戏", True, (100, 150, 200))
            glow_rect = glow_surf.get_rect(center=(WIDTH//2 + offset, 100 + offset))
            screen.blit(glow_surf, glow_rect)
        screen.blit(title, title_rect)
        
        for card in self.cards:
            is_hovered = finger_visible and finger_pos is not None and card['rect'].collidepoint(finger_pos)
            
            if is_hovered:
                color = COLORS['menu_card_hover']
                progress_width = (self.hover_timer / self.hover_duration) * card['rect'].width
                if progress_width > 0:
                    progress_rect = pygame.Rect(card['rect'].x, card['rect'].bottom - 10, 
                                               progress_width, 5)
                    pygame.draw.rect(screen, card['color'], progress_rect, border_radius=2)
            else:
                color = COLORS['menu_card']
            
            pygame.draw.rect(screen, color, card['rect'], border_radius=20)
            pygame.draw.rect(screen, card['color'], card['rect'], 3, border_radius=20)
            
            title_font_small = get_font(36)
            title_text = title_font_small.render(card['title'], True, COLORS['white'])
            title_rect = title_text.get_rect(center=(card['rect'].centerx, card['rect'].y + 40))
            screen.blit(title_text, title_rect)
            
            desc_font = get_font(24)
            y_offset = 100
            for line in card['desc']:
                desc_text = desc_font.render(line, True, COLORS['white'])
                desc_rect = desc_text.get_rect(center=(card['rect'].centerx, card['rect'].y + y_offset))
                screen.blit(desc_text, desc_rect)
                y_offset += 30
        
        hint_font = get_font(20)
        hints = [
            "将食指悬停在卡片上选择模式",
            "识别双手食指位置，绿色光标为左手，黄色为右手",
            "按 ESC 返回主菜单，R 查看报告"
        ]
        for i, hint in enumerate(hints):
            hint_text = hint_font.render(hint, True, COLORS['white'])
            hint_rect = hint_text.get_rect(center=(WIDTH//2, HEIGHT - 80 + i*25))
            screen.blit(hint_text, hint_rect)

# -------------------------- Main Game Class --------------------------
class Game:
    def __init__(self):
        # 初始化DNN识别器
        self.dnn_recognizer = DNNLeftRightRecognizer()
        
        self.data_tracker = DataTracker("user")
        self.memory_training = MemoryTraining()
        self.judgment_training = JudgmentTraining()
        self.main_menu = MainMenu()
        
        self.reaction_targets = []
        self.max_reaction_targets = 8
        self.reaction_spawn_timer = 0
        self.reaction_spawn_interval = 0.8
        
        self.training_mode = None
        
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.judge_text = ""
        self.judge_text_time = 0
        self.judge_position = (WIDTH//2, HEIGHT//2)
        
        self.left_finger_norm = (0, 0)
        self.right_finger_norm = (0, 0)
        self.left_finger_pos = (0, 0)
        self.right_finger_pos = (0, 0)
        self.left_finger_visible = False
        self.right_finger_visible = False
        
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        self.left_hand_side = None
        self.right_hand_side = None
        
        self.left_trail = []
        self.right_trail = []
        self.max_trail_length = 15
        
        self.game_running = True
        self.show_report = False
        
        self.hits = 0
        self.misses = 0
        self.current_frame = None
        self.mode_switch_timer = 0
        
    def enter_mode(self, mode):
        self.training_mode = mode
        self.data_tracker.update_mode(mode)
        
        if mode == "memory":
            self.memory_training.active = True
            self.memory_training.start_new_round()
        elif mode == "judgment":
            self.judgment_training.start_new_session()
        else:
            self.memory_training.active = False
            self.judgment_training.active = False
            self.reaction_targets.clear()
    
    def update(self):
        self._update_finger_screen_positions()
        
        if self.mode_switch_timer > 0:
            self.mode_switch_timer -= 1
        
        if self.training_mode is None:
            self.main_menu.update(
                self.left_finger_pos if self.left_finger_visible else 
                self.right_finger_pos if self.right_finger_visible else None,
                self.left_finger_visible or self.right_finger_visible
            )
            if self.main_menu.selected_mode:
                self.enter_mode(self.main_menu.selected_mode)
                self.main_menu.selected_mode = None
        else:
            if self.training_mode == "reaction":
                self._update_reaction_mode()
            elif self.training_mode == "memory":
                self._update_memory_mode()
            elif self.training_mode == "judgment":
                self._update_judgment_mode()
        
        self._update_trails()
        
        if self.judge_text_time > 0:
            self.judge_text_time -= 1/FPS
        else:
            self.judge_text = ""
    
    def _update_reaction_mode(self):
        center_x, center_y = WIDTH/2, HEIGHT/2
        area_width = WIDTH * 0.8
        area_height = HEIGHT * 0.8
        half_w = area_width / 2
        half_h = area_height / 2
        radius = 50
        x_min = max(center_x - half_w + radius, radius)
        x_max = min(center_x + half_w - radius, WIDTH - radius)
        y_min = max(center_y - half_h + radius, radius)
        y_max = min(center_y + half_h - radius, HEIGHT - radius)
        
        self.reaction_spawn_timer += 1/FPS
        if self.reaction_spawn_timer >= self.reaction_spawn_interval:
            self.reaction_spawn_timer = 0
            if len(self.reaction_targets) < self.max_reaction_targets:
                target = Target(x_range=(x_min, x_max), y_range=(y_min, y_max))
                self.reaction_targets.append(target)
        
        for target in self.reaction_targets[:]:
            result = target.update()
            if result == "timeout":
                self.reaction_targets.remove(target)
                self.misses += 1
                self.combo = 0
                self.data_tracker.record_miss()
                self.judge_text = "错过!"
                self.judge_text_time = 0.5
                self.judge_position = (target.x, target.y)
        
        for target in self.reaction_targets[:]:
            hit = False
            if self.left_finger_visible and target.check_hit(*self.left_finger_pos):
                hit = True
            elif self.right_finger_visible and target.check_hit(*self.right_finger_pos):
                hit = True
            if hit:
                self.reaction_targets.remove(target)
                self.hits += 1
                self.combo += 1
                self.max_combo = max(self.max_combo, self.combo)
                self.score += 100
                self.data_tracker.record_hit()
                self.data_tracker.update_score(self.score, self.combo)
                self.judge_text = "击中!"
                self.judge_text_time = 0.5
                self.judge_position = (target.x, target.y)
    
    def _update_memory_mode(self):
        result = self.memory_training.update()
        if result == "recall_start":
            self.judge_text = "开始回忆!"
            self.judge_text_time = 1.0
        
        if self.left_finger_visible or self.right_finger_visible:
            finger_pos = self.left_finger_pos if self.left_finger_visible else self.right_finger_pos
            mem_result = self.memory_training.handle_click(finger_pos)
            
            if mem_result == "round_complete":
                self.score += 100 * self.memory_training.round_size
                self.combo += 1
                self.max_combo = max(self.max_combo, self.combo)
                self.data_tracker.record_memory_round(self.memory_training.round_size, self.memory_training.round_size)
                self.data_tracker.update_score(self.score, self.combo)
                self.memory_training.round_size = min(8, self.memory_training.round_size + 1)
                self.memory_training.start_new_round()
                self.judge_text = "完成!"
                self.judge_text_time = 1.0
            elif mem_result == "wrong":
                self.score = max(0, self.score - 50)
                self.judge_text = "错误!"
                self.judge_text_time = 0.5
                self.combo = 0
    
    def _update_judgment_mode(self):
        left_gesture = None
        right_gesture = None
        
        if self.left_hand_landmarks is not None:
            left_gesture = recognize_gesture(self.left_hand_landmarks, self.dnn_recognizer)
        if self.right_hand_landmarks is not None:
            right_gesture = recognize_gesture(self.right_hand_landmarks, self.dnn_recognizer)
        
        self.judgment_training.update_gesture(left_gesture, right_gesture)
        self.judgment_training.update()
        
        if self.judgment_training.completed and not self.judgment_training.active:
            self.score += self.judgment_training.score
            self.data_tracker.update_score(self.score, self.combo)
            self.training_mode = None
            self.main_menu.active = True
    
    def _update_finger_screen_positions(self):
        if self.left_finger_visible:
            self.left_finger_pos = (
                int(self.left_finger_norm[0] * WIDTH),
                int(self.left_finger_norm[1] * HEIGHT)
            )
            self.left_finger_pos = (
                max(0, min(WIDTH, self.left_finger_pos[0])),
                max(0, min(HEIGHT, self.left_finger_pos[1]))
            )
        if self.right_finger_visible:
            self.right_finger_pos = (
                int(self.right_finger_norm[0] * WIDTH),
                int(self.right_finger_norm[1] * HEIGHT)
            )
            self.right_finger_pos = (
                max(0, min(WIDTH, self.right_finger_pos[0])),
                max(0, min(HEIGHT, self.right_finger_pos[1]))
            )
    
    def _update_trails(self):
        if self.left_finger_visible:
            self.left_trail.append(self.left_finger_pos)
            if len(self.left_trail) > self.max_trail_length:
                self.left_trail.pop(0)
        else:
            self.left_trail.clear()
        if self.right_finger_visible:
            self.right_trail.append(self.right_finger_pos)
            if len(self.right_trail) > self.max_trail_length:
                self.right_trail.pop(0)
        else:
            self.right_trail.clear()
    
    def draw_camera_background(self, surface):
        if self.current_frame is not None:
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            frame_surface = pygame.transform.scale(frame_surface, (WIDTH, HEIGHT))
            surface.blit(frame_surface, (0, 0))
    
    def draw_hand_skeleton(self, surface, hand_landmarks):
        if hand_landmarks is None:
            return
        points = []
        for lm in hand_landmarks.landmark:
            x = int(lm.x * WIDTH)
            y = int(lm.y * HEIGHT)
            points.append((x, y))
        
        finger_map = {
            1: 'thumb', 2: 'thumb', 3: 'thumb', 4: 'thumb',
            5: 'index', 6: 'index', 7: 'index', 8: 'index',
            9: 'middle', 10: 'middle', 11: 'middle', 12: 'middle',
            13: 'ring', 14: 'ring', 15: 'ring', 16: 'ring',
            17: 'pinky', 18: 'pinky', 19: 'pinky', 20: 'pinky',
        }
        
        connections = mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            start_pos = points[start_idx]
            end_pos = points[end_idx]
            if start_idx in finger_map:
                color = HAND_COLORS[finger_map[start_idx]]
            elif end_idx in finger_map:
                color = HAND_COLORS[finger_map[end_idx]]
            else:
                color = HAND_COLORS['wrist']
            pygame.draw.line(surface, color, start_pos, end_pos, 3)
        
        for i, pos in enumerate(points):
            if i in finger_map:
                color = HAND_COLORS[finger_map[i]]
            else:
                color = HAND_COLORS['wrist']
            pygame.draw.circle(surface, color, pos, 8)
            pygame.draw.circle(surface, COLORS['white'], pos, 3)
    
    def draw_ui(self, surface):
        def draw_text_with_bg(text, font_size, pos, color=COLORS['white'], bg=COLORS['text_bg']):
            font = get_font(font_size)
            txt = font.render(text, True, color)
            rect = txt.get_rect(center=pos)
            bg_rect = rect.inflate(20,10)
            pygame.draw.rect(surface, bg, bg_rect)
            pygame.draw.rect(surface, COLORS['white'], bg_rect, 2)
            surface.blit(txt, rect)
        
        if self.training_mode == "reaction":
            mode_display = "反应训练"
        elif self.training_mode == "memory":
            mode_display = "记忆训练"
        elif self.training_mode == "judgment":
            mode_display = "判断训练"
        else:
            return
        
        draw_text_with_bg(mode_display, 48, (WIDTH//2, 40))
        draw_text_with_bg(f"得分: {self.score}", 36, (100, 100))
        
        hint_font = get_font(24)
        hint_text = hint_font.render("按 ESC 返回主菜单", True, COLORS['info'])
        hint_rect = hint_text.get_rect(topleft=(20, HEIGHT - 40))
        surface.blit(hint_text, hint_rect)
        
        report_text = hint_font.render("R 查看报告", True, COLORS['success'])
        report_rect = report_text.get_rect(topright=(WIDTH - 20, 20))
        surface.blit(report_text, report_rect)
    
    def draw_fingers(self, surface):
        if self.left_finger_visible:
            x, y = self.left_finger_pos
            for i, pos in enumerate(self.left_trail):
                alpha = int(200 * (i / len(self.left_trail))) if self.left_trail else 0
                size = 6 + i // 3
                if size > 0:
                    trail_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.circle(trail_surf, (*COLORS['success'][:3], alpha), (size, size), size)
                    surface.blit(trail_surf, (pos[0]-size, pos[1]-size))
            pygame.draw.circle(surface, COLORS['success'], (x, y), 25, 3)
            pygame.draw.line(surface, COLORS['success'], (x-35, y), (x-15, y), 3)
            pygame.draw.line(surface, COLORS['success'], (x+15, y), (x+35, y), 3)
            pygame.draw.line(surface, COLORS['success'], (x, y-35), (x, y-15), 3)
            pygame.draw.line(surface, COLORS['success'], (x, y+15), (x, y+35), 3)
            pygame.draw.circle(surface, COLORS['success'], (x, y), 8)
            pygame.draw.circle(surface, COLORS['white'], (x, y), 4)
        
        if self.right_finger_visible:
            x, y = self.right_finger_pos
            for i, pos in enumerate(self.right_trail):
                alpha = int(200 * (i / len(self.right_trail))) if self.right_trail else 0
                size = 6 + i // 3
                if size > 0:
                    trail_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.circle(trail_surf, (*COLORS['info'][:3], alpha), (size, size), size)
                    surface.blit(trail_surf, (pos[0]-size, pos[1]-size))
            pygame.draw.circle(surface, COLORS['info'], (x, y), 25, 3)
            pygame.draw.line(surface, COLORS['info'], (x-35, y), (x-15, y), 3)
            pygame.draw.line(surface, COLORS['info'], (x+15, y), (x+35, y), 3)
            pygame.draw.line(surface, COLORS['info'], (x, y-35), (x, y-15), 3)
            pygame.draw.line(surface, COLORS['info'], (x, y+15), (x, y+35), 3)
            pygame.draw.circle(surface, COLORS['info'], (x, y), 8)
            pygame.draw.circle(surface, COLORS['white'], (x, y), 4)
    
    def draw_judgment(self, surface):
        if self.judge_text:
            font = get_font(48)
            color = COLORS['success'] if "击中" in self.judge_text or "正确" in self.judge_text or "完成" in self.judge_text else COLORS['warning']
            txt = font.render(self.judge_text, True, color)
            rect = txt.get_rect(center=self.judge_position)
            bg_rect = rect.inflate(20,10)
            pygame.draw.rect(surface, COLORS['text_bg'], bg_rect)
            pygame.draw.rect(surface, COLORS['white'], bg_rect, 2)
            surface.blit(txt, rect)
    
    def update_finger_positions(self, results):
        self.left_finger_visible = False
        self.right_finger_visible = False
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        self.left_hand_side = None
        self.right_hand_side = None
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[hand_idx].classification[0].label
                tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                if label == "Left":
                    self.left_hand_landmarks = hand_landmarks
                    self.left_hand_side = label
                    self.left_finger_norm = (tip.x, tip.y)
                    self.left_finger_visible = True
                else:
                    self.right_hand_landmarks = hand_landmarks
                    self.right_hand_side = label
                    self.right_finger_norm = (tip.x, tip.y)
                    self.right_finger_visible = True
    
    def draw(self, surface):
        #self.draw_camera_background(surface)
        surface.fill((0, 0, 0))  # 添加纯黑色背景
        
        if self.training_mode is None:
            self.main_menu.draw(surface, 
                               self.left_finger_pos if self.left_finger_visible else 
                               self.right_finger_pos if self.right_finger_visible else None,
                               self.left_finger_visible or self.right_finger_visible)
            self.draw_fingers(surface)
        else:
            if self.training_mode == "reaction":
                for t in self.reaction_targets:
                    t.draw(surface)
                self.draw_fingers(surface)
            elif self.training_mode == "memory":
                self.memory_training.draw(surface)
                self.draw_fingers(surface)
            elif self.training_mode == "judgment":
                self.judgment_training.draw(surface)
                if self.left_hand_landmarks is not None:
                    self.draw_hand_skeleton(surface, self.left_hand_landmarks)
                if self.right_hand_landmarks is not None:
                    self.draw_hand_skeleton(surface, self.right_hand_landmarks)
            
            self.draw_ui(surface)
            self.draw_judgment(surface)
        
        if self.show_report:
            self.data_tracker.show_report_screen(surface)

# -------------------------- Main Loop --------------------------
def main():
    game = Game()
    
    while game.game_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.game_running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if game.training_mode is not None:
                        game.training_mode = None
                        game.main_menu.active = True
                    else:
                        game.game_running = False
                if event.key == pygame.K_r:
                    if game.show_report:
                        game.data_tracker.save_session()
                        game.show_report = False
                    else:
                        game.data_tracker.calculate_metrics()
                        game.show_report = True
                if event.key == pygame.K_s and game.show_report:
                    fn = game.data_tracker.save_session()
                      
                    print(f"报告已保存: {fn}")
        
              
        
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        game.current_frame = frame.copy()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        game.update_finger_positions(results)
        game.update()
        game.draw(render_surface)
        scaled_surface = pygame.transform.smoothscale(render_surface, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        screen.blit(scaled_surface, (0, 0))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    
    game.data_tracker.calculate_metrics()
    print(game.data_tracker.generate_report())
    game.data_tracker.save_session()
    game.dnn_recognizer.close()

if __name__ == "__main__":
    main()