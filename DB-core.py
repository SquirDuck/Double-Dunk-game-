import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class MiniDoubleDunkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.width = 400
        self.height = 400
        self.hoop_pos = np.array([200.0, 50.0], dtype=np.float32)

        # Actions: 0: Đứng yên, 1: Lên, 2: Xuống, 3: Trái, 4: Phải, 5: Ném
        self.action_space = spaces.Discrete(6)

        # Observation (Vector 6 chiều, chuẩn hóa 0 -> 1):
        # [player_x, player_y, def_x, def_y, dist_to_hoop, has_ball]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self.window = None
        self.clock = None

    def _get_obs(self):
        dist = np.linalg.norm(self.player_pos - self.hoop_pos) / math.sqrt(self.width**2 + self.height**2)
        return np.array([
            self.player_pos[0] / self.width,
            self.player_pos[1] / self.height,
            self.def_pos[0] / self.width,
            self.def_pos[1] / self.height,
            dist,
            1.0 if self.has_ball else 0.0
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_pos = np.array([200.0, 320.0], dtype=np.float32)
        self.def_pos = np.array([200.0, 180.0], dtype=np.float32)
        self.has_ball = True
        self.step_count = 0
        self.max_steps = 150

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        speed = 5.0
        prev_dist = np.linalg.norm(self.player_pos - self.hoop_pos)

        # 1. Thực thi Action của Player
        if action == 1: self.player_pos[1] -= speed
        elif action == 2: self.player_pos[1] += speed
        elif action == 3: self.player_pos[0] -= speed
        elif action == 4: self.player_pos[0] += speed

        # Giới hạn trong sân
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.width - 20)
        self.player_pos[1] = np.clip(self.player_pos[1], 50, self.height - 20)

        # 2. Logic phòng thủ của Defender (Chặn đường giữa bóng và rổ)
        target = (self.player_pos + self.hoop_pos) / 2.0
        diff = target - self.def_pos
        dist_diff = np.linalg.norm(diff)
        if dist_diff > 0:
            self.def_pos += (diff / dist_diff) * 2.5

        # 3. Tính Reward & Kiểm tra trạng thái kết thúc (Terminal)
        reward = 0.0
        terminated = False
        truncated = self.step_count >= self.max_steps

        curr_dist = np.linalg.norm(self.player_pos - self.hoop_pos)
        
        # Phần thưởng định hình (Reward Shaping): khuyến khích tiến lại gần rổ
        reward += (prev_dist - curr_dist) * 0.05 
        reward -= 0.01 # Phạt nhẹ theo thời gian để kích thích ghi điểm nhanh

        # Bị cướp bóng nếu chạm hậu vệ
        if np.linalg.norm(self.player_pos - self.def_pos) < 22:
            reward -= 10.0
            terminated = True

        # Hành động Ném bóng (Action == 5)
        if action == 5 and self.has_ball:
            self.has_ball = False
            # Xác suất ném trúng phụ thuộc khoảng cách tới rổ
            shoot_prob = max(0.1, 1.0 - (curr_dist / 250.0))
            if np.random.rand() < shoot_prob:
                reward += 20.0 # Ném trúng
            else:
                reward -= 5.0  # Ném trượt
            terminated = True

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, {}

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((205, 133, 63)) # Sân gỗ

        # Rổ bóng
        pygame.draw.circle(canvas, (255, 69, 0), (int(self.hoop_pos[0]), int(self.hoop_pos[1])), 12, 3)
        # Hậu vệ (Đỏ)
        pygame.draw.circle(canvas, (220, 20, 60), (int(self.def_pos[0]), int(self.def_pos[1])), 12)
        # Cầu thủ (Xanh dương)
        pygame.draw.circle(canvas, (30, 144, 255), (int(self.player_pos[0]), int(self.player_pos[1])), 12)
        # Bóng (Vàng cam)
        if self.has_ball:
            pygame.draw.circle(canvas, (255, 140, 0), (int(self.player_pos[0] + 6), int(self.player_pos[1])), 6)

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
