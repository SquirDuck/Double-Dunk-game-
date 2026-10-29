import math
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

class MiniDoubleDunkEnv(gym.Env):
    """
    Môi trường bóng rổ 1v1 đơn giản hóa:
    - Player (Tấn công): Cố gắng lách qua hậu vệ, tiếp cận rổ và ném bóng.
    - Defender (Hậu vệ): Di chuyển cản đường giữa bóng và rổ.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.width = 400.0
        self.height = 400.0
        self.hoop_pos = np.array([200.0, 50.0], dtype=np.float32)

        # 6 Hành động rời rạc:
        # 0: Đứng yên, 1: Lên, 2: Xuống, 3: Trái, 4: Phải, 5: Ném
        self.action_space = spaces.Discrete(6)

        # Trạng thái 6 chiều (Vector chuẩn hóa [0.0 -> 1.0]):
        # [player_x, player_y, def_x, def_y, dist_to_hoop, has_ball]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self.window = None
        self.clock = None

    def _get_obs(self):
        max_dist = math.sqrt(self.width**2 + self.height**2)
        dist = np.linalg.norm(self.player_pos - self.hoop_pos) / max_dist
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
        self.max_steps = 120

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        speed = 5.0
        prev_dist = np.linalg.norm(self.player_pos - self.hoop_pos)

        # 1. Thực thi hành động di chuyển của Cầu thủ
        if action == 1: self.player_pos[1] -= speed
        elif action == 2: self.player_pos[1] += speed
        elif action == 3: self.player_pos[0] -= speed
        elif action == 4: self.player_pos[0] += speed

        # Giới hạn trong sân đấu
        self.player_pos[0] = np.clip(self.player_pos[0], 20.0, self.width - 20.0)
        self.player_pos[1] = np.clip(self.player_pos[1], 50.0, self.height - 20.0)

        # 2. Hậu vệ tự động đuổi theo cản phá (Heuristic)
        target = (self.player_pos + self.hoop_pos) / 2.0
        diff = target - self.def_pos
        dist_diff = np.linalg.norm(diff)
        if dist_diff > 0:
            self.def_pos += (diff / dist_diff) * 2.2

        # 3. Phạt/Thưởng và kiểm tra điều kiện kết thúc
        reward = -0.01  # Phạt nhẹ theo thời gian (ép hoàn thành sớm)
        curr_dist = np.linalg.norm(self.player_pos - self.hoop_pos)
        reward += (prev_dist - curr_dist) * 0.05  # Thưởng khi tiến lại gần rổ

        terminated = False
        truncated = self.step_count >= self.max_steps

        # Bị cướp bóng nếu chạm vào người hậu vệ
        if np.linalg.norm(self.player_pos - self.def_pos) < 20.0:
            reward = -10.0
            terminated = True

        # Ném rổ (Action == 5)
        if action == 5 and self.has_ball:
            self.has_ball = False
            # Càng đứng gần rổ xác suất ném trúng càng cao
            hit_prob = max(0.05, 1.0 - (curr_dist / 220.0))
            if np.random.rand() < hit_prob:
                reward = 15.0
            else:
                reward = -5.0
            terminated = True

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, {}

    def _render_frame(self):
        import pygame
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((int(self.width), int(self.height)))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((205, 133, 63))  # Màu gỗ sân bóng

        # Rổ bóng
        pygame.draw.circle(canvas, (255, 69, 0), (int(self.hoop_pos[0]), int(self.hoop_pos[1])), 12, 3)
        # Hậu vệ (Đỏ)
        pygame.draw.circle(canvas, (220, 20, 60), (int(self.def_pos[0]), int(self.def_pos[1])), 12)
        # Cầu thủ tấn công (Xanh)
        pygame.draw.circle(canvas, (30, 144, 255), (int(self.player_pos[0]), int(self.player_pos[1])), 12)
        # Quả bóng (Cam)
        if self.has_ball:
            pygame.draw.circle(canvas, (255, 140, 0), (int(self.player_pos[0] + 6), int(self.player_pos[1])), 6)

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()


class QNetwork(nn.Module):
    """Mạng Nơ-ron sâu xấp xỉ hàm Q(s, a)."""
    def __init__(self, state_dim=6, action_dim=6):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Bộ nhớ lưu trữ các lượt chơi (s, a, r, s', done) để lấy mẫu ngẫu nhiên."""
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(done)
        )

    def __len__(self):
        return len(self.buffer)
