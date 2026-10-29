import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from core import MiniDoubleDunkEnv, QNetwork, ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim=6, action_dim=6):
        self.action_dim = action_dim
        
        # Mạng dự đoán (Policy Net) và Mạng mục tiêu (Target Net)
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(capacity=20000)

        # Tham số RL
        self.gamma = 0.99
        self.batch_size = 64
        
        # Chiến lược Exploration: Bắt đầu từ 100% ngẫu nhiên -> giảm dần về 5%
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.996

    def select_action(self, state):
        # 1. EXPLORATION: Chọn ngẫu nhiên hoàn toàn
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        # 2. EXPLOITATION: Khai thác kiến thức (chọn action có Q cao nhất)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        # Lấy mẫu ngẫu nhiên (phá vỡ tương quan chuỗi thời gian)
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Tính Q(s, a) hiện tại theo Policy Network
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Tính Q mục tiêu theo Phương trình Bellman: Target = r + gamma * max_a'(Q_target(s', a'))
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Cập nhật trọng số mạng nơ-ron
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        # Đồng bộ trọng số từ Policy Net sang Target Net
        self.target_net.load_state_dict(self.policy_net.state_dict())


def main():
    # Bước 1: Huấn luyện ở chế độ Headless (tắt render để đạt tốc độ tối đa)
    env = MiniDoubleDunkEnv(render_mode=None)
    agent = DQNAgent(state_dim=6, action_dim=6)
    
    num_episodes = 600
    target_update_freq = 10

    print("==================================================")
    print("BẮT ĐẦU HUẤN LUYỆN DQN TỪ CON SỐ 0 (CHƯA BIẾT GÌ)")
    print("==================================================")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Đẩy trải nghiệm vào buffer
            agent.memory.push(state, action, reward, next_state, float(done))
            
            # Cập nhật gradient
            agent.update()

            state = next_state
            total_reward += reward

        # Giảm dần xác suất ngẫu nhiên
        agent.decay_epsilon()

        # Cập nhật Target Net định kỳ
        if episode % target_update_freq == 0:
            agent.update_target_network()

        if episode % 30 == 0:
            print(f"Trận {episode:3d}/{num_episodes} | "
                  f"Reward: {total_reward:6.2f} | "
                  f"Epsilon (Tỷ lệ Explore): {agent.epsilon:.3f}")

    # Bước 2: Lưu mô hình
    torch.save(agent.policy_net.state_dict(), "double_dunk_dqn.pth")
    print("\n--- ĐÃ LƯU TRỌNG SỐ MÔ HÌNH VÀO: double_dunk_dqn.pth ---")
    env.close()

    # Bước 3: Đánh giá trực quan (Mở cửa sổ Pygame hiển thị kết quả)
    print("\n--- MỞ CỬA SỔ XEM AI BIỂU DIỄN SAU KHI HỌC ---")
    eval_env = MiniDoubleDunkEnv(render_mode="human")
    
    for ep in range(5):
        state, _ = eval_env.reset()
        done = False
        eval_reward = 0
        while not done:
            # Khi test, tắt hẳn Explore (epsilon = 0)
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action = agent.policy_net(state_t).argmax().item()
            
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            state = next_state
            eval_reward += reward
        print(f"Trận biểu diễn {ep + 1}: Reward đạt được = {eval_reward:.2f}")

    eval_env.close()

if __name__ == "__main__":
    main()
