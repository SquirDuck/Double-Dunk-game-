from stable_baselines3 import PPO
from basketball_env import MiniDoubleDunkEnv

# 1. Khởi tạo môi trường không render để tối đa hóa tốc độ CPU
train_env = MiniDoubleDunkEnv(render_mode=None)

# 2. Định nghĩa Model RL
# Policy: MlpPolicy (Multi-Layer Perceptron Policy)
model = PPO(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99, # Hệ số chiết khấu phần thưởng tương lai
    verbose=1
)

print("--- Đang huấn luyện Model (50,000 steps)... ---")
model.learn(total_timesteps=50000)
model.save("dunk_ppo_model")
print("--- Đã huấn luyện xong và lưu model! ---")
train_env.close()

# 3. Trực quan hóa: Load model và mở cửa sổ Pygame để xem AI chơi
print("--- Mở game xem AI thi đấu ---")
test_env = MiniDoubleDunkEnv(render_mode="human")
obs, info = test_env.reset()

for episode in range(5):
    done = False
    total_reward = 0
    obs, info = test_env.reset()
    while not done:
        # Policy dự đoán hành động tốt nhất dựa trên observation hiện tại
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Episode {episode + 1} kết thúc với Reward: {total_reward:.2f}")

test_env.close()
