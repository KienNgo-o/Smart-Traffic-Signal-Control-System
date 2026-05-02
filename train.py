import os
import torch
import numpy as np
from env.sumo_env import SumoEnv
from agent.dqn import D3QNAgent
from agent.replay_buffer import PrioritizedReplayBuffer
# Giả định bạn có hàm set_seed trong utils
from utils.seed import set_global_seed 
import matplotlib.pyplot as plt

def train():
    # 1. Khởi tạo môi trường
    env = SumoEnv(sumocfg_file="sumo/scenario.sumocfg", use_gui=False)
    
    set_global_seed(42)
    # [ĐÃ SỬA LỖI 1]: Khóa PRNG của thư viện Gymnasium cho không gian hành động
    env.action_space.seed(42)
    # [ĐÃ SỬA LỖI SEQUENCING]: 
    # BẮT BUỘC phải kích hoạt SUMO một lần để Controller quét cấu hình vật lý.
    # Nếu không, số làn xe (num_lanes) sẽ bị bằng 0.
    env.reset()
    
    num_lanes = len(env.controller.lanes)
    num_phases = env.controller.num_phases
    action_dim = env.action_space.n
    
    # 2. Khởi tạo Tác tử (Lúc này num_lanes và num_phases đã có giá trị thực)
    agent = D3QNAgent(num_lanes=num_lanes, num_phases=num_phases, action_dim=action_dim, lr=1e-3)
    
    # 3. Siêu tham số
    batch_size = 64
    target_update_freq = 10
    num_episodes = 150
    
    # =================================================================
    # [ĐÃ SỬA LỖI HARDCODE]: Lịch trình động (Dynamic Scheduling)
    # Tự động tính toán decay và increment dựa trên độ dài num_episodes.
    # Dù bạn có đổi num_episodes thành 500 hay 1000, code vẫn chạy chuẩn 100%.
    # =================================================================
    
    # A. Tính Epsilon Decay (Để chạm sàn 0.04 vào đúng 90% số tập)
    epsilon_start = 1.0
    epsilon_end = 0.04
    target_episode = num_episodes * 0.9 
    # Công thức toán học: epsilon_start * (decay)^target = epsilon_end
    epsilon_decay = (epsilon_end / epsilon_start) ** (1.0 / target_episode)
    epsilon = epsilon_start
    
    # B. Tính Beta Increment cho PER (Để chạm 1.0 vào đúng bước cuối cùng)
    # Giả định: 7200 giây mô phỏng / 16 giây mỗi action = ~450 steps/tập
    max_steps_per_ep = 7200 / 16 
    total_expected_steps = max_steps_per_ep * num_episodes
    beta_start = 0.4
    beta_increment = (1.0 - beta_start) / total_expected_steps
    
    # Khởi tạo Buffer với thông số động vừa tính
    buffer = PrioritizedReplayBuffer(
        max_size=50000, 
        beta=beta_start, 
        beta_increment_per_sampling=beta_increment
    )
    # =================================================================
    
    os.makedirs("models", exist_ok=True)
    print(f"Bắt đầu huấn luyện trên thiết bị: {agent.device}")
    episode_rewards = []
    # 4. Vòng lặp Huấn luyện
    for episode in range(1, num_episodes + 1):
        # [ĐÃ SỬA LỖI 2]: Truyền một seed khác nhau cho mỗi tập (ví dụ: 43, 44, 45...)
        # Việc này ép AI mỗi tập phải đối mặt với một luồng giao thông hoàn toàn mới (tránh học vẹt).
        # ĐỒNG THỜI, vì có quy luật toán học rõ ràng (42 + episode), quá trình 
        # train này vẫn đảm bảo tính tái lập 100% nếu chạy lại từ đầu!
        raw_state, _ = env.reset(seed=42 + episode)
        
        # Chuẩn hóa state ngay từ đầu để dùng cho cả hành động và lưu trữ
        state = agent.preprocess_state(raw_state)
        
        total_reward = 0
        step_count = 0
        done = False
        
        while not done:
            # --- CHỌN HÀNH ĐỘNG (Epsilon-Greedy) ---
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = agent.policy_net(state)
                    action = q_values.argmax().item()
            
            # --- TƯƠNG TÁC MÔI TRƯỜNG ---
            raw_next_state, reward, terminated, truncated, info = env.step(action)
            duration = info["action_duration"] # Lấy thời gian thực từ EdgeController
            done = terminated or truncated

            next_state = agent.preprocess_state(raw_next_state)

            # Lưu thêm duration vào buffer
            buffer.add(state, action, reward, next_state, done, duration)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            # --- HỌC HỎI TỪ KINH NGHIỆM ---
            if buffer.tree.n_entries > batch_size:
                batch, idxs, is_weights = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones, durations = batch
                
                # Chuyển durations thành Tensor để tính lũy thừa trên GPU
                durations_t = torch.FloatTensor(durations).to(agent.device)
                
                # [SỬA LỖI HIỆU NĂNG 2]: states và next_states đã là Tensor sẵn
                # Chỉ cần ghép chúng lại bằng torch.cat, loại bỏ hoàn toàn cổ chai I/O
                states_t = torch.cat(states, dim=0)
                next_states_t = torch.cat(next_states, dim=0)
                
                # Các primitives (số nguyên/thực) thì vẫn chuyển đổi bình thường
                actions_t = torch.LongTensor(actions).to(agent.device)
                rewards_t = torch.FloatTensor(rewards).to(agent.device)
                dones_t = torch.FloatTensor(dones).to(agent.device)
                is_weights_t = torch.FloatTensor(is_weights).to(agent.device)
                loss, td_errors = agent.compute_loss(
                states_t, actions_t, rewards_t, next_states_t, dones_t, durations_t, is_weights_t
                 )

                agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), max_norm=1.0)
                agent.optimizer.step()

                buffer.update_priorities(idxs, td_errors)
     
                
        # --- CẬP NHẬT TẬP (Episode End) ---
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if episode % target_update_freq == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        print(f"Episode {episode:03d} | Steps: {step_count:04d} | Reward: {total_reward:8.2f} | Epsilon: {epsilon:.3f}")
        
        # [THÊM DÒNG NÀY]: Lưu reward của tập hiện tại vào mảng
        episode_rewards.append(total_reward)
        
        if episode % 20 == 0:
            torch.save(agent.policy_net.state_dict(), f"models/d3qn_per_ep{episode}.pth")

    torch.save(agent.policy_net.state_dict(), "models/d3qn_per_final.pth")
    print("Huấn luyện hoàn tất!")
    env.close()

    # =================================================================
    # [ĐOẠN NÀY ĐỂ XUẤT CSV VÀ VẼ BIỂU ĐỒ]
    # =================================================================
    # 1. Lưu data thô ra file CSV phòng trường hợp cần dùng phần mềm khác vẽ (như Excel)
    np.savetxt("models/training_rewards.csv", episode_rewards, delimiter=",", fmt="%.2f")
    
    # 2. Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    
    # Đường màu xanh nhạt: Dữ liệu thô (nhiễu, giật)
    plt.plot(episode_rewards, label='Raw Reward', alpha=0.3, color='blue')
    
    # Đường màu đỏ: Moving Average (Trung bình trượt) giúp nhìn rõ xu hướng học
    window_size = 10
    if len(episode_rewards) >= window_size:
        # Tính trung bình mỗi 10 tập
        smoothed_rewards = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        # Dịch chuyển trục X để khớp với đường Raw
        plt.plot(range(window_size-1, len(episode_rewards)), smoothed_rewards, 
                 label=f'Moving Average (Window={window_size})', color='red', linewidth=2)
                 
    plt.xlabel('Episode (Số lượng tập)', fontsize=12, fontweight='bold')
    plt.ylabel('Total Reward (Điểm phần thưởng)', fontsize=12, fontweight='bold')
    plt.title('Đường cong Hội tụ của D3QN Agent trong quá trình học', fontsize=14, pad=15)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Lưu file ảnh
    plt.tight_layout()
    plt.savefig("models/reward_convergence_plot.png", dpi=300)
    print("\n[+] Đã lưu file dữ liệu thô tại: models/training_rewards.csv")
    print("[+] Đã lưu biểu đồ thành công tại: models/reward_convergence_plot.png")

if __name__ == "__main__":
    train()