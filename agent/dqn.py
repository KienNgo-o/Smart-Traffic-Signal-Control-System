import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DuelingDQN(nn.Module):
    """
    Mạng Neural kiến trúc Dueling: Tách biệt Value (Giá trị trạng thái) 
    và Advantage (Lợi thế hành động).
    """
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        
        # Luồng chia sẻ (Shared Feature Layers)
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Luồng Value: Đánh giá xem ngã tư hiện tại "tốt" hay "tệ"
        self.val_fc = nn.Linear(128, 64)
        self.val_out = nn.Linear(64, 1)
        
        # Luồng Advantage: Đánh giá xem "Chuyển pha" hay "Giữ pha" tốt hơn
        self.adv_fc = nn.Linear(128, 64)
        self.adv_out = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Tính Value
        val = F.relu(self.val_fc(x))
        val = self.val_out(val) # Shape: (batch_size, 1)
        
        # Tính Advantage
        adv = F.relu(self.adv_fc(x))
        adv = self.adv_out(adv) # Shape: (batch_size, action_dim)
        
        # Kết hợp Q-value với công thức trừ Mean để ổn định toán học
        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_values


class D3QNAgent:
    """
    Tác tử Double Dueling DQN (D3QN).
    Xử lý One-hot encoding và logic Double Q-Learning.
    """
    def __init__(self, num_lanes, num_phases, action_dim, lr=1e-4, gamma=0.99):
        self.num_lanes = num_lanes
        self.num_phases = num_phases
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Tính toán lại State Dimension thực tế sau khi One-hot
        # State = Queue (num_lanes) + Wait (num_lanes) + Phase (One-hot num_phases)
        self.state_dim = (num_lanes * 2) + num_phases
        
        # Khởi tạo Policy Net và Target Net
        self.policy_net = DuelingDQN(self.state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(self.state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net chỉ dùng để infer, không train
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def preprocess_state(self, state_array):
        """
        Chuẩn hóa state: Scale Queue/Wait và One-hot Phase ID.
        """
        # 1. Tách các thành phần (Giả sử có 8 làn)
        queues = state_array[:self.num_lanes]
        waits = state_array[self.num_lanes: 2 * self.num_lanes]
        phase_id = int(state_array[-1])
        
        # 2. Scale dữ liệu (Tránh Gradient Exploding)
        # Giả định max_queue = 50, max_wait = 1000 để đưa về khoảng [0, 1]
        norm_queues = queues / 50.0
        norm_waits = waits / 1000.0
        
        # 3. One-hot Encoding cho Phase
        one_hot_phase = np.zeros(self.num_phases, dtype=np.float32)
        one_hot_phase[phase_id] = 1.0
        
        # 4. Nối lại thành vector hoàn chỉnh
        processed_state = np.concatenate((norm_queues, norm_waits, one_hot_phase))
        return torch.FloatTensor(processed_state).unsqueeze(0).to(self.device)

    def compute_loss(self, states, actions, rewards, next_states, dones, durations, is_weights):
        """Lõi Semi-MDP: Chiết khấu tương lai dựa trên gamma^duration."""
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            target_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            
            # [CỐT LÕI SEMI-MDP]: Chiết khấu động theo thời gian thực thi của hành động
            # expected_q = r + (gamma^tau) * Q_target
            discount_factors = torch.pow(self.gamma, durations)
            expected_q_values = rewards + (discount_factors * target_q_values * (1 - dones))
            
        # Tính Loss với Importance Sampling Weights
        loss = (is_weights * F.smooth_l1_loss(current_q_values, expected_q_values, reduction='none')).mean()
        td_errors = torch.abs(current_q_values - expected_q_values).detach().cpu().numpy()
        
        return loss, td_errors