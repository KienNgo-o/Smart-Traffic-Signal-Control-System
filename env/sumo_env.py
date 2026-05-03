import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Đảm bảo SUMO_HOME được nạp
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ.get('SUMO_HOME'), 'tools'))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

from env.edge_controller import EdgeController

class SumoEnv(gym.Env):
    """
    Môi trường Gymnasium chuẩn Production cho SUMO.
    Tích hợp libsumo để tối ưu hiệu năng và tránh kẹt TCP Socket.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, sumocfg_file, use_gui=False, max_steps=7200, reward_gamma=0.99):
        super(SumoEnv, self).__init__()
        self.sumocfg = sumocfg_file
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.current_step = 0
        self.reward_gamma = reward_gamma
        
        # 1. TỐI ƯU HIỆU NĂNG: Khởi tạo module giao tiếp tĩnh
        if self.use_gui:
            import traci
            self.sumo_engine = traci
        else:
            # libsumo dùng C++ bindings trực tiếp vào RAM, nhanh hơn 3-5 lần
            import libsumo as traci
            self.sumo_engine = traci
            
        # 2. Khởi tạo Edge Controller
        # Truyền sumo_engine vào để controller dùng chung một context
        self.controller = EdgeController("center", engine=self.sumo_engine) 
        
        # Action Space: 0 = Keep Phase, 1 = Switch Phase
        self.action_space = spaces.Discrete(2)
        
        # 3. TỰ ĐỘNG HÓA OBSERVATION SPACE
        num_lanes = len(self.controller.lanes)
        obs_dim = (num_lanes * 2) + 1  # queue + wait per lane + 1 phase_id
        
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Tracking metrics
        self.prev_total_wait = 0.0
        self.prev_total_time_loss = 0.0
        self.prev_phi = 0.0 
        self.sumo_running = False

    def _get_active_vehicle_metrics(self):
        """Collect active-vehicle metrics on controlled incoming lanes."""
        vehicle_ids = set()
        for lane in self.controller.lanes:
            vehicle_ids.update(self.sumo_engine.lane.getLastStepVehicleIDs(lane))

        vehicle_count = len(vehicle_ids)
        total_time_loss = 0.0
        for veh_id in vehicle_ids:
            try:
                total_time_loss += self.sumo_engine.vehicle.getTimeLoss(veh_id)
            except Exception:
                continue

        return vehicle_count, total_time_loss

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Nếu truyền seed từ bên ngoài vào (ví dụ lúc đánh giá mô hình)
        current_seed = seed if seed is not None else 42
        
        if self.sumo_running:
            self.sumo_engine.close()
            
        
        binary = "sumo-gui" if self.use_gui else "sumo"
        cmd = [
            binary, "-c", self.sumocfg, 
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--waiting-time-memory", "10000",
            "--time-to-teleport", "300",
            # BỎ THAM SỐ NÀY: "--random",  <-- Xóa cờ này đi
            "--seed", str(current_seed)  # THÊM THAM SỐ NÀY: Ép SUMO dùng đúng seed
        ]

        if options is not None and "tripinfo" in options:
            cmd.extend([
                "--tripinfo-output", options["tripinfo"],
                "--tripinfo-output.write-unfinished", "true",
                "--device.emissions.probability", "1.0"
            ])
        self.sumo_engine.start(cmd)
        self.sumo_running = True
        
        self.controller.setup()
        self.prev_total_wait = 0.0
        self.prev_total_time_loss = 0.0
        self.prev_phi = 0.0
        
        obs = self.controller.get_state()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        # [ĐÃ SỬA]: Lấy chính xác số giây thực tế đã trôi qua
        duration = self.controller.apply_action(action)
        self.current_step += duration
        
        next_state = self.controller.get_state()
        
        num_lanes = len(self.controller.lanes)
        queues = next_state[:num_lanes]
        waits = next_state[num_lanes:2*num_lanes]
        
        # 4. Reward aligned with evaluation: average waiting time and time loss.
        current_total_wait = np.sum(waits)
        vehicle_count, current_total_time_loss = self._get_active_vehicle_metrics()
        normalizer = max(1, vehicle_count)

        avg_active_wait = current_total_wait / normalizer
        avg_active_time_loss = current_total_time_loss / normalizer
        max_lane_wait = float(np.max(waits)) if len(waits) else 0.0

        wait_diff = self.prev_total_wait - current_total_wait
        time_loss_diff = self.prev_total_time_loss - current_total_time_loss
        self.prev_total_wait = current_total_wait
        self.prev_total_time_loss = current_total_time_loss
        
        mean_wait_penalty = -0.35 * avg_active_wait
        time_loss_penalty = -0.25 * avg_active_time_loss
        delta_wait_reward = 0.02 * wait_diff
        delta_time_loss_reward = 0.01 * time_loss_diff
        tail_wait_penalty = -0.03 * max_lane_wait
        queue_penalty = -0.02 * np.sum(queues)
        switch_penalty = -0.30 if action == 1 else 0.0

        base_reward = (
            mean_wait_penalty
            + time_loss_penalty
            + delta_wait_reward
            + delta_time_loss_reward
            + tail_wait_penalty
            + queue_penalty
            + switch_penalty
        )
        
        # PBRS (Potential-Based Reward Shaping) cho Semi-MDP:
        # dùng cùng gamma^duration với target Q-learning.
        discount = self.reward_gamma ** float(duration)
        current_phi = -((0.65 * avg_active_wait) + (0.35 * avg_active_time_loss))
        pbrs_term = (discount * current_phi) - self.prev_phi
        self.prev_phi = current_phi
        
        total_reward = base_reward + (pbrs_term * 0.05) 
        
        # 5. ĐIỀU KIỆN KẾT THÚC CHUẨN XÁC
        terminated = self.sumo_engine.simulation.getMinExpectedNumber() <= 0
        truncated = self.current_step >= self.max_steps
        
        info = {
            "step": self.current_step,
            "total_wait": current_total_wait,
            "avg_active_wait": avg_active_wait,
            "total_time_loss": current_total_time_loss,
            "avg_active_time_loss": avg_active_time_loss,
            "total_queue": np.sum(queues),
            "action_duration": duration # Truyền log thời gian ra ngoài
        }
        
        return np.array(next_state, dtype=np.float32), float(total_reward), terminated, truncated, info

    def close(self):
        if self.sumo_running:
            self.sumo_engine.close()
            self.sumo_running = False
