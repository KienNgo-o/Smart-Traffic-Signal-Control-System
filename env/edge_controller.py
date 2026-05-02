import numpy as np

class EdgeController:
    """
    Bộ điều khiển biên (Edge Actuator) quản lý vòng lặp vật lý và an toàn cấp thấp.
    Đã sửa lỗi ghi đè string và đồng bộ Step.
    """
    def __init__(self, tls_id, engine, yellow_time=4, all_red_time=2, min_green=10):
        self.tls_id = tls_id
        self.engine = engine 
        
        self.yellow_time = yellow_time
        self.all_red_time = all_red_time
        self.min_green = min_green
        
        self.lanes = []
        self.num_phases = 0
        self.action_duration = 0 # Đếm tổng số giây mô phỏng đã trôi qua sau 1 action

    def setup(self):
        """Khởi tạo danh sách làn và quét chính xác số pha của chương trình đang chạy."""
        raw_lanes = self.engine.trafficlight.getControlledLanes(self.tls_id)
        self.lanes = sorted(list(set(raw_lanes)))
        
        # [ĐÃ SỬA]: Tìm ID của chương trình đèn đang hoạt động
        active_program = self.engine.trafficlight.getProgram(self.tls_id)
        logics = self.engine.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)
        
        # Duyệt qua các chương trình để lấy đúng số pha của Active Program
        for logic in logics:
            if logic.programID == active_program:
                self.num_phases = len(logic.phases)
                break

    def get_state(self):
        queues = []
        waits = []
        
        for lane in self.lanes:
            # ==========================================================
            # [SỬA LỖI HIỆU NĂNG]: Dùng API cấp độ Lane thay vì Vehicle
            # Tính toán trực tiếp bằng lõi C++ của SUMO, nhanh hơn hàng trăm lần.
            # ==========================================================
            q = self.engine.lane.getLastStepHaltingNumber(lane) # Đếm xe dừng (<0.1 m/s)
            w = self.engine.lane.getWaitingTime(lane)           # Tổng thời gian chờ
            
            queues.append(q)
            waits.append(w)
            
        current_phase = self.engine.trafficlight.getPhase(self.tls_id)
        
        # Tiêm nhiễu Gaussian
        noise = np.random.normal(0, 0.5, len(queues))
        noisy_queues = np.clip(np.array(queues) + noise, 0, None).astype(np.float32)
        
        state = np.concatenate((noisy_queues, np.array(waits, dtype=np.float32), [float(current_phase)]))
        return state

    def _block_and_step(self, duration):
        """Tịnh tiến môi trường N giây."""
        for _ in range(int(duration)):
            self.engine.simulationStep()
            self.action_duration += 1 # Tracking tổng thời gian

    def apply_action(self, action):
        """
        Phiên dịch Action nhị phân thành chuỗi lệnh điều khiển an toàn 6 pha.
        """
        self.action_duration = 0 # Reset bộ đếm thời gian thực thi
        current_phase = self.engine.trafficlight.getPhase(self.tls_id)
        
        # [SỬA LỖI SMDP]: Khóa cứng thời gian thực thi của một Step
        # Tổng = Vàng (4s) + Đỏ (2s) + Xanh tối thiểu (10s) = 16 giây
        fixed_step_duration = self.yellow_time + self.all_red_time + self.min_green
        
        # CHUYỂN LUỒNG (Action = 1) VÀ chỉ thực thi khi đang ở pha Xanh
        if action == 1 and current_phase in [0, 3]:
            
            # Bước 1: Pha Vàng (Cảnh báo dừng)
            yellow_phase = current_phase + 1
            self.engine.trafficlight.setPhase(self.tls_id, yellow_phase)
            self._block_and_step(self.yellow_time)
            
            # Bước 2: Pha All-Red (Dọn dẹp ngã tư, chống va chạm đuôi)
            all_red_phase = current_phase + 2
            self.engine.trafficlight.setPhase(self.tls_id, all_red_phase)
            self._block_and_step(self.all_red_time)
            
            # Bước 3: Pha Xanh cho luồng tiếp theo & Khóa thời gian Min Green
            green_phase = (current_phase + 3) % self.num_phases
            self.engine.trafficlight.setPhase(self.tls_id, green_phase)
            self._block_and_step(self.min_green)
            
        # GIỮ LUỒNG (Action = 0) hoặc bỏ qua lệnh
        else:
            # [SỬA Ở ĐÂY]: Tịnh tiến mô phỏng ĐÚNG 16 giây để cân bằng hoàn toàn 
            # với khoảng thời gian của lệnh Chuyển luồng, đưa bài toán về MDP chuẩn.
            self._block_and_step(fixed_step_duration)
            
        return self.action_duration