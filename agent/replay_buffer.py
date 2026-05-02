import numpy as np


class SumTree:
    """
    Cấu trúc dữ liệu cây nhị phân (Binary Tree) để truy vấn và cập nhật 
    độ ưu tiên (Priority) cực nhanh trong O(log N).
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # Mảng lưu trữ cây: Các node lá lưu priority, các node cha lưu tổng của 2 node con
        self.tree = np.zeros(2 * capacity - 1)
        # Mảng lưu trữ dữ liệu thật (Experience transition)
        self.data = np.zeros(capacity, dtype=object)
        
        self.write_ptr = 0
        self.n_entries = 0

    def update(self, tree_idx, priority):
        """Cập nhật priority và lan truyền tổng lên tận node gốc."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Lan truyền lên cha (Propagate)
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, priority, data):
        """Thêm dữ liệu mới vào node lá và cập nhật cây."""
        # data bây giờ sẽ là (state, action, reward, next_state, done, duration)
        tree_idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(tree_idx, priority)
        
        self.write_ptr += 1
        if self.write_ptr >= self.capacity:
            self.write_ptr = 0 # Ghi đè (Overwrite) khi buffer đầy
            
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, s):
        """Duyệt từ gốc xuống lá để tìm mẫu dựa trên giá trị ngẫu nhiên s."""
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # Đã đến node lá
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            
            # Nếu s nhỏ hơn tổng của nhánh trái, đi sang nhánh trái
            if s <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            # Ngược lại, trừ đi nhánh trái và đi sang nhánh phải
            else:
                s -= self.tree[left_child_idx]
                parent_idx = right_child_idx
                
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0] # Node gốc chứa tổng tất cả priority


class PrioritizedReplayBuffer:
    """
    Bộ đệm kinh nghiệm có ưu tiên (PER). 
    Sử dụng SumTree để lấy mẫu kinh nghiệm (Transitions).
    """
    def __init__(self, max_size, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.tree = SumTree(max_size)
        self.max_size = max_size
        self.alpha = alpha  # Mức độ sử dụng PER (0 = Random, 1 = Hoàn toàn ưu tiên)
        self.beta = beta    # Trọng số bù đắp Bias (Importance Sampling)
        self.beta_increment = beta_increment_per_sampling
        self.epsilon = 1e-5 # Đảm bảo priority không bao giờ bằng 0 (Luôn có cơ hội được lấy mẫu)
        self.max_priority = 1.0 # Priority mặc định cho mẫu mới

    def add(self, state, action, reward, next_state, done, duration):
        """Lưu trữ transition mới với mức ưu tiên cao nhất để đảm bảo nó được duyệt ít nhất 1 lần."""
        """Lưu thêm tham số duration của hành động vào bộ đệm."""
        experience = (state, action, reward, next_state, done, duration)
        self.tree.add(self.max_priority, experience)


    def sample(self, batch_size):
        """Lấy mẫu một batch dựa trên xác suất tỷ lệ thuận với độ ưu tiên."""
        batch = []
        idxs = []
        segment = self.tree.total_priority / batch_size
        priorities = []

        # Tăng dần beta tới 1.0 (Bù đắp bias tối đa vào cuối quá trình huấn luyện)
        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            
            idx, p, data = self.tree.get_leaf(s)
            idxs.append(idx)
            priorities.append(p)
            batch.append(data)

        # 1. Bóc tách dữ liệu (Thêm duration vào)
        states, actions, rewards, next_states, dones, durations = zip(*batch)
        
        # 2. Tính toán Importance Sampling (IS) weights
        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max() # Chuẩn hóa để tránh nổ gradient
        
        return (states, actions, rewards, next_states, dones, durations), idxs, is_weights

    def update_priorities(self, idxs, td_errors):
        """Cập nhật lại priority trên cây dựa trên sai số dự đoán mới nhất."""
        for idx, td_error in zip(idxs, td_errors):
            # Tính priority mới: p_i = (|TD_Error| + epsilon)^alpha
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)