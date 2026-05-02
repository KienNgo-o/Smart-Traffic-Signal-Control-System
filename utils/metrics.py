import xml.etree.ElementTree as ET
import numpy as np
import os
from typing import Dict

def evaluate_tripinfo(tripinfo_file: str) -> Dict[str, float]:
    """
    Trích xuất Thời gian chờ, Tổng thời gian di chuyển (Duration) và Thông lượng.
    """
    if not os.path.exists(tripinfo_file):
        print(f"[Cảnh báo] Không tìm thấy file {tripinfo_file}")
        return {"avg_waiting_time": 0.0, "avg_travel_time": 0.0, "throughput": 0}

    waiting_times = []
    travel_times = []
    throughput = 0

    try:
        context = ET.iterparse(tripinfo_file, events=("end",))
        for event, elem in context:
            if elem.tag == 'tripinfo':
                throughput += 1
                waiting_times.append(float(elem.get('waitingTime', 0.0)))
                
                # Trích xuất 'duration' (Tổng thời gian chuyến đi)
                travel_times.append(float(elem.get('duration', 0.0)))
                
                elem.clear()
                
        avg_wait = float(np.mean(waiting_times)) if throughput > 0 else 0.0
        avg_travel = float(np.mean(travel_times)) if throughput > 0 else 0.0
        
        return {
            "avg_waiting_time": avg_wait,
            "avg_travel_time": avg_travel, # Đã sửa thành travel_time
            "throughput": throughput
        }
    except Exception as e:
        print(f"[Lỗi] Trích xuất dữ liệu XML thất bại: {e}")
        return {"avg_waiting_time": 0.0, "avg_travel_time": 0.0, "throughput": 0}

def evaluate_queue(detector_file: str) -> float:
    """
    Trích xuất Chiều dài hàng đợi trung bình (Queue Length) từ file của camera cảm biến.
    """
    if not os.path.exists(detector_file):
        return 0.0
        
    queues = []
    try:
        context = ET.iterparse(detector_file, events=("end",))
        for event, elem in context:
            if elem.tag == 'interval':
                # Lấy chiều dài hàng đợi tối đa trung bình trong mỗi chu kỳ quét
                q_len = float(elem.get('meanMaxJamLengthInVehicles', 0.0))
                queues.append(q_len)
                elem.clear()
                
        return float(np.mean(queues)) if queues else 0.0
    except:
        return 0.0

def evaluate_tripinfo_advanced(tripinfo_file: str, sim_duration: float = 7200.0) -> Dict[str, float]:
    """
    Extract robust evaluation metrics from SUMO tripinfo output.

    Main throughput counts completed trips only. When
    --tripinfo-output.write-unfinished is enabled, unfinished vehicles are
    reported separately as unfinished_trips instead of inflating throughput.
    """
    if not os.path.exists(tripinfo_file):
        print(f"[Cảnh báo] Không tìm thấy file {tripinfo_file}")
        return {
            "throughput": 0,
            "unfinished_trips": 0,
            "reported_trips": 0,
            "throughput_per_hour": 0.0,
            "avg_waiting_time": 0.0,
            "p95_waiting_time": 0.0,
            "avg_time_loss": 0.0,
            "avg_travel_time": 0.0,
            "avg_depart_delay": 0.0,
            "avg_stop_count": 0.0,
            "jain_fairness": 0.0,
            "lane_wait_std": 0.0,
            "total_fuel_mg": 0.0,
            "total_co2_mg": 0.0,
            "total_nox_mg": 0.0,
            "avg_co2_mg_per_completed_vehicle": 0.0,
            "avg_fuel_mg_per_completed_vehicle": 0.0,
        }

    def safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    completed_waits, completed_losses, completed_durations = [], [], []
    completed_depart_delays, completed_stop_counts = [], []
    all_waits = []
    lane_waits = {}

    total_emissions = {
        "fuel_abs": 0.0,
        "CO2_abs": 0.0,
        "NOx_abs": 0.0,
        "CO_abs": 0.0,
        "HC_abs": 0.0,
        "PMx_abs": 0.0,
    }
    completed_emissions = {key: 0.0 for key in total_emissions}

    completed_trips = 0
    unfinished_trips = 0
    reported_trips = 0

    try:
        for _, elem in ET.iterparse(tripinfo_file, events=("end",)):
            if elem.tag != "tripinfo":
                continue

            reported_trips += 1
            arrival = safe_float(elem.get("arrival"), default=-1.0)
            completed = arrival >= 0.0

            wait = safe_float(elem.get("waitingTime"))
            all_waits.append(wait)

            emission_values = {}
            em = elem.find("emissions")
            if em is not None:
                for key in total_emissions:
                    value = safe_float(em.get(key))
                    emission_values[key] = value
                    total_emissions[key] += value

            if completed:
                completed_trips += 1
                completed_waits.append(wait)
                completed_losses.append(safe_float(elem.get("timeLoss")))
                completed_durations.append(safe_float(elem.get("duration")))
                completed_depart_delays.append(safe_float(elem.get("departDelay")))
                completed_stop_counts.append(safe_float(elem.get("waitingCount")))

                lane = elem.get("departLane", "unknown")
                lane_waits.setdefault(lane, []).append(wait)

                for key, value in emission_values.items():
                    completed_emissions[key] += value
            else:
                unfinished_trips += 1

            elem.clear()

        lane_avg_waits = np.array([np.mean(v) for v in lane_waits.values() if v], dtype=np.float64)
        service = 1.0 / (lane_avg_waits + 1.0)
        denominator = len(service) * np.sum(service ** 2)
        fairness = float((service.sum() ** 2) / denominator) if denominator > 0 else 0.0

        return {
            "throughput": completed_trips,
            "unfinished_trips": unfinished_trips,
            "reported_trips": reported_trips,
            "throughput_per_hour": completed_trips / max(sim_duration / 3600.0, 1e-9),
            "avg_waiting_time": float(np.mean(completed_waits)) if completed_waits else 0.0,
            "p95_waiting_time": float(np.percentile(completed_waits, 95)) if completed_waits else 0.0,
            "avg_all_reported_waiting_time": float(np.mean(all_waits)) if all_waits else 0.0,
            "avg_time_loss": float(np.mean(completed_losses)) if completed_losses else 0.0,
            "avg_travel_time": float(np.mean(completed_durations)) if completed_durations else 0.0,
            "avg_depart_delay": float(np.mean(completed_depart_delays)) if completed_depart_delays else 0.0,
            "avg_stop_count": float(np.mean(completed_stop_counts)) if completed_stop_counts else 0.0,
            "jain_fairness": fairness,
            "lane_wait_std": float(np.std(lane_avg_waits)) if len(lane_avg_waits) else 0.0,
            "total_fuel_mg": total_emissions["fuel_abs"],
            "total_co2_mg": total_emissions["CO2_abs"],
            "total_nox_mg": total_emissions["NOx_abs"],
            "completed_fuel_mg": completed_emissions["fuel_abs"],
            "completed_co2_mg": completed_emissions["CO2_abs"],
            "completed_nox_mg": completed_emissions["NOx_abs"],
            "avg_co2_mg_per_completed_vehicle": completed_emissions["CO2_abs"] / max(completed_trips, 1),
            "avg_fuel_mg_per_completed_vehicle": completed_emissions["fuel_abs"] / max(completed_trips, 1),
            "avg_nox_mg_per_completed_vehicle": completed_emissions["NOx_abs"] / max(completed_trips, 1),
        }
    except Exception as e:
        print(f"[Lỗi] Trích xuất dữ liệu XML nâng cao thất bại: {e}")
        return {}
