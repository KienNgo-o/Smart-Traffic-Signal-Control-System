import math

def calculate_webster_cycle(v_n, v_s, v_e, v_w, saturation_flow=1800, yellow_time=4, all_red_time=2):
    """
    Tính toán thời lượng Xanh tối ưu theo phương pháp Webster.
    [ĐÃ SỬA LỖI]: Chia đèn công bằng kể cả khi ngã tư quá tải.
    """
    # 1. Tìm lưu lượng tới hạn
    v_ns = max(v_n, v_s)
    v_ew = max(v_e, v_w)

    # 2. Tính tỷ lệ lưu lượng 
    y_ns = v_ns / saturation_flow
    y_ew = v_ew / saturation_flow
    
    # [SỬA Ở ĐÂY]: Lưu lại Y_raw để chia tỷ lệ sau này
    Y_raw = y_ns + y_ew
    Y_capped = min(Y_raw, 0.95) # Chặn an toàn để tính chu kỳ

    # 3. Tính thời gian mất mát
    num_phases = 2
    L = num_phases * (yellow_time + all_red_time)

    # 4. Tính chu kỳ tối ưu
    C_o = (1.5 * L + 5) / (1 - Y_capped)
    C_o = max(40.0, min(C_o, 120.0))

    # 5. Phân bổ thời gian xanh chuẩn xác
    total_green = int(round(C_o - L))

    # [SỬA Ở ĐÂY]: Chia theo Y_raw thay vì Y_capped
    g_ns = int(round((y_ns / Y_raw) * total_green))
    g_ew = total_green - g_ns

    return g_ns, g_ew, int(round(C_o))

def generate_tls_xml(g_ns, g_ew, filename="sumo/tls_webster.add.xml"):
    xml_content = f"""<additional>
    <tlLogic id="center" type="static" programID="webster_baseline" offset="0">
        <phase duration="{g_ns}" state="GGGggrrrrrGGGggrrrrr"/>
        <phase duration="4"      state="yyyyyrrrrryyyyyrrrrr"/>
        <phase duration="2"      state="rrrrrrrrrrrrrrrrrrrr"/>
        
        <phase duration="{g_ew}" state="rrrrrGGGggrrrrrGGGgg"/>
        <phase duration="4"      state="rrrrryyyyyrrrrryyyyy"/>
        <phase duration="2"      state="rrrrrrrrrrrrrrrrrrrr"/>
    </tlLogic>
</additional>
    """
    with open(filename, "w") as f:
        f.write(xml_content)
    print(f"[SUCCESS] Đã lưu lịch trình Webster vào {filename}")

if __name__ == "__main__":
    # =========================================================
    # [SỬA LỖI KHUNG GIỜ]: Môi trường mô phỏng giờ đây chạy 7200s.
    # Nửa đầu (3600s) là vắng, nửa sau (3600s) là kẹt xe.
    # Để đối đầu công bằng với AI (AI có khả năng tự thay đổi đèn), 
    # Webster phải lấy TRUNG BÌNH CỘNG của cả 2 khung giờ để cài đặt 1 chu kỳ chết.
    # =========================================================
    
    # Tính luồng quy đổi (Xe / Giờ) từ probability trong routes.xml
    flow_off_peak = int(0.05 * 3600) # = 180 xe/giờ
    flow_peak = int(0.30 * 3600)     # = 1080 xe/giờ
    
    # Lấy luồng đại diện (Average) để set cứng cho đèn tĩnh
    avg_flow = (flow_off_peak + flow_peak) // 2 # = 630 xe/giờ

    g_ns, g_ew, c_opt = calculate_webster_cycle(avg_flow, avg_flow, avg_flow, avg_flow)

    print("=== TÍNH TOÁN CHU KỲ TỐI ƯU WEBSTER ===")
    print(f"Lưu lượng trung bình: {avg_flow} xe/h/hướng")
    print(f"Chu kỳ tổng (Co)    : {c_opt} giây")
    print(f"Pha Xanh Bắc-Nam    : {g_ns} giây")
    print(f"Pha Xanh Đông-Tây   : {g_ew} giây")
    
    generate_tls_xml(g_ns, g_ew)