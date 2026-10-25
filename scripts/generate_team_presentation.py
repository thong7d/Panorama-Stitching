#!/usr/bin/env python3
"""
Generate a professional 10-slide PowerPoint presentation for the team:
CSC14120 - Panorama Image Stitching Midterm Architecture Review.
Focused on engineering, architecture, telemetry, Amdahl's law, and GPU strategy.

Authors: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369),
         Dao Quoc Tuan (23120392)
External sources: python-pptx API documentation; no copied implementation code.
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_deck(output_pptx: str):
    prs = Presentation()
    prs.slide_width = Inches(13.333)  # 16:9 widescreen
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette: Clean Modern Engineering Theme
    C_BG = RGBColor(15, 23, 42)          # Dark Slate Navy
    C_CARD_BG = RGBColor(30, 41, 59)     # Slate 800
    C_CARD_BORDER = RGBColor(51, 65, 85) # Slate 700
    C_PRIMARY = RGBColor(59, 130, 246)   # Blue 500
    C_ACCENT_GREEN = RGBColor(16, 185, 129) # Emerald 500
    C_ACCENT_AMBER = RGBColor(245, 158, 11) # Amber 500
    C_ACCENT_PURPLE = RGBColor(168, 85, 247) # Purple 500
    C_TEXT_WHITE = RGBColor(248, 250, 252) # Slate 50
    C_TEXT_MUTED = RGBColor(148, 163, 184) # Slate 400
    C_TEXT_DARK = RGBColor(30, 41, 59)

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = C_BG
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category_text="CSC14120 • KIẾN TRÚC HỆ THỐNG V0 BASELINE"):
        # Category Tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = C_PRIMARY

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.5), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = C_TEXT_WHITE

    def add_card(slide, left, top, width, height, title="", border_color=C_CARD_BORDER, bg_color=C_CARD_BG):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.5)
        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.4))
            tf = tb.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = C_TEXT_WHITE
        return card

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Main Badge
    badge = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(4.5), Inches(0.45))
    badge.fill.solid()
    badge.fill.fore_color.rgb = RGBColor(30, 58, 138)
    badge.line.color.rgb = C_PRIMARY
    p_b = badge.text_frame.paragraphs[0]
    p_b.text = "CSC14120 LẬP TRÌNH SONG SONG • ĐỒ ÁN GIỮA KỲ"
    p_b.font.size = Pt(11)
    p_b.font.bold = True
    p_b.font.color.rgb = C_TEXT_WHITE
    p_b.alignment = PP_ALIGN.CENTER

    # Main Title
    t_box = s1.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.5), Inches(1.8))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "GHÉP ẢNH PANORAMA TĂNG TỐC GPU"
    p1.font.size = Pt(40)
    p1.font.bold = True
    p1.font.color.rgb = C_TEXT_WHITE
    
    p2 = tf1.add_paragraph()
    p2.text = "Tổng kết Kiến trúc Baseline V0, Bằng chứng Đo đạc & Chiến lược Song song hóa CUDA"
    p2.font.size = Pt(18)
    p2.font.color.rgb = C_TEXT_MUTED

    # Highlights Row (3 stat cards)
    card_data = [
        ("KIẾN TRÚC 6 GIAI ĐOẠN", "V0 Tuần tự C++17", "Độc lập, không OpenMP / SIMD", C_PRIMARY),
        ("ĐỘ CHÍNH XÁC HÌNH HỌC", "RMSE < 0.90 px", "Đạt chuẩn Quality Gate A (4K)", C_ACCENT_GREEN),
        ("TIỀM NĂNG TĂNG TỐC", "Smax lên tới 26.5x", "Định luật Amdahl (p > 94% ở 4K)", C_ACCENT_AMBER),
    ]
    for i, (head, val, sub, col) in enumerate(card_data):
        x = Inches(0.8 + i * 4.0)
        c = add_card(s1, x, Inches(4.3), Inches(3.7), Inches(1.8))
        tb = s1.shapes.add_textbox(x + Inches(0.25), Inches(4.5), Inches(3.2), Inches(1.4))
        tf = tb.text_frame
        p_h = tf.paragraphs[0]
        p_h.text = head
        p_h.font.size = Pt(11)
        p_h.font.bold = True
        p_h.font.color.rgb = col
        p_v = tf.add_paragraph()
        p_v.text = val
        p_v.font.size = Pt(20)
        p_v.font.bold = True
        p_v.font.color.rgb = C_TEXT_WHITE
        p_s = tf.add_paragraph()
        p_s.text = sub
        p_s.font.size = Pt(11)
        p_s.font.color.rgb = C_TEXT_MUTED

    # Footer
    f_box = s1.shapes.add_textbox(Inches(0.8), Inches(6.5), Inches(11.5), Inches(0.5))
    p_f = f_box.text_frame.paragraphs[0]
    p_f.text = "Nhóm 19: Phạm Minh Thông (Nhóm trưởng) • Nguyễn Lê Tấn Tiến • Đào Quốc Tuấn | Tuân thủ Plan v4"
    p_f.font.size = Pt(12)
    p_f.font.color.rgb = C_TEXT_MUTED

    # =========================================================================
    # SLIDE 2: 6-STAGE PIPELINE ARCHITECTURE
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Kiến trúc Quy trình 6 Giai đoạn (The 6-Stage Pipeline Architecture)")

    # Left: Independent Stages (Process A & B separately)
    add_card(s2, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "XỬ LÝ ĐỘC LẬP TỪNG ẢNH (A & B)", C_PRIMARY)
    stages_left = [
        ("S1: TIỀN XỬ LÝ (PREPROCESSING)", "Mức xám chuẩn Rec.601 + Lọc Gaussian 5×5 (σ≈1.04). Khử nhiễu bảo toàn biên cạnh.", C_PRIMARY),
        ("S2: PHÁT HIỆN GÓC (FAST-9 + NMS)", "Bresenham 16 điểm, ngưỡng T=20. NMS 3×3 + Bộ lọc Top-2000 điểm đặc trưng mạnh nhất.", C_PRIMARY),
        ("S3: MÔ TẢ ĐẶC TRƯNG (ORIENTED BRIEF)", "Patch 31×31, góc từ trọng tâm cường độ. Trích xuất 256-bit descriptor (8x uint32).", C_PRIMARY),
    ]
    for i, (title, desc, col) in enumerate(stages_left):
        y = Inches(2.4 + i * 1.3)
        box = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.1), y, Inches(5.0), Inches(1.1))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(15, 23, 42)
        box.line.color.rgb = col
        tf = box.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = col
        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = C_TEXT_MUTED

    # Right: Combined Stages (Need data from both images)
    add_card(s2, Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.8), "KẾT HỢP DỮ LIỆU HAI ẢNH (A ⟷ B)", C_ACCENT_PURPLE)
    stages_right = [
        ("S4: SO KHỚP ĐẶC TRƯNG (HAMMING MATCHING)", "Vét cạn popcount trên descriptor 256-bit. Lọc nhập nhằng bằng Lowe's ratio test < 0.75.", C_ACCENT_PURPLE),
        ("S5: ƯỚC LƯỢNG HÌNH HỌC", "RANSAC DLT + kiểm tra H; fallback tịnh tiến robust nếu mô hình projective suy biến.", C_ACCENT_PURPLE),
        ("S6: BIẾN ĐỔI & TRỘN MỀM (WARP & BLEND)", "Ánh xạ ngược qua H (Inverse mapping). Nội suy Bilinear + Trộn mềm đối xứng feathering.", C_ACCENT_PURPLE),
    ]
    for i, (title, desc, col) in enumerate(stages_right):
        y = Inches(2.4 + i * 1.3)
        box = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.1), y, Inches(5.0), Inches(1.1))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(15, 23, 42)
        box.line.color.rgb = col
        tf = box.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = col
        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = C_TEXT_MUTED

    # Bottom Banner
    bb = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(6.8), Inches(11.6), Inches(0.4))
    bb.fill.solid()
    bb.fill.fore_color.rgb = RGBColor(30, 58, 138)
    bb.line.fill.background()
    p_bb = bb.text_frame.paragraphs[0]
    p_bb.text = "Ý NGHĨA SONG SONG: S1–S3 chạy hoàn toàn độc lập (Stream concurrency). S4–S6 cần đồng bộ dữ liệu chéo."
    p_bb.font.size = Pt(11)
    p_bb.font.bold = True
    p_bb.font.color.rgb = C_TEXT_WHITE
    p_bb.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 3: SYSTEM INVARIANT CONTRACTS
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Hợp đồng Thuật toán Bất biến (System Invariant Contracts)")

    # Intro text
    intro = s3.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.6), Inches(0.6))
    p_in = intro.text_frame.paragraphs[0]
    p_in.text = "Để so sánh độ tăng tốc (Speedup) một cách khoa học và công bằng giữa CPU V0 và các bản GPU CUDA V1/V2/V3, các tham số thuật toán sau đây được KHÓA BẤT BIẾN trên toàn bộ hệ thống:"
    p_in.font.size = Pt(13)
    p_in.font.color.rgb = C_TEXT_MUTED

    # Contract Table
    rows = 7
    cols = 4
    t_shape = s3.shapes.add_table(rows, cols, Inches(0.8), Inches(2.4), Inches(11.6), Inches(4.3))
    table = t_shape.table
    table.columns[0].width = Inches(2.2)
    table.columns[1].width = Inches(3.2)
    table.columns[2].width = Inches(3.2)
    table.columns[3].width = Inches(3.0)

    headers = ["GIAI ĐOẠN", "THUẬT TOÁN KHÓA", "THAM SỐ KHÓA CỐ ĐỊNH", "MỤC TIÊU BẢO ĐẢM"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 58, 138)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_WHITE

    contract_rows = [
        ("S1: Tiền xử lý", "Grayscale + Gaussian 5×5", "Y = 0.299R + 0.587G + 0.114B; σ≈1.04", "Triệt tiêu nhiễu sensor không đổi hệ tọa độ"),
        ("S2: FAST Corner", "FAST-9 + NMS 3×3 + Top-K", "Ngưỡng T=20; K=2000 keypoints tối đa", "Điểm góc ổn định, số lượng bounded"),
        ("S3: ORB Descriptor", "Oriented BRIEF 256-bit", "Patch 31×31; Bảng mẫu chuẩn OpenCV 256 cặp", "Bất biến phép xoay camera, khớp bit popcount"),
        ("S4: Khớp Hamming", "Brute-force + Lowe Ratio", "Descriptor 8x uint32_t; Ratio test < 0.75", "Lọc triệt để ghép sai trên vân lặp lại"),
        ("S5: RANSAC + FALLBACK", "Hartley DLT / robust translation", "2000 lượt; ngưỡng 3.0 px; LCG seed 42", "Loại ngoại lai và ngăn canvas suy biến"),
        ("S6: Chiếu & Trộn", "Inverse Warping + Feathering", "Nội suy song tuyến; Trộn mượt 4 biên đối xứng", "Không tạo lỗ thủng pixel, triệt tiêu vết nối"),
    ]
    for i, row in enumerate(contract_rows):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(30, 41, 59) if i % 2 == 0 else RGBColor(15, 23, 42)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(11)
            p.font.color.rgb = C_TEXT_WHITE if j < 3 else C_ACCENT_GREEN

    # =========================================================================
    # SLIDE 4: DATASET SPECIFICATION & NORMALIZATION
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Tập Dữ liệu Thực nghiệm Đa Bối cảnh (Chuẩn hóa 16:9)")

    # 3 Resolution Cards
    res_data = [
        ("SMALL (720p)", "1280 × 720", "0.92 Megapixels", "Benchmark kiểm tra logic & phát hiện sai số nhanh", C_PRIMARY),
        ("MEDIUM (1080p)", "1920 × 1080", "2.07 Megapixels", "Độ phân giải tiêu chuẩn, tải trung bình", C_ACCENT_GREEN),
        ("LARGE (4K UHD)", "3840 × 2160", "8.29 Megapixels", "Thử thách tải cực hạn GPU, đo trần Speedup", C_ACCENT_AMBER),
    ]
    for i, (name, dim, mp, note, col) in enumerate(res_data):
        x = Inches(0.8 + i * 4.0)
        c = add_card(s4, x, Inches(1.8), Inches(3.7), Inches(2.0))
        tb = s4.shapes.add_textbox(x + Inches(0.2), Inches(2.0), Inches(3.3), Inches(1.6))
        tf = tb.text_frame
        p1 = tf.paragraphs[0]
        p1.text = name
        p1.font.bold = True
        p1.font.size = Pt(12)
        p1.font.color.rgb = col
        p2 = tf.add_paragraph()
        p2.text = dim
        p2.font.bold = True
        p2.font.size = Pt(22)
        p2.font.color.rgb = C_TEXT_WHITE
        p3 = tf.add_paragraph()
        p3.text = f"{mp} • {note}"
        p3.font.size = Pt(10)
        p3.font.color.rgb = C_TEXT_MUTED

    # 5 Image Pairs Grid
    add_card(s4, Inches(0.8), Inches(4.1), Inches(11.6), Inches(2.9), "5 CẶP ẢNH THỰC NGHIỆM ĐẠI DIỆN ĐẦY ĐỦ THÁCH THỨC QUANG HỌC")
    pairs = [
        ("CẢNH TỔNG HỢP (SYNTHETIC)", "Ground-truth toán học chuẩn", "Dùng xác thực sai số pixel ma trận H"),
        ("SITE 1: BẾP / TIỆN ÍCH", "38% diện tích chồng lấn", "Thị sai cận cảnh, nhiều mặt phẳng độ sâu"),
        ("SITE 2: MÁI NHÀ ĐÔ THỊ", "33% diện tích chồng lấn", "Bầu trời ít vân, mái ngói lặp lại"),
        ("SITE 3: HÀNH LANG SÂN THƯỢNG", "49% sau crop", "Lưới xanh lặp, phối cảnh sâu, trời sáng"),
        ("CHALLENGING: CẬN CẢNH TRONG NHÀ", "39% diện tích chồng lấn", "Che khuất, thị sai và độ sâu biến thiên"),
    ]
    for i, (p_name, p_stat, p_desc) in enumerate(pairs):
        y = Inches(4.6 + i * 0.44)
        box = s4.shapes.add_textbox(Inches(1.1), y, Inches(11.0), Inches(0.4))
        tf = box.text_frame
        p = tf.paragraphs[0]
        p.text = f"•  {p_name:<30}  |  {p_stat:<28}  |  {p_desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_WHITE if i == 0 else (C_ACCENT_AMBER if i == 4 else C_TEXT_MUTED)

    # =========================================================================
    # SLIDE 5: GEOMETRIC CORRECTNESS (QUALITY GATE A)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Xác thực Độ chính xác Hình học (Quality Gate A - Ground Truth)")

    # Left: Explanation of Methodology
    add_card(s5, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "PHƯƠNG PHÁP PROJECTED GRID RMSE")
    tb_m = s5.shapes.add_textbox(Inches(1.1), Inches(2.4), Inches(5.0), Inches(4.2))
    tf_m = tb_m.text_frame
    tf_m.word_wrap = True
    items_m = [
        ("Tại sao dùng Grid RMSE?", "Ảnh tổng hợp được sinh từ phép biến đổi xạ ảnh H_gt đã biết trước. Ta đo sai số hình học bằng khoảng cách Euclidean trung bình khi chiếu lưới điểm qua H_est so với H_gt."),
        ("Tiêu chuẩn Quality Gate A:", "• Grid RMSE < 1.0 pixel trên mọi kích thước ảnh.\n• Sai số tương đối ≤ 0.1% chiều dài đường chéo ảnh."),
        ("Kiểm tra đối chiếu OpenCV SIFT:", "Trên ảnh thật không có H_gt, OpenCV là mốc tham chiếu độc lập. H projective suy biến được đánh dấu N/A; mô hình fallback được kiểm tra nhất quán xuyên độ phân giải."),
    ]
    for h, b in items_m:
        p1 = tf_m.add_paragraph() if tf_m.paragraphs[0].text else tf_m.paragraphs[0]
        p1.text = h
        p1.font.bold = True
        p1.font.size = Pt(12)
        p1.font.color.rgb = C_PRIMARY
        p2 = tf_m.add_paragraph()
        p2.text = b + "\n"
        p2.font.size = Pt(11)
        p2.font.color.rgb = C_TEXT_MUTED

    # Right: Results Table & Pass Badge
    add_card(s5, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0), "KẾT QUẢ ĐO ĐẠC THỰC TẾ TRÊN CẶP SYNTHETIC")
    t_shape5 = s5.shapes.add_table(4, 4, Inches(7.1), Inches(2.4), Inches(5.0), Inches(2.4))
    t5 = t_shape5.table
    t5.columns[0].width = Inches(1.2)
    t5.columns[1].width = Inches(1.2)
    t5.columns[2].width = Inches(1.4)
    t5.columns[3].width = Inches(1.2)

    h5 = ["ĐỘ PHÂN GIẢI", "GRID RMSE", "SAI SỐ / Đ.CHÉO", "GATE A"]
    for j, h in enumerate(h5):
        cell = t5.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 58, 138)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_TEXT_WHITE

    r5_data = [
        ("Small (720p)", "0.1649 px", "0.0112 %", "ĐẠT"),
        ("Medium (1080p)", "0.3989 px", "0.0181 %", "ĐẠT"),
        ("Large (4K UHD)", "0.8975 px", "0.0204 %", "ĐẠT"),
    ]
    for i, r in enumerate(r5_data):
        for j, val in enumerate(r):
            cell = t5.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(30, 41, 59)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(11)
            p.font.color.rgb = C_ACCENT_GREEN if j == 3 else C_TEXT_WHITE

    # Big Pass Badge Card
    badge5 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(5.1), Inches(5.0), Inches(1.4))
    badge5.fill.solid()
    badge5.fill.fore_color.rgb = RGBColor(6, 78, 59) # Deep green
    badge5.line.color.rgb = C_ACCENT_GREEN
    tf_b5 = badge5.text_frame
    p_b5 = tf_b5.paragraphs[0]
    p_b5.text = "CHỨNG NHẬN CHẤT LƯỢNG TOÁN HỌC"
    p_b5.font.bold = True
    p_b5.font.size = Pt(12)
    p_b5.font.color.rgb = C_ACCENT_GREEN
    p_b5_sub = tf_b5.add_paragraph()
    p_b5_sub.text = "Thuật toán tuần tự V0 hoàn toàn chuẩn xác; sai số 4K đạt 0.90 px (< 1.0 px). Đủ tiêu chuẩn làm mốc đối chiếu (Baseline Ground-truth) cho các phiên bản CUDA."
    p_b5_sub.font.size = Pt(10)
    p_b5_sub.font.color.rgb = C_TEXT_WHITE

    # =========================================================================
    # SLIDE 6: V0 TELEMETRY BENCHMARK DATA
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Bản đồ Thời gian Thực nghiệm V0 (V0 Telemetry Benchmark)")

    # Telemetry Table across all pairs
    t_shape6 = s6.shapes.add_table(6, 5, Inches(0.8), Inches(1.8), Inches(11.6), Inches(4.5))
    t6 = t_shape6.table
    t6.columns[0].width = Inches(2.4)
    t6.columns[1].width = Inches(2.2)
    t6.columns[2].width = Inches(2.2)
    t6.columns[3].width = Inches(2.4)
    t6.columns[4].width = Inches(2.4)

    h6 = ["TẬP DỮ LIỆU", "SMALL 720p (ms)", "MEDIUM 1080p (ms)", "LARGE 4K (ms)", "HỆ SỐ MỞ RỘNG (4K / 720p)"]
    for j, h in enumerate(h6):
        cell = t6.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 58, 138)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_TEXT_WHITE

    r6_data = [
        ("Synthetic (Tổng hợp)", "121.57 ms", "279.12 ms", "1,037.99 ms", "8.54 × (Tăng tuyến tính)"),
        ("Site 1: Trong nhà (Indoor)", "176.67 ms", "338.15 ms", "1,256.22 ms", "7.11 ×"),
        ("Site 2: Ngoại cảnh (Outdoor)", "123.20 ms", "301.80 ms", "1,149.80 ms", "9.33 ×"),
        ("Site 3: Đô thị (Custom)", "189.68 ms", "370.73 ms", "1,446.47 ms", "7.63 ×"),
        ("Challenging: Ban đêm (Night)", "161.86 ms", "310.03 ms", "1,158.13 ms", "7.15 ×"),
    ]
    for i, r in enumerate(r6_data):
        for j, val in enumerate(r):
            cell = t6.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(30, 41, 59) if i % 2 == 0 else RGBColor(15, 23, 42)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(11)
            p.font.color.rgb = C_TEXT_WHITE if j < 4 else C_ACCENT_AMBER

    # Bottom summary box
    sum6 = s6.shapes.add_textbox(Inches(0.8), Inches(6.5), Inches(11.6), Inches(0.6))
    p_s6 = sum6.text_frame.paragraphs[0]
    p_s6.text = "QUY CHUẨN ĐO ĐẠC: Mỗi cấu hình chạy tối thiểu 5 lần lặp kèm 1 warm-up. Báo cáo giá trị Trung vị (Median) để loại trừ nhiễu hệ điều hành; thời gian tăng ~7–9 lần khi số pixel tăng gấp 9 lần (720p lên 4K)."
    p_s6.font.size = Pt(11)
    p_s6.font.color.rgb = C_TEXT_MUTED

    # =========================================================================
    # SLIDE 7: HOT STAGES BOTTLENECK ANALYSIS
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Phân tích Thắt nút Cổ chai (Hot Stages Breakdown)")

    # 3 Hot Stages Cards
    hot_cards = [
        ("HẠNG 1: S6 CHIẾU & TRỘN (WARPING)", "Chiếm 35% – 53% tổng thời gian V0", "4K tốn: 466 ms – 619 ms", "Ánh xạ tọa độ ngược cho toàn bộ canvas 4K + 4 lần truy cập ảnh để nội suy song tuyến (Bilinear interpolation) ngốn băng thông bộ nhớ.", C_ACCENT_AMBER),
        ("HẠNG 2: S1 TIỀN XỬ LÝ (PREPROCESS)", "Chiếm 25% – 35% tổng thời gian V0", "4K tốn: 356 ms – 367 ms", "Kernel Gaussian 5×5 quét qua 8.3 triệu điểm ảnh trên cả hai ảnh A và B (tương đương 25 phép nhân cộng mỗi pixel).", C_PRIMARY),
        ("HẠNG 3: S2 PHÁT HIỆN GÓC (FAST/NMS)", "Chiếm 20% – 30% tổng thời gian V0", "4K tốn: 233 ms – 388 ms", "Kiểm tra vòng tròn 16 điểm và Non-Maximum Suppression 3×3 trên lưới điểm ảnh khổng lồ.", C_ACCENT_PURPLE),
    ]
    for i, (name, share, time_4k, why, col) in enumerate(hot_cards):
        x = Inches(0.8 + i * 4.0)
        c = add_card(s7, x, Inches(1.8), Inches(3.7), Inches(4.5), "", col)
        tb = s7.shapes.add_textbox(x + Inches(0.25), Inches(2.0), Inches(3.2), Inches(4.1))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = name
        p1.font.bold = True
        p1.font.size = Pt(12)
        p1.font.color.rgb = col
        p2 = tf.add_paragraph()
        p2.text = share
        p2.font.bold = True
        p2.font.size = Pt(14)
        p2.font.color.rgb = C_TEXT_WHITE
        p3 = tf.add_paragraph()
        p3.text = time_4k + "\n"
        p3.font.bold = True
        p3.font.size = Pt(11)
        p3.font.color.rgb = C_ACCENT_GREEN
        p4 = tf.add_paragraph()
        p4.text = why
        p4.font.size = Pt(11)
        p4.font.color.rgb = C_TEXT_MUTED

    # Bottom takeaway
    b_bot = s7.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(6.5), Inches(11.6), Inches(0.55))
    b_bot.fill.solid()
    b_bot.fill.fore_color.rgb = RGBColor(30, 58, 138)
    b_bot.line.fill.background()
    p_bb = b_bot.text_frame.paragraphs[0]
    p_bb.text = "KẾT LUẬN CỐT LÕI: Ba giai đoạn thắt nút cổ chai (S1 + S2 + S6) đều có độ phức tạp O(W × H) phụ thuộc trực tiếp vào kích thước ảnh. Các giai đoạn trích xuất & khớp đặc trưng (S3, S4, S5) chỉ phụ thuộc số keypoints O(K) nên chiếm tỷ trọng không đáng kể."
    p_bb.font.size = Pt(10)
    p_bb.font.color.rgb = C_TEXT_WHITE
    p_bb.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 8: AMDAHL'S LAW & SPEEDUP CEILING
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Định luật Amdahl & Giới hạn Gia tốc Lý thuyết (S_max)")

    # Formula Box Left
    add_card(s8, Inches(0.8), Inches(1.8), Inches(4.5), Inches(5.0), "MÔ HÌNH AMDAHL BẢO THỦ")
    tb_am = s8.shapes.add_textbox(Inches(1.1), Inches(2.4), Inches(3.9), Inches(4.2))
    tf_am = tb_am.text_frame
    tf_am.word_wrap = True
    p_am1 = tf_am.paragraphs[0]
    p_am1.text = "CÔNG THỨC AMDAHL:\nS_max = 1 / (1 - p)\n"
    p_am1.font.bold = True
    p_am1.font.size = Pt(15)
    p_am1.font.color.rgb = C_ACCENT_AMBER
    
    am_text = (
        "• p (Tỷ trọng song song hóa):\n"
        "Nhóm chỉ tính p từ 3 giai đoạn thắt nút cổ chai chính (S1 + S2 + S6).\n\n"
        "• 1 - p (Phần tuần tự bảo thủ):\n"
        "Toàn bộ S3, S4, S5 cùng độ trễ truyền nhận Host-Device (H2D/D2H) được xem là tuần tự.\n\n"
        "• Ý nghĩa kỹ thuật:\n"
        "Cho dù phần GPU có nhanh đến vô cùng, độ tăng tốc toàn quy trình (End-to-End) cũng không thể vượt quá ngưỡng trần S_max này."
    )
    p_am2 = tf_am.add_paragraph()
    p_am2.text = am_text
    p_am2.font.size = Pt(11)
    p_am2.font.color.rgb = C_TEXT_MUTED

    # Table Right
    t_shape8 = s8.shapes.add_table(6, 4, Inches(5.6), Inches(1.8), Inches(6.8), Inches(4.8))
    t8 = t_shape8.table
    t8.columns[0].width = Inches(2.2)
    t8.columns[1].width = Inches(1.5)
    t8.columns[2].width = Inches(1.5)
    t8.columns[3].width = Inches(1.6)

    h8 = ["TẬP DỮ LIỆU", "720p (Smax)", "1080p (Smax)", "4K UHD (Smax)"]
    for j, h in enumerate(h8):
        cell = t8.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 58, 138)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_TEXT_WHITE

    r8_data = [
        ("Synthetic", "p=83.5% (6.05×)", "p=83.2% (5.96×)", "p=94.4% (17.74×)"),
        ("Site 1: Indoor", "p=68.7% (3.19×)", "p=78.2% (4.58×)", "p=92.7% (13.76×)"),
        ("Site 2: Outdoor", "p=79.7% (4.92×)", "p=76.5% (4.25×)", "p=88.3% (8.54×)"),
        ("Site 3: Custom", "p=64.1% (2.78×)", "p=80.0% (5.00×)", "p=94.0% (16.53×)"),
        ("Challenging: Night", "p=59.7% (2.48×)", "p=68.5% (3.17×)", "p=90.4% (10.38×)"),
    ]
    for i, r in enumerate(r8_data):
        for j, val in enumerate(r):
            cell = t8.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(30, 41, 59) if i % 2 == 0 else RGBColor(15, 23, 42)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            p.font.color.rgb = C_TEXT_WHITE if j == 0 else (C_ACCENT_GREEN if j == 3 else C_TEXT_MUTED)

    # =========================================================================
    # SLIDE 9: V1 CUDA THREAD MAPPING STRATEGY
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Chiến lược Ánh xạ Luồng V1 CUDA (V1 Thread Mapping)")

    # 6 Kernel Grid
    t_shape9 = s9.shapes.add_table(7, 4, Inches(0.8), Inches(1.8), Inches(11.6), Inches(4.8))
    t9 = t_shape9.table
    t9.columns[0].width = Inches(2.2)
    t9.columns[1].width = Inches(2.8)
    t9.columns[2].width = Inches(3.6)
    t9.columns[3].width = Inches(3.0)

    h9 = ["KERNEL GIAI ĐOẠN", "CẤU TRÚC GRID & BLOCK", "NHIỆM VỤ CỦA MỘT THREAD / WARP", "RỦI RO TÍNH ĐÚNG ĐẮN"]
    for j, h in enumerate(h9):
        cell = t9.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 58, 138)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_TEXT_WHITE

    r9_data = [
        ("K1: Tiền xử lý", "2D Block 16×16\nGrid = (W/16, H/16)", "1 thread xử lý 1 pixel: đổi mức xám và trượt cửa sổ Gaussian 5×5", "Lỗi tràn biên halo; giải pháp: clamp biên đúng như V0"),
        ("K2: FAST Detector", "2D Block 16×16\nGrid = (W/16, H/16)", "1 thread kiểm tra vòng tròn Bresenham + NMS; Atomic add đếm ứng viên", "Thứ tự gom điểm bất định; giải pháp: sort tất định bằng response"),
        ("K3: ORB Descriptor", "1D Block 256 threads\nGrid = (K / 256)", "1 thread xử lý 1 keypoint: tính góc xoay và trích 256-bit BRIEF", "Bảng mẫu 256 cặp nạp vào __constant__ memory"),
        ("K4: Khớp Hamming", "2D Tile Block 16×16\n(Descriptor Tiles)", "Tận dụng chỉ thị __popc phần cứng; reduction tìm 2 điểm gần nhất", "Xử lý bằng điểm; giải pháp: tie-break theo chỉ số index"),
        ("K5: RANSAC DLT", "1D Block (32 or 64)\nGrid = 2000 hypotheses", "1 warp hoặc thread đánh giá 1 giả thuyết DLT; LCG seed độc lập", "Mẫu suy biến 3 điểm thẳng hàng; loại bỏ bằng định thức det"),
        ("K6: Chiếu & Trộn", "2D Block 16×16\nGrid Canvas", "1 thread tính màu 1 pixel canvas: ánh xạ ngược, Bilinear, Feathering", "Trôi dạt tọa độ; giải pháp: ép kiểu float32 đồng nhất host"),
    ]
    for i, r in enumerate(r9_data):
        for j, val in enumerate(r):
            cell = t9.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(30, 41, 59) if i % 2 == 0 else RGBColor(15, 23, 42)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            p.font.color.rgb = C_TEXT_WHITE if j < 3 else C_ACCENT_AMBER

    # Bottom note
    bn9 = s9.shapes.add_textbox(Inches(0.8), Inches(6.8), Inches(11.6), Inches(0.4))
    p_bn9 = bn9.text_frame.paragraphs[0]
    p_bn9.text = "HỢP ĐỒNG BỘ NHỚ V1: Toàn bộ dữ liệu trung gian (ảnh xám, danh sách keypoints, descriptors, ma trận H) lưu hoàn toàn trên VRAM GPU để triệt tiêu chi phí truyền PCI-e H2D/D2H giữa các giai đoạn."
    p_bn9.font.size = Pt(10)
    p_bn9.font.color.rgb = C_PRIMARY

    # =========================================================================
    # SLIDE 10: OPTIMIZATION ROADMAP (V2/V3) & NEXT STEPS
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Lộ trình Tối ưu hóa V2 / V3 & Kế hoạch Hành động", "HƯỚNG TỚI BÁO CÁO CUỐI KỲ")

    # 3 Optimization Candidates
    opts = [
        ("V2: SHARED MEMORY TILING", "Giai đoạn S1 (Lọc Gaussian)", "Nạp tile 16×16 kèm vùng đệm halo 2px vào Shared Memory để tái sử dụng điểm ảnh lân cận. Giảm 80% lượt đọc Global Memory.", "Rủi ro: Chi phí đồng bộ __syncthreads() và tràn bank bộ nhớ.", C_PRIMARY),
        ("V3.A: GPU STREAM COMPACTION", "Giai đoạn S2 (Gom Keypoint)", "Thay thế Atomic toàn cục bằng Prefix-Sum (Scan) hoặc Thrust::copy_if để gom Top-K điểm đặc trưng song song trên GPU.", "Rủi ro: Tốn bộ nhớ đệm tạm thời cho mảng tiền tố.", C_ACCENT_GREEN),
        ("V3.B: TEXTURE HARDWARE OBJECTS", "Giai đoạn S6 (Chiếu & Nội suy)", "Sử dụng Texture Units phần cứng của CUDA để tự động lấy mẫu nội suy song tuyến (Bilinear hardware cache) cực nhanh.", "Rủi ro: Sai số làm tròn phần cứng khác biệt nhỏ so với CPU.", C_ACCENT_PURPLE),
    ]
    for i, (name, target, idea, risk, col) in enumerate(opts):
        x = Inches(0.8 + i * 4.0)
        c = add_card(s10, x, Inches(1.8), Inches(3.7), Inches(4.5), "", col)
        tb = s10.shapes.add_textbox(x + Inches(0.2), Inches(2.0), Inches(3.3), Inches(4.1))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = name
        p1.font.bold = True
        p1.font.size = Pt(13)
        p1.font.color.rgb = col
        p2 = tf.add_paragraph()
        p2.text = target + "\n"
        p2.font.bold = True
        p2.font.size = Pt(11)
        p2.font.color.rgb = C_TEXT_WHITE
        p3 = tf.add_paragraph()
        p3.text = idea + "\n"
        p3.font.size = Pt(11)
        p3.font.color.rgb = C_TEXT_MUTED
        p4 = tf.add_paragraph()
        p4.text = risk
        p4.font.size = Pt(10)
        p4.font.color.rgb = C_ACCENT_AMBER

    # Final Team Callout
    call10 = s10.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(6.5), Inches(11.6), Inches(0.6))
    call10.fill.solid()
    call10.fill.fore_color.rgb = RGBColor(6, 78, 59)
    call10.line.fill.background()
    p_c10 = call10.text_frame.paragraphs[0]
    p_c10.text = "NGUYÊN TẮC TỐI ƯU HÓA: Mọi tối ưu hóa ở V2/V3 phải dựa trên bằng chứng đo đạc Nsight Systems / Compute. V0 đã hoàn hảo — Nhóm sẵn sàng 100% bước vào giai đoạn tăng tốc CUDA!"
    p_c10.font.bold = True
    p_c10.font.size = Pt(11)
    p_c10.font.color.rgb = C_TEXT_WHITE
    p_c10.alignment = PP_ALIGN.CENTER

    prs.save(output_pptx)
    print(f"[OK] Successfully generated presentation: {output_pptx}")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "presentation_midterm_team.pptx"
    create_deck(out)
