import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from pillow_heif import register_heif_opener

# HEIC対応
register_heif_opener()

st.set_page_config(page_title="Leaf Area Analyzer", layout="centered")

st.title("🍃 葉の面積解析")

# 設定
with st.sidebar:
    st.header("設定")
    sw = st.number_input("スポンジ幅 (cm)", value=4.0)
    sh = st.number_input("スポンジ高さ (cm)", value=2.6)
    px_per_cm2 = (sw * sh) / (400 * 260)

img_file = st.file_uploader("画像を選択 (JPG, PNG, HEIC)", type=['jpg', 'jpeg', 'png', 'heic'])

if img_file:
    # 画像読み込み
    raw_img = Image.open(img_file).convert("RGB")
    
    # 処理を軽くするためリサイズ（整数型）
    w, h = raw_img.size
    new_w = 800
    new_h = int(h * (new_w / w))
    input_image = raw_img.resize((new_w, new_h), Image.LANCZOS)
    
    st.info("スポンジの4隅をタップしてください。")
    
    # 【最重要】エラーが出る引数 (fill_color等) をすべて削除し、最小構成にします
    canvas_result = st_canvas(
        background_image=input_image,
        drawing_mode="point",
        width=new_w,
        height=new_h,
        point_display_radius=10,
        key="canvas_minimal",
    )

    # 解析処理
    if canvas_result is not None and canvas_result.json_data is not None:
        objs = canvas_result.json_data.get("objects")
        if objs and len(objs) >= 4:
            pts = np.float32([[obj["left"], obj["top"]] for obj in objs[:4]])
            img_array = np.array(input_image)
            
            # 歪み補正
            dst_pts = np.float32([[0, 0], [400, 0], [400, 260], [0, 260]])
            matrix = cv2.getPerspectiveTransform(pts, dst_pts)
            trimmed = cv2.warpPerspective(img_array, matrix, (400, 260))
            
            # 面積計算 (HSV)
            hsv = cv2.cvtColor(trimmed, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([95, 255, 255]))
            area = np.sum(mask > 0) * px_per_cm2
            
            st.image([trimmed, mask], caption=["補正後", "マスク"], width=300)
            st.success(f"推定面積: {area:.4f} cm²")
            
            csv = f"filename,area_cm2\n{img_file.name},{area}\n"
            st.download_button("CSV保存", csv, f"LA_{img_file.name}.csv", "text/csv")
