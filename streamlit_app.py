import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from pillow_heif import register_heif_opener

# HEIC形式をPillowで扱えるように登録
register_heif_opener()

st.set_page_config(page_title="Leaf Area Analyzer", layout="centered")

st.title("🍃 葉の面積解析 (HEIC対応版)")

# 設定（スポンジサイズ）
with st.sidebar:
    st.header("設定")
    sw = st.number_input("スポンジ幅 (cm)", value=4.0)
    sh = st.number_input("スポンジ高さ (cm)", value=2.6)
    px_per_cm2 = (sw * sh) / (400 * 260)

# 画像入力（HEICを追加）
img_file = st.file_uploader("画像を選択 (JPG, PNG, HEIC)", type=['jpg', 'jpeg', 'png', 'heic'])

if img_file:
    # 1. 画像の読み込みとリサイズ
    raw_image = Image.open(img_file).convert("RGB")
    
    # スマホの高解像度対策：幅1200pxにリサイズ
    max_width = 1200
    if raw_image.width > max_width:
        ratio = max_width / raw_image.width
        new_size = (max_width, int(raw_image.height * ratio))
        input_image = raw_image.resize(new_size, Image.LANCZOS)
    else:
        input_image = raw_image
    
    st.subheader("1. スポンジの4隅をタップ")
    st.info("左上→右上→右下→左下の順で4点をタップしてください。")
    
    # 2. キャンバスの表示
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        background_image=input_image,
        drawing_mode="point",
        point_display_radius=12,
        update_freq=10,
        display_toolbar=True,
        width=input_image.width,
        height=input_image.height,
        key="canvas_heic",
    )

    # 3. 4点以上タップされたら解析開始
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects")
        if objects and len(objects) >= 4:
            pts = []
            for obj in objects[:4]:
                pts.append([obj["left"], obj["top"]])
            pts = np.float32(pts)
            
            img_array = np.array(input_image)
            
            # 歪み補正
            dst_pts = np.float32([[0, 0], [400, 0], [400, 260], [0, 260]])
            matrix = cv2.getPerspectiveTransform(pts, dst_pts)
            trimmed = cv2.warpPerspective(img_array, matrix, (400, 260))
            
            st.subheader("2. 解析結果")
            
            # 葉の抽出 (HSV)
            hsv = cv2.cvtColor(trimmed, cv2.COLOR_RGB2HSV)
            lower_green = np.array([25, 30, 30])
            upper_green = np.array([95, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            pixels = np.sum(mask > 0)
            area = pixels * px_per_cm2
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(trimmed, caption="補正後のスポンジ範囲")
            with col2:
                st.image(mask, caption="解析マスク")
                
            st.success(f"推定面積: {area:.4f} cm²")
            
            csv_data = f"filename,area_cm2\n{img_file.name},{area}\n"
            st.download_button(
                label="結果をCSVで保存",
                data=csv_data,
                file_name=f"LA_{img_file.name}.csv",
                mime="text/csv"
            )
