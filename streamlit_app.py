import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from pillow_heif import register_heif_opener

# HEIC形式をPillowで扱えるように登録
register_heif_opener()

st.set_page_config(page_title="Leaf Area Analyzer", layout="centered")

st.title("🍃 葉の面積解析")

# 画像処理をキャッシュして安定化させる
@st.cache_data
def process_uploaded_image(file):
    raw_image = Image.open(file).convert("RGB")
    max_width = 1000
    if raw_image.width > max_width:
        ratio = max_width / raw_image.width
        new_w = int(max_width)
        new_h = int(raw_image.height * ratio)
        img = raw_image.resize((new_w, new_h), Image.LANCZOS)
    else:
        img = raw_image
    return img

# 設定（スポンジサイズ）
with st.sidebar:
    st.header("設定")
    sw = st.number_input("スポンジ幅 (cm)", value=4.0)
    sh = st.number_input("スポンジ高さ (cm)", value=2.6)
    px_per_cm2 = (sw * sh) / (400 * 260)

# 画像入力
img_file = st.file_uploader("画像を選択 (JPG, PNG, HEIC)", type=['jpg', 'jpeg', 'png', 'heic'])

if img_file:
    # キャッシュされた画像を取得
    input_image = process_uploaded_image(img_file)
    
    st.subheader("1. スポンジの4隅をタップ")
    st.info("左上→右上→右下→左下の順で4点をタップしてください。")
    
    # キャンバスの表示（引数を最小限に）
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        background_image=input_image,
        drawing_mode="point",
        point_display_radius=10,
        update_freq=50,
        display_toolbar=True,
        width=int(input_image.width),
        height=int(input_image.height),
        key="leaf_canvas_v4", # キーを刷新
    )

    # 3. 解析開始
    if canvas_result and canvas_result.json_data:
        objs = canvas_result.json_data.get("objects")
        if objs and len(objs) >= 4:
            pts = []
            for obj in objs[:4]:
                pts.append([obj["left"], obj["top"]])
            pts = np.float32(pts)
            
            img_array = np.array(input_image)
            
            # 歪み補正 (Perspective Transform)
            dst_pts = np.float32([[0, 0], [400, 0], [400, 260], [0, 260]])
            matrix = cv2.getPerspectiveTransform(pts, dst_pts)
            trimmed = cv2.warpPerspective(img_array, matrix, (400, 260))
            
            st.subheader("2. 解析結果")
            
            # 葉の抽出 (HSV)
            hsv = cv2.cvtColor(trimmed, cv2.COLOR_RGB2HSV)
            lower_green = np.array([25, 35, 35])
            upper_green = np.array([95, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            pixels = np.sum(mask > 0)
            area = pixels * px_per_cm2
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(trimmed, caption="補正後")
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
