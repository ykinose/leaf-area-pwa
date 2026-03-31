import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from pillow_heif import register_heif_opener

# HEIC対応
register_heif_opener()

st.set_page_config(page_title="Leaf Area Analyzer", layout="centered")

st.title("🍃 葉の面積解析 (安定版)")

# 設定
with st.sidebar:
    st.header("設定")
    sw = st.number_input("スポンジ幅 (cm)", value=4.0)
    sh = st.number_input("スポンジ高さ (cm)", value=2.6)
    px_per_cm2 = (sw * sh) / (400 * 260)

img_file = st.file_uploader("画像を選択 (JPG, PNG, HEIC)", type=['jpg', 'jpeg', 'png', 'heic'])

if img_file:
    # 画像読み込みとリサイズ
    raw_img = Image.open(img_file).convert("RGB")
    w, h = raw_img.size
    new_w = 700  # 表示サイズを少し小さくして安定させます
    new_h = int(h * (new_w / w))
    input_image = raw_img.resize((new_w, new_h), Image.LANCZOS)
    
    st.info("スポンジの4隅をタップしてください。")

    # 【重要】エラーの原因である background_image をあえて空(None)にします。
    # その代わり、CSSで背後に画像を配置します。
    st.markdown(
        f"""
        <style>
        .stCanvasContainer {{
            background-image: url("data:image/png;base64,{st.image_to_url(input_image, input_image.width, "RGB", "PNG", "canvas_bg")}");
            background-size: contain;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # background_image=None にすることでライブラリ内のバグを回避
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        background_image=None, 
        update_freq=50,
        drawing_mode="point",
        width=new_w,
        height=new_h,
        point_display_radius=10,
        key="canvas_bypass",
    )

    # 以降の解析処理は同じ
    if canvas_result and canvas_result.json_data:
        objs = canvas_result.json_data.get("objects")
        if objs and len(objs) >= 4:
            pts = np.float32([[obj["left"], obj["top"]] for obj in objs[:4]])
            img_array = np.array(input_image)
            
            # 歪み補正
            dst_pts = np.float32([[0, 0], [400, 0], [400, 260], [0, 260]])
            matrix = cv2.getPerspectiveTransform(pts, dst_pts)
            trimmed = cv2.warpPerspective(img_array, matrix, (400, 260))
            
            st.subheader("解析結果")
            hsv = cv2.cvtColor(trimmed, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([95, 255, 255]))
            area = np.sum(mask > 0) * px_per_cm2
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(trimmed, caption="補正後")
            with col2:
                st.image(mask, caption="解析マスク")
            
            st.success(f"推定面積: {area:.4f} cm²")
            
            csv = f"filename,area_cm2\n{img_file.name},{area}\n"
            st.download_button("CSV保存", csv, f"LA_{img_file.name}.csv", "text/csv")
