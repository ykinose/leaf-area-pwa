import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from pillow_heif import register_heif_opener
import io
import base64

# HEIC対応
register_heif_opener()

st.set_page_config(page_title="Leaf Area Analyzer", layout="centered")

st.title("🍃 葉の面積解析 (安定稼働版)")

def get_b64_image(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

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
    
    # 表示サイズ計算
    display_w = 700
    display_h = int(raw_img.height * (display_w / raw_img.width))
    input_image = raw_img.resize((display_w, display_h), Image.LANCZOS)
    
    # 本物の画像をBase64化（CSS用）
    bg_url = get_b64_image(input_image)
    
    # 【解決の鍵】1ピクセルだけの透明なダミー画像を作成（ライブラリのバグ回避用）
    dummy_img = Image.new("RGBA", (display_w, display_h), (0, 0, 0, 0))

    st.info("スポンジの4隅をタップしてください。")

    # CSSで本物の画像を背後に配置
    st.markdown(
        f"""
        <style>
        .stCanvasContainer {{
            background-image: url("{bg_url}");
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # background_imageに「透明な画像」を渡すことで、ライブラリを正常動作させる
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        background_image=dummy_img, 
        drawing_mode="point",
        width=display_w,
        height=display_h,
        point_display_radius=10,
        update_freq=50,
        key="canvas_dummy_fix",
    )

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
            mask = cv2.inRange(hsv, np.array([25, 35, 35]), np.array([95, 255, 255]))
            area = np.sum(mask > 0) * px_per_cm2
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(trimmed, caption="補正後")
            with col2:
                st.image(mask, caption="解析マスク")
            
            st.success(f"推定面積: {area:.4f} cm²")
            
            csv = f"filename,area_cm2\n{img_file.name},{area}\n"
            st.download_button("CSV保存", csv, f"LA_{img_file.name}.csv", "text/csv")
