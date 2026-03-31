import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
import io

# HEIC対応
register_heif_opener()

st.set_page_config(page_title="Leaf Area Analyzer", layout="centered")

st.title("🍃 葉の面積解析 (超安定版)")

# 設定
with st.sidebar:
    st.header("1. スポンジサイズ設定")
    sw = st.number_input("スポンジ幅 (cm)", value=4.0)
    sh = st.number_input("スポンジ高さ (cm)", value=2.6)
    px_per_cm2 = (sw * sh) / (400 * 260)

img_file = st.file_uploader("画像を選択 (JPG, PNG, HEIC)", type=['jpg', 'jpeg', 'png', 'heic'])

if img_file:
    # 画像読み込み
    raw_img = Image.open(img_file).convert("RGB")
    w, h = raw_img.size
    
    # プレビュー表示
    st.subheader("2. 画像プレビューと座標指定")
    st.image(raw_img, use_container_width=True, caption=f"元のサイズ: {w} x {h}")
    
    st.info(f"画像の4隅の座標（ピクセル）を入力してください。\n(0,0)は左上です。")
    
    # 座標入力 (数値入力ならPython 3.14でも絶対に落ちません)
    col1, col2 = st.columns(2)
    with col1:
        x1 = st.number_input("左上 X", 0, w, int(w*0.2))
        y1 = st.number_input("左上 Y", 0, h, int(h*0.2))
        x4 = st.number_input("左下 X", 0, w, int(w*0.2))
        y4 = st.number_input("左下 Y", 0, h, int(h*0.8))
    with col2:
        x2 = st.number_input("右上 X", 0, w, int(w*0.8))
        y2 = st.number_input("右上 Y", 0, h, int(h*0.2))
        x3 = st.number_input("右下 X", 0, w, int(w*0.8))
        y3 = st.number_input("右下 Y", 0, h, int(h*0.8))

    if st.button("解析実行"):
        # 座標の準備
        pts = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        img_array = np.array(raw_img)
        
        # 歪み補正 (400x260px)
        dst_pts = np.float32([[0, 0], [400, 0], [400, 260], [0, 260]])
        matrix = cv2.getPerspectiveTransform(pts, dst_pts)
        trimmed = cv2.warpPerspective(img_array, matrix, (400, 260))
        
        st.subheader("3. 解析結果")
        
        # 葉の抽出 (HSV)
        hsv = cv2.cvtColor(trimmed, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([25, 35, 30]), np.array([95, 255, 255]))
        
        # 面積算出
        area = np.sum(mask > 0) * px_per_cm2
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.image(trimmed, caption="補正後のスポンジ")
        with res_col2:
            st.image(mask, caption="解析マスク")
        
        st.success(f"推定面積: {area:.4f} cm²")
        
        csv = f"filename,area_cm2\n{img_file.name},{area}\n"
        st.download_button("CSV保存", csv, f"LA_{img_file.name}.csv", "text/csv")
