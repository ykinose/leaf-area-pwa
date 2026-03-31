import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from streamlit_image_coordinates import streamlit_image_coordinates

# HEIC対応
register_heif_opener()

st.set_page_config(page_title="Leaf Area Analyzer", layout="centered")

st.title("🍃 葉の面積解析 (ポチポチ座標取得版)")

# 座標を一時的に保存するための「記憶領域（セッション）」を準備
if "points" not in st.session_state:
    st.session_state.points = []

# 設定
with st.sidebar:
    st.header("設定")
    sw = st.number_input("スポンジ幅 (cm)", value=4.0)
    sh = st.number_input("スポンジ高さ (cm)", value=2.6)
    px_per_cm2 = (sw * sh) / (400 * 260)
    if st.button("座標をリセット"):
        st.session_state.points = []
        st.rerun()

img_file = st.file_uploader("画像を選択 (JPG, PNG, HEIC)", type=['jpg', 'jpeg', 'png', 'heic'])

if img_file:
    # 画像読み込み
    raw_img = Image.open(img_file).convert("RGB")
    
    st.subheader("1. スポンジの4隅を順にタップ")
    st.write(f"現在の選択点数: **{len(st.session_state.points)} / 4**")
    st.info("左上 → 右上 → 右下 → 左下 の順でタップしてください。")

    # 画像を表示し、クリックされた座標を受け取る
    # ※ width=700で表示を固定し、スマホでも操作しやすくします
    value = streamlit_image_coordinates(raw_img, width=700, key="coords")

    # クリックされたら座標をリストに追加（4点まで）
    if value is not None:
        point = (value["x"], value["y"])
        if point not in st.session_state.points and len(st.session_state.points) < 4:
            st.session_state.points.append(point)
            st.rerun()

    # 4点揃ったら解析ボタンを表示
    if len(st.session_state.points) == 4:
        if st.button("✨ 面積を計算する"):
            # 座標の準備
            pts = np.float32(st.session_state.points)
            img_array = np.array(raw_img)
            
            # 元画像とプレビューの比率を計算して座標を補正
            ratio = raw_img.width / 700
            pts_corrected = pts * ratio
            
            # 歪み補正 (400x260px)
            dst_pts = np.float32([[0, 0], [400, 0], [400, 260], [0, 260]])
            matrix = cv2.getPerspectiveTransform(pts_corrected, dst_pts)
            trimmed = cv2.warpPerspective(img_array, matrix, (400, 260))
            
            st.divider()
            st.subheader("2. 解析結果")
            
            # 葉の抽出 (HSV)
            hsv = cv2.cvtColor(trimmed, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, np.array([25, 35, 30]), np.array([95, 255, 255]))
            area = np.sum(mask > 0) * px_per_cm2
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(trimmed, caption="補正後")
            with col2:
                st.image(mask, caption="解析マスク")
            
            st.success(f"推定面積: {area:.4f} cm²")
            
            csv = f"filename,area_cm2\n{img_file.name},{area}\n"
            st.download_button("CSV保存", csv, f"LA_{img_file.name}.csv", "text/csv")
