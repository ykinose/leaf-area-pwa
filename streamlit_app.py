import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pillow_heif import register_heif_opener
from streamlit_image_coordinates import streamlit_image_coordinates

# HEIC対応
register_heif_opener()

st.set_page_config(page_title="Leaf Area Analyzer", layout="centered")

st.title("🍃 葉の面積解析 (座標可視化版)")

# セッション状態の管理
if "points" not in st.session_state:
    st.session_state.points = []

# 設定
with st.sidebar:
    st.header("設定")
    sw = st.number_input("スポンジ幅 (cm)", value=4.0)
    sh = st.number_input("スポンジ高さ (cm)", value=2.6)
    if st.button("座標をリセット"):
        st.session_state.points = []
        st.rerun()

img_file = st.file_uploader("画像を選択", type=['jpg', 'jpeg', 'png', 'heic'])

if img_file:
    raw_img = Image.open(img_file).convert("RGB")
    
    # 描画用のコピーを作成して、選択済みの点に赤い丸を描く
    display_img = raw_img.copy()
    draw = ImageDraw.Draw(display_img)
    # スマホでも見やすいように、画像サイズに応じた半径で描画
    radius = max(display_img.width // 100, 10)
    
    for p in st.session_state.points:
        # プレビュー上の座標(700px基準)を元画像座標に変換して描画
        ratio = raw_img.width / 700
        real_x, real_y = p[0] * ratio, p[1] * ratio
        draw.ellipse((real_x-radius, real_y-radius, real_x+radius, real_y+radius), fill="red", outline="white")

    st.subheader(f"1. 4隅をタップ ({len(st.session_state.points)} / 4)")
    st.info("左上 → 右上 → 右下 → 左下 の順にタップしてください。")

    # 画像表示と座標取得
    value = streamlit_image_coordinates(display_img, width=700, key="coords")

    if value is not None:
        point = (value["x"], value["y"])
        if point not in st.session_state.points and len(st.session_state.points) < 4:
            st.session_state.points.append(point)
            st.rerun()

    # 4点揃ったら解析
    if len(st.session_state.points) == 4:
        if st.button("✨ 面積を計算する"):
            # 座標補正
            ratio = raw_img.width / 700
            pts = np.float32(st.session_state.points) * ratio
            
            # --- 【追加機能】縦横サイズをタップした座標から自動計算 ---
            # 幅（上辺と下辺の平均）
            width_auto = (np.linalg.norm(pts[0]-pts[1]) + np.linalg.norm(pts[3]-pts[2])) / 2
            # 高さ（左辺と右辺の平均）
            height_auto = (np.linalg.norm(pts[0]-pts[3]) + np.linalg.norm(pts[1]-pts[2])) / 2
            
            # 1pxあたりの面積を再計算（スポンジの実寸 / 補正後の総ピクセル数）
            px_per_cm2 = (sw * sh) / (width_auto * height_auto)
            
            # 歪み補正 (タップした範囲の平均的なサイズに投影)
            dst_pts = np.float32([[0, 0], [width_auto, 0], [width_auto, height_auto], [0, height_auto]])
            matrix = cv2.getPerspectiveTransform(pts, dst_pts)
            trimmed = cv2.warpPerspective(np.array(raw_img), matrix, (int(width_auto), int(height_auto)))
            
            st.divider()
            st.subheader("2. 解析結果")
            
            # 葉の抽出 (HSV)
            hsv = cv2.cvtColor(trimmed, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([95, 255, 255]))
            area = np.sum(mask > 0) * px_per_cm2
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(trimmed, caption="補正後のスポンジ")
            with col2:
                st.image(mask, caption="解析マスク")
            
            st.success(f"推定面積: {area:.4f} cm²")
            st.write(f"（補正サイズ: {width_auto:.1f} x {height_auto:.1f} px）")
            
            csv = f"filename,area_cm2,width_px,height_px\n{img_file.name},{area},{width_auto},{height_auto}\n"
            st.download_button("CSV保存", csv, f"LA_{img_file.name}.csv", "text/csv")
