import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pillow_heif import register_heif_opener
from streamlit_image_coordinates import streamlit_image_coordinates

# HEIC対応
register_heif_opener()

st.set_page_config(page_title="Leaf Area Manual Selector", layout="centered")

# --- スマホのタッチ操作を安定させるCSS ---
st.markdown(
    """
    <style>
    /* 画像表示エリアでのスクロールやズームを防止し、タップの反応を速める */
    .stImageCoordinates, img {
        touch-action: none !important;
        -webkit-user-select: none !important;
        -webkit-touch-callout: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🍃 葉の面積解析 (スマホ完全対応版)")

# セッション状態
if "sponge_pts" not in st.session_state:
    st.session_state.sponge_pts = []
if "leaf_pts" not in st.session_state:
    st.session_state.leaf_pts = []

# 設定
with st.sidebar:
    st.header("設定")
    sw = st.number_input("スポンジ幅 (cm)", value=4.0)
    sh = st.number_input("スポンジ高さ (cm)", value=2.6)
    if st.button("全データをリセット"):
        st.session_state.sponge_pts = []
        st.session_state.leaf_pts = []
        st.rerun()

img_file = st.file_uploader("画像を選択", type=['jpg', 'jpeg', 'png', 'heic'])

if img_file:
    raw_img = Image.open(img_file).convert("RGB")
    display_img = raw_img.copy()
    draw = ImageDraw.Draw(display_img)
    
    # 表示サイズを固定（スマホの画面幅に合わせる）
    disp_width = 700 
    ratio = raw_img.width / disp_width
    radius = max(display_img.width // 120, 8)

    # フェーズ判定
    is_sponge_done = len(st.session_state.sponge_pts) >= 4

    if not is_sponge_done:
        st.subheader(f"1. スポンジの4隅を選択 ({len(st.session_state.sponge_pts)} / 4)")
        st.info("左上 → 右上 → 右下 → 左下の順にタップ")
        for p in st.session_state.sponge_pts:
            draw.ellipse((p[0]*ratio-radius, p[1]*ratio-radius, p[0]*ratio+radius, p[1]*ratio+radius), fill="red")
    else:
        st.subheader(f"2. 葉の輪郭をタップして囲む ({len(st.session_state.leaf_pts)} 点)")
        st.warning("葉を1周するようにタップ。最後に計算ボタンを押してください。")
        # スポンジ枠
        s_pts = [(p[0]*ratio, p[1]*ratio) for p in st.session_state.sponge_pts]
        draw.polygon(s_pts, outline="red", width=5)
        # 葉の点
        for p in st.session_state.leaf_pts:
            draw.ellipse((p[0]*ratio-radius//2, p[1]*ratio-radius//2, p[0]*ratio+radius//2, p[1]*ratio+radius//2), fill="yellow")
        if len(st.session_state.leaf_pts) >= 2:
            l_pts = [(p[0]*ratio, p[1]*ratio) for p in st.session_state.leaf_pts]
            draw.line(l_pts + [l_pts[0]], fill="yellow", width=3)

    # 座標取得実行
    val = streamlit_image_coordinates(display_img, width=disp_width, key="touch_coords")

    if val:
        # プレビュー上の(x, y)を取得
        new_pt = (val["x"], val["y"])
        
        # 直前の座標と同じでないかチェック（連打防止）
        if not is_sponge_done:
            if not st.session_state.sponge_pts or new_pt != st.session_state.sponge_pts[-1]:
                st.session_state.sponge_pts.append(new_pt)
                st.rerun()
        else:
            if not st.session_state.leaf_pts or new_pt != st.session_state.leaf_pts[-1]:
                st.session_state.leaf_pts.append(new_pt)
                st.rerun()

    # 操作ボタン
    if is_sponge_done:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("一つ消す"):
                if st.session_state.leaf_pts: st.session_state.leaf_pts.pop()
                st.rerun()
        with col2:
            if len(st.session_state.leaf_pts) >= 3:
                if st.button("✨ 確定して計算"):
                    # 座標変換と解析
                    s_pts_real = np.float32(st.session_state.sponge_pts) * ratio
                    l_pts_real = np.array(st.session_state.leaf_pts) * ratio
                    
                    w_px = (np.linalg.norm(s_pts_real[0]-s_pts_real[1]) + np.linalg.norm(s_pts_real[3]-s_pts_real[2])) / 2
                    h_px = (np.linalg.norm(s_pts_real[0]-s_pts_real[3]) + np.linalg.norm(s_pts_real[1]-s_pts_real[2])) / 2
                    px_per_cm2 = (sw * sh) / (w_px * h_px)
                    
                    dst_pts = np.float32([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]])
                    matrix = cv2.getPerspectiveTransform(s_pts_real, dst_pts)
                    trimmed = cv2.warpPerspective(np.array(raw_img), matrix, (int(w_px), int(h_px)))
                    
                    l_pts_reshaped = l_pts_real.reshape(-1, 1, 2).astype(np.float32)
                    l_pts_transformed = cv2.perspectiveTransform(l_pts_reshaped, matrix)
                    
                    mask = np.zeros(trimmed.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [l_pts_transformed.astype(np.int32)], 255)
                    
                    area = np.sum(mask > 0) * px_per_cm2
                    
                    st.divider()
                    st.image([trimmed, mask], caption=["補正後", "マスク"], width=300)
                    st.success(f"推定面積: {area:.4f} cm²")
                    st.download_button("CSV保存", f"file,area\n{img_file.name},{area}", f"LA_{img_file.name}.csv")
