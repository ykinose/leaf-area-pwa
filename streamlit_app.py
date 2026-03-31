import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pillow_heif import register_heif_opener
from streamlit_image_coordinates import streamlit_image_coordinates

# HEIC対応
register_heif_opener()

st.set_page_config(page_title="Leaf Area Fast", layout="centered")

# --- レスポンス向上のための強力なCSS ---
st.markdown(
    """
    <style>
    /* タップ時の遅延を防止し、ズームなどの余計な挙動を完全にカット */
    .stImageCoordinates, img {
        touch-action: manipulation !important;
        -webkit-tap-highlight-color: transparent !important;
        cursor: crosshair;
    }
    /* ボタンの反応を良くする */
    button {
        min-height: 44px;
        min-width: 44px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🍃 葉の面積解析 (高速反応版)")

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
    # 読み込み時にキャッシュを利用するか、リサイズしてメモリ負荷を下げる
    @st.cache_data
    def load_and_resize(file):
        img = Image.open(file).convert("RGB")
        # 処理速度向上のため、解析に支障ない範囲(1200px)まで落とす
        if img.width > 1200:
            ratio = 1200 / img.width
            img = img.resize((1200, int(img.height * ratio)), Image.LANCZOS)
        return img

    raw_img = load_and_resize(img_file)
    
    # 描画用。さらに小さく(600px)してスマホの描画負荷を軽減
    disp_width = 600 
    draw_ratio = raw_img.width / disp_width
    
    display_img = raw_img.copy()
    display_img.thumbnail((disp_width, disp_width)) # プレビュー用を軽量化
    draw = ImageDraw.Draw(display_img)
    radius = 6

    is_sponge_done = len(st.session_state.sponge_pts) >= 4

    # UI表示
    if not is_sponge_done:
        st.subheader(f"1. スポンジの4隅 ({len(st.session_state.sponge_pts)}/4)")
        for p in st.session_state.sponge_pts:
            draw.ellipse((p[0]-radius, p[1]-radius, p[0]+radius, p[1]+radius), fill="red")
    else:
        st.subheader(f"2. 葉の輪郭 ({len(st.session_state.leaf_pts)}点)")
        # スポンジ枠
        s_pts = st.session_state.sponge_pts
        if len(s_pts) == 4:
            draw.polygon(s_pts, outline="red", width=3)
        # 葉の点
        for p in st.session_state.leaf_pts:
            draw.ellipse((p[0]-radius//2, p[1]-radius//2, p[0]+radius//2, p[1]+radius//2), fill="yellow")
        if len(st.session_state.leaf_pts) >= 2:
            draw.line(st.session_state.leaf_pts + [st.session_state.leaf_pts[0]], fill="yellow", width=2)

    # 座標取得（軽量化したdisplay_imgを渡す）
    val = streamlit_image_coordinates(display_img, width=disp_width, key="fast_coords")

    if val:
        new_pt = (val["x"], val["y"])
        # セッションへの保存処理
        if not is_sponge_done:
            if not st.session_state.sponge_pts or new_pt != st.session_state.sponge_pts[-1]:
                st.session_state.sponge_pts.append(new_pt)
                st.rerun()
        else:
            if not st.session_state.leaf_pts or new_pt != st.session_state.leaf_pts[-1]:
                st.session_state.leaf_pts.append(new_pt)
                st.rerun()

    if is_sponge_done:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("戻る"):
                if st.session_state.leaf_pts: st.session_state.leaf_pts.pop()
                st.rerun()
        with col2:
            if len(st.session_state.leaf_pts) >= 3 and st.button("計算"):
                # 解析計算 (元の解像度に合わせて座標を復元)
                s_pts_real = np.float32(st.session_state.sponge_pts) * draw_ratio
                l_pts_real = np.array(st.session_state.leaf_pts) * draw_ratio
                
                # ...（以下、計算ロジックは前回と同様）...
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
                st.success(f"面積: {area:.4f} cm²")
                st.image([trimmed, mask], width=300)
