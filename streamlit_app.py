import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pillow_heif import register_heif_opener
from streamlit_image_coordinates import streamlit_image_coordinates

# HEIC形式対応
register_heif_opener()

st.set_page_config(page_title="Leaf Area GrabCut Pro", layout="centered")

# スマホ操作安定化CSS
st.markdown(
    """
    <style>
    .stImageCoordinates, img {
        touch-action: manipulation !important;
        -webkit-tap-highlight-color: transparent !important;
    }
    .stButton button { width: 100%; min-height: 3rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🍃 葉の面積解析 (GrabCut版)")

# セッション状態の初期化
if "sponge_pts" not in st.session_state:
    st.session_state.sponge_pts = []
if "rect_pts" not in st.session_state:
    st.session_state.rect_pts = []
if "phase" not in st.session_state:
    st.session_state.phase = "sponge"

# サイドバー設定
with st.sidebar:
    st.header("1. スポンジ実寸設定")
    sw = st.number_input("幅 (cm)", value=4.0, step=0.1)
    sh = st.number_input("高さ (cm)", value=2.6, step=0.1)
    
    st.divider()
    if st.button("🔄 最初からリセット"):
        for key in ["sponge_pts", "rect_pts", "phase"]:
            st.session_state[key] = [] if key != "phase" else "sponge"
        st.rerun()

img_file = st.file_uploader("画像を選択", type=['jpg', 'jpeg', 'png', 'heic'])

if img_file:
    @st.cache_data
    def load_preprocess(file):
        img = Image.open(file).convert("RGB")
        if img.width > 1200:
            ratio = 1200 / img.width
            img = img.resize((1200, int(img.height * ratio)), Image.LANCZOS)
        return img

    raw_img = load_preprocess(img_file)
    disp_w = 600
    draw_ratio = raw_img.width / disp_w
    display_img = raw_img.copy()
    display_img.thumbnail((disp_w, disp_w))
    draw = ImageDraw.Draw(display_img)
    radius = 6

    # --- PHASE 1: スポンジ選択 ---
    if st.session_state.phase == "sponge":
        st.subheader(f"Step 1: スポンジの4隅 ({len(st.session_state.sponge_pts)}/4)")
        for p in st.session_state.sponge_pts:
            # 座標をintに変換して描画
            ix, iy = int(p[0]), int(p[1])
            draw.ellipse((ix-radius, iy-radius, ix+radius, iy+radius), fill="red", outline="white")
        
        val = streamlit_image_coordinates(display_img, width=disp_w, key="s_coords")
        if val:
            pt = (val["x"], val["y"])
            if not st.session_state.sponge_pts or pt != st.session_state.sponge_pts[-1]:
                st.session_state.sponge_pts.append(pt)
                if len(st.session_state.sponge_pts) == 4:
                    st.session_state.phase = "confirm_sponge"
                st.rerun()

    # --- PHASE 2: 確認 ---
    elif st.session_state.phase == "confirm_sponge":
        st.subheader("スポンジ範囲の確認")
        # 座標をintのリストに変換して描画
        i_pts = [(int(p[0]), int(p[1])) for p in st.session_state.sponge_pts]
        draw.polygon(i_pts, outline="red", width=3)
        st.image(display_img, width=disp_w)
        
        if st.button("✅ OK（次は葉を囲む）"):
            st.session_state.phase = "rect"
            st.rerun()
        if st.button("↩️ 選び直す"):
            st.session_state.sponge_pts = []
            st.session_state.phase = "sponge"
            st.rerun()

    # --- PHASE 3: 葉の範囲指定 ---
    elif st.session_state.phase == "rect":
        st.subheader(f"Step 2: 葉を四角で囲む ({len(st.session_state.rect_pts)}/2)")
        i_s_pts = [(int(p[0]), int(p[1])) for p in st.session_state.sponge_pts]
        draw.polygon(i_s_pts, outline="red", width=2)
        
        for p in st.session_state.rect_pts:
            ix, iy = int(p[0]), int(p[1])
            draw.ellipse((ix-radius, iy-radius, ix+radius, iy+radius), fill="cyan", outline="white")
        
        if len(st.session_state.rect_pts) == 2:
            p1, p2 = st.session_state.rect_pts[0], st.session_state.rect_pts[1]
            # 【重要】すべての座標をintにキャストして描画
            draw.rectangle((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])), outline="cyan", width=3)

        val = streamlit_image_coordinates(display_img, width=disp_w, key="r_coords")
        if val:
            pt = (val["x"], val["y"])
            if not st.session_state.rect_pts or pt != st.session_state.rect_pts[-1]:
                st.session_state.rect_pts.append(pt)
                if len(st.session_state.rect_pts) > 2:
                    st.session_state.rect_pts = [pt]
                st.rerun()

        if len(st.session_state.rect_pts) == 2:
            if st.button("✨ 解析を実行する"):
                s_pts_real = np.float32(st.session_state.sponge_pts) * draw_ratio
                r_pts_real = np.array(st.session_state.rect_pts) * draw_ratio
                
                w_px = (np.linalg.norm(s_pts_real[0]-s_pts_real[1]) + np.linalg.norm(s_pts_real[3]-s_pts_real[2])) / 2
                h_px = (np.linalg.norm(s_pts_real[0]-s_pts_real[3]) + np.linalg.norm(s_pts_real[1]-s_pts_real[2])) / 2
                
                dst_pts = np.float32([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]])
                matrix = cv2.getPerspectiveTransform(s_pts_real, dst_pts)
                trimmed = cv2.warpPerspective(np.array(raw_img), matrix, (int(w_px), int(h_px)))
                
                r_pts_reshaped = r_pts_real.reshape(-1, 1, 2).astype(np.float32)
                r_pts_trans = cv2.perspectiveTransform(r_pts_reshaped, matrix)
                
                x0, y0 = np.min(r_pts_trans, axis=0)[0]
                x1, y1 = np.max(r_pts_trans, axis=0)[0]
                
                grab_rect = (int(max(0, x0)), int(max(0, y0)), int(max(1, x1-x0)), int(max(1, y1-y0)))
                
                mask = np.zeros(trimmed.shape[:2], np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                cv2.grabCut(trimmed, mask, grab_rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                bin_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                
                area = np.sum(bin_mask) * ((sw * sh) / (w_px * h_px))
                
                st.divider()
                st.success(f"推定面積: {area:.4f} cm²")
                overlay = trimmed.copy()
                overlay[bin_mask == 0] = [180, 180, 255]
                st.image([trimmed, overlay], caption=["補正スポンジ", "解析結果"], width=300)
                st.download_button("📊 保存", f"file,area\n{img_file.name},{area}", f"LA_{img_file.name}.csv")
