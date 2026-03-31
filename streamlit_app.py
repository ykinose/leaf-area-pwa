import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pillow_heif import register_heif_opener
from streamlit_image_coordinates import streamlit_image_coordinates

register_heif_opener()

st.set_page_config(page_title="Leaf Area GrabCut", layout="centered")

# スマホ操作安定化CSS
st.markdown(
    """
    <style>
    .stImageCoordinates, img { touch-action: manipulation !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🍃 葉の面積解析 (GrabCut版)")

# セッション状態の管理
if "sponge_pts" not in st.session_state:
    st.session_state.sponge_pts = []
if "rect_pts" not in st.session_state:
    st.session_state.rect_pts = [] # 矩形選択用の2点
if "phase" not in st.session_state:
    st.session_state.phase = "sponge"

with st.sidebar:
    st.header("設定")
    sw = st.number_input("スポンジ幅 (cm)", value=4.0)
    sh = st.number_input("スポンジ高さ (cm)", value=2.6)
    if st.button("リセット"):
        for key in ["sponge_pts", "rect_pts", "phase"]:
            st.session_state[key] = [] if key != "phase" else "sponge"
        st.rerun()

img_file = st.file_uploader("画像を選択", type=['jpg', 'jpeg', 'png', 'heic'])

if img_file:
    @st.cache_data
    def get_raw_img(file):
        img = Image.open(file).convert("RGB")
        if img.width > 1200:
            img = img.resize((1200, int(img.height*(1200/img.width))), Image.LANCZOS)
        return img

    raw_img = get_raw_img(img_file)
    disp_w = 600
    ratio = raw_img.width / disp_w
    display_img = raw_img.copy()
    display_img.thumbnail((disp_w, disp_w))
    draw = ImageDraw.Draw(display_img)

    # --- PHASE 1: スポンジ選択 ---
    if st.session_state.phase == "sponge":
        st.subheader("1. スポンジの4隅を選択")
        for p in st.session_state.sponge_pts:
            draw.ellipse((p[0]-6, p[1]-6, p[0]+6, p[1]+6), fill="red")
        
        val = streamlit_image_coordinates(display_img, width=disp_w, key="s_click")
        if val:
            pt = (val["x"], val["y"])
            if not st.session_state.sponge_pts or pt != st.session_state.sponge_pts[-1]:
                st.session_state.sponge_pts.append(pt)
                if len(st.session_state.sponge_pts) == 4:
                    st.session_state.phase = "rect"
                st.rerun()

    # --- PHASE 2: 抽出範囲選択 (GrabCut) ---
    elif st.session_state.phase == "rect":
        st.subheader("2. 葉を囲むように対角の2点をタップ")
        st.info("左上と右下など、葉が収まる四角形を指定してください。")
        
        # スポンジ枠表示
        draw.polygon(st.session_state.sponge_pts, outline="red", width=3)
        
        # 矩形描画
        for p in st.session_state.rect_pts:
            draw.rectangle((p[0]-4, p[1]-4, p[0]+4, p[1]+4), fill="cyan")
        if len(st.session_state.rect_pts) == 2:
            draw.rectangle([st.session_state.rect_pts[0], st.session_state.rect_pts[1]], outline="cyan", width=3)

        val = streamlit_image_coordinates(display_img, width=disp_w, key="r_click")
        if val:
            pt = (val["x"], val["y"])
            if not st.session_state.rect_pts or pt != st.session_state.rect_pts[-1]:
                st.session_state.rect_pts.append(pt)
                if len(st.session_state.rect_pts) > 2: # 3点目はリセットして入れ替え
                    st.session_state.rect_pts = [pt]
                st.rerun()

        if len(st.session_state.rect_pts) == 2:
            if st.button("✨ GrabCutで抽出実行"):
                # --- 元コードのGrabCutロジックを再現 ---
                s_pts_real = np.float32(st.session_state.sponge_pts) * ratio
                w_px = (np.linalg.norm(s_pts_real[0]-s_pts_real[1]) + np.linalg.norm(s_pts_real[3]-s_pts_real[2])) / 2
                h_px = (np.linalg.norm(s_pts_real[0]-s_pts_real[3]) + np.linalg.norm(s_pts_real[1]-s_pts_real[2])) / 2
                
                # 歪み補正
                dst_pts = np.float32([[0,0], [w_px,0], [w_px,h_px], [0,h_px]])
                matrix = cv2.getPerspectiveTransform(s_pts_real, dst_pts)
                trimmed = cv2.warpPerspective(np.array(raw_img), matrix, (int(w_px), int(h_px)))
                
                # 矩形座標を補正後画像に変換
                r_pts_real = np.array(st.session_state.rect_pts) * ratio
                r_pts_reshaped = r_pts_real.reshape(-1, 1, 2).astype(np.float32)
                r_pts_trans = cv2.perspectiveTransform(r_pts_reshaped, matrix)
                
                x0, y0 = np.min(r_pts_trans, axis=0)[0]
                x1, y1 = np.max(r_pts_trans, axis=0)[0]
                # GrabCut用の矩形設定 (x, y, w, h)
                rect = (int(max(0, x0)), int(max(0, y0)), int(max(1, x1-x0)), int(max(1, y1-y0)))
                
                # GrabCut実行
                mask = np.zeros(trimmed.shape[:2], np.uint8)
                bgd = np.zeros((1, 65), np.float64)
                fgd = np.zeros((1, 65), np.float64)
                cv2.grabCut(trimmed, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
                
                # 前景(確定+見込み)を抽出
                bin_mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
                
                area = np.sum(bin_mask) * ((sw*sh)/(w_px*h_px))
                
                st.divider()
                st.subheader("3. 解析結果")
                st.success(f"面積: {area:.4f} cm²")
                
                # 結果表示
                res_img = trimmed.copy()
                res_img[bin_mask == 0] = [180, 180, 255] # 背景を薄青に
                st.image([trimmed, res_img], caption=["補正画像", "抽出結果 (青は背景)"], width=300)
                st.download_button("CSV保存", f"file,area\n{img_file.name},{area}", f"LA_{img_file.name}.csv")
