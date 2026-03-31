import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pillow_heif import register_heif_opener
from streamlit_image_coordinates import streamlit_image_coordinates

# HEIC形式（iPhone等）への対応
register_heif_opener()

st.set_page_config(page_title="Leaf Area GrabCut Pro", layout="centered")

# スマホのタップ遅延とズーム動作を防止するCSS
st.markdown(
    """
    <style>
    .stImageCoordinates, img {
        touch-action: manipulation !important;
        -webkit-tap-highlight-color: transparent !important;
    }
    /* ボタンを押しやすく */
    .stButton button {
        width: 100%;
        min-height: 3rem;
    }
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

img_file = st.file_uploader("画像を選択 (JPG, PNG, HEIC)", type=['jpg', 'jpeg', 'png', 'heic'])

if img_file:
    # 画像の読み込みと軽量化（処理速度向上）
    @st.cache_data
    def load_and_preprocess(file):
        img = Image.open(file).convert("RGB")
        if img.width > 1200:
            ratio = 1200 / img.width
            img = img.resize((1200, int(img.height * ratio)), Image.LANCZOS)
        return img

    raw_img = load_and_preprocess(img_file)
    
    # 描画用（スマホの画面幅に合わせたサイズ）
    disp_w = 600
    draw_ratio = raw_img.width / disp_w
    display_img = raw_img.copy()
    display_img.thumbnail((disp_w, disp_w))
    draw = ImageDraw.Draw(display_img)
    radius = 6

    # --- PHASE 1: スポンジの4隅選択 ---
    if st.session_state.phase == "sponge":
        st.subheader(f"Step 1: スポンジの4隅を選択 ({len(st.session_state.sponge_pts)}/4)")
        st.info("左上 → 右上 → 右下 → 左下の順にタップしてください。")
        
        for p in st.session_state.sponge_pts:
            draw.ellipse((p[0]-radius, p[1]-radius, p[0]+radius, p[1]+radius), fill="red", outline="white")
        
        val = streamlit_image_coordinates(display_img, width=disp_w, key="sponge_coords")
        if val:
            pt = (val["x"], val["y"])
            if not st.session_state.sponge_pts or pt != st.session_state.sponge_pts[-1]:
                st.session_state.sponge_pts.append(pt)
                if len(st.session_state.sponge_pts) == 4:
                    st.session_state.phase = "confirm_sponge"
                st.rerun()

    # --- PHASE 2: 誤爆防止の確認 ---
    elif st.session_state.phase == "confirm_sponge":
        st.subheader("スポンジ範囲の確認")
        st.success("4隅を選択しました。この範囲でよろしいですか？")
        draw.polygon(st.session_state.sponge_pts, outline="red", width=3)
        st.image(display_img, width=disp_w)
        
        if st.button("✅ OK（次は葉を囲む）"):
            st.session_state.phase = "rect"
            st.rerun()
        if st.button("↩️ 選び直す"):
            st.session_state.sponge_pts = []
            st.session_state.phase = "sponge"
            st.rerun()

    # --- PHASE 3: 葉の範囲指定 (GrabCut用の矩形) ---
    elif st.session_state.phase == "rect":
        st.subheader(f"Step 2: 葉を四角形で囲む ({len(st.session_state.rect_pts)}/2)")
        st.warning("葉が完全に収まるように、対角の2点（左上と右下など）をタップしてください。")
        
        # 背景にスポンジ枠を表示
        draw.polygon(st.session_state.sponge_pts, outline="red", width=2)
        
        # 選択中の点を表示
        for p in st.session_state.rect_pts:
            draw.ellipse((p[0]-radius, p[1]-radius, p[0]+radius, p[1]+radius), fill="cyan", outline="white")
        
        # 2点あれば矩形を描画（型エラー回避のためフラットなタプルで渡す）
        if len(st.session_state.rect_pts) == 2:
            p1, p2 = st.session_state.rect_pts[0], st.session_state.rect_pts[1]
            draw.rectangle((p1[0], p1[1], p2[0], p2[1]), outline="cyan", width=3)

        val = streamlit_image_coordinates(display_img, width=disp_w, key="rect_coords")
        if val:
            pt = (val["x"], val["y"])
            if not st.session_state.rect_pts or pt != st.session_state.rect_pts[-1]:
                st.session_state.rect_pts.append(pt)
                if len(st.session_state.rect_pts) > 2: # 3点目は1点目として上書き
                    st.session_state.rect_pts = [pt]
                st.rerun()

        if len(st.session_state.rect_pts) == 2:
            if st.button("✨ 解析を実行する"):
                # --- 元コードのGrabCutロジックの実行 ---
                # 1. 座標を実スケールに変換
                s_pts_real = np.float32(st.session_state.sponge_pts) * draw_ratio
                r_pts_real = np.array(st.session_state.rect_pts) * draw_ratio
                
                # 2. スポンジのピクセルサイズを計算
                w_px = (np.linalg.norm(s_pts_real[0]-s_pts_real[1]) + np.linalg.norm(s_pts_real[3]-s_pts_real[2])) / 2
                h_px = (np.linalg.norm(s_pts_real[0]-s_pts_real[3]) + np.linalg.norm(s_pts_real[1]-s_pts_real[2])) / 2
                
                # 3. 歪み補正（スポンジを長方形に展開）
                dst_pts = np.float32([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]])
                matrix = cv2.getPerspectiveTransform(s_pts_real, dst_pts)
                trimmed = cv2.warpPerspective(np.array(raw_img), matrix, (int(w_px), int(h_px)))
                
                # 4. 指定した矩形範囲を補正後の座標系に変換
                r_pts_reshaped = r_pts_real.reshape(-1, 1, 2).astype(np.float32)
                r_pts_trans = cv2.perspectiveTransform(r_pts_reshaped, matrix)
                
                x0, y0 = np.min(r_pts_trans, axis=0)[0]
                x1, y1 = np.max(r_pts_trans, axis=0)[0]
                
                # GrabCut用の矩形設定 (x, y, width, height)
                grab_rect = (
                    int(max(0, x0)), 
                    int(max(0, y0)), 
                    int(max(1, x1 - x0)), 
                    int(max(1, y1 - y0))
                )
                
                # 5. GrabCutアルゴリズムの実行
                mask = np.zeros(trimmed.shape[:2], np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                try:
                    cv2.grabCut(trimmed, mask, grab_rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                    # 前景（確定+見込み）を 1, 背景を 0 にする
                    bin_mask = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype('uint8')
                    
                    # 6. 面積算出
                    px_per_cm2 = (sw * sh) / (w_px * h_px)
                    leaf_pixels = np.sum(bin_mask)
                    area = leaf_pixels * px_per_cm2
                    
                    st.divider()
                    st.subheader("Step 3: 解析結果")
                    st.success(f"推定面積: {area:.4f} cm²")
                    
                    # 結果の可視化（背景を薄青にする）
                    overlay = trimmed.copy()
                    overlay[bin_mask == 0] = [180, 180, 255]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(trimmed, caption="補正後のスポンジ")
                    with col2:
                        st.image(overlay, caption="抽出結果 (青は背景)")
                    
                    # CSV保存
                    csv_data = f"filename,area_cm2,width_px,height_px\n{img_file.name},{area},{w_px},{h_px}"
                    st.download_button(
                        label="📊 結果をCSVで保存",
                        data=csv_data,
                        file_name=f"LA_{img_file.name}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"解析中にエラーが発生しました。範囲を広めに指定してみてください。: {e}")
