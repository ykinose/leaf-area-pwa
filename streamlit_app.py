import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Leaf Area Analyzer", layout="centered")

st.title("🍃 葉の面積解析 (Mobile PWA)")

# 設定（スポンジサイズ）
with st.sidebar:
    st.header("設定")
    sw = st.number_input("スポンジ幅 (cm)", value=4.0)
    sh = st.number_input("スポンジ高さ (cm)", value=2.6)
    # 400x260pxの枠に対する1pxあたりの面積
    px_per_cm2 = (sw * sh) / (400 * 260)

# 画像入力
img_file = st.file_uploader("画像を選択", type=['jpg', 'jpeg', 'png'])
camera_file = st.camera_input("カメラで撮影")
target = img_file or camera_file

if target:
    # 画像の読み込みを安定させる処理
    input_image = Image.open(target).convert("RGB")
    
    st.subheader("1. スポンジの4隅をタップ")
    st.info("左上→右上→右下→左下の順で4点をタップしてください。")
    
    # キャンバスの表示（エラー回避のため、画像が存在する場合のみ実行）
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        background_image=input_image,
        drawing_mode="point",
        point_display_radius=10,
        update_freq=10,
        display_toolbar=True,
        key="canvas",
    )

    # 4点以上タップされたら解析開始
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects")
        if objects and len(objects) >= 4:
            # 座標抽出
            pts = []
            for obj in objects[:4]:
                pts.append([obj["left"], obj["top"]])
            pts = np.float32(pts)
            
            # 画像をnumpy配列に変換
            img_array = np.array(input_image)
            
            # 歪み補正 (Perspective Transform)
            dst_pts = np.float32([[0, 0], [400, 0], [400, 260], [0, 260]])
            matrix = cv2.getPerspectiveTransform(pts, dst_pts)
            trimmed = cv2.warpPerspective(img_array, matrix, (400, 260))
            
            st.subheader("2. 解析結果")
            
            # 葉の抽出ロジック (HSV色空間)
            hsv = cv2.cvtColor(trimmed, cv2.COLOR_RGB2HSV)
            lower_green = np.array([30, 40, 40])
            upper_green = np.array([90, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # 面積計算
            pixels = np.sum(mask > 0)
            area = pixels * px_per_cm2
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(trimmed, caption="補正後のスポンジ範囲")
            with col2:
                st.image(mask, caption="解析マスク（白が葉）")
                
            st.success(f"推定面積: {area:.4f} cm²")
            
            # CSVデータの準備
            csv_data = f"filename,area_cm2\n{target.name},{area}\n"
            st.download_button(
                label="結果をCSVで保存",
                data=csv_data,
                file_name=f"LA_{target.name}.csv",
                mime="text/csv"
            )
