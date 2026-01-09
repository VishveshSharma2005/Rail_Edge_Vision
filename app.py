import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import altair as alt
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="RailVision EdgeAI Command Center",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Dark Mode" Professional Look
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DUMMY AI MODELS (Replace with your Real Pipelines)
# ==========================================
def run_ai_pipeline(frame, enable_enhancement):
    """
    Simulates your full pipeline:
    Blur Check -> Deblur -> Low Light -> OCR -> Damage
    """
    # Simulate processing delay
    time.sleep(0.03) 
    
    # 1. Simulate Enhancement (Just visual trick for demo if no model loaded)
    processed_frame = frame.copy()
    if enable_enhancement:
        # Fake "Enhancement": Increase contrast/brightness to look 'cleaned'
        processed_frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
        # Add a green border to show "AI Active"
        cv2.rectangle(processed_frame, (0,0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 10)
    
    # 2. Simulate OCR & Damage Data
    # In real code, return actual model outputs here
    mock_data = {
        "wagon_id": f"WR-{np.random.randint(10000, 99999)}",
        "blur_score": np.random.uniform(0.1, 0.9),
        "defects": np.random.choice(["None", "Crack", "Rust"], p=[0.8, 0.1, 0.1]),
        "confidence": np.random.uniform(85, 99)
    }
    
    return processed_frame, mock_data

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/high-speed-train.png", width=80)
    st.title("RailVision Config")
    
    st.markdown("---")
    st.subheader("📡 Input Source")
    input_source = st.radio("Select Feed:", ("Upload Video", "RTSP Stream (Simulated)"))
    
    st.markdown("---")
    st.subheader("🧠 AI Modules")
    enable_deblur = st.checkbox("Motion Deblurring (GAN)", value=True)
    enable_lowlight = st.checkbox("Night Vision (Zero-DCE)", value=True)
    enable_ocr = st.checkbox("Wagon OCR", value=True)
    
    st.markdown("---")
    st.info("System Status: **ONLINE**\n\nDevice: **NVIDIA Jetson AGX**\n\nTemp: **42°C**")

# ==========================================
# 4. MAIN DASHBOARD UI
# ==========================================
st.title("🚄 RailVision: Intelligent Wagon Inspection")
st.markdown("Real-time edge analytics for high-speed freight monitoring.")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["🔴 Live Inspector", "📊 Analytics Hub", "⚙️ System Health"])

# --- TAB 1: LIVE INSPECTOR ---
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Camera Feed (Side View)")
        video_placeholder = st.empty()
        
        # Comparison Toggle
        view_mode = st.radio("View Mode:", ["Split Screen (Raw vs Enhanced)", "Final Output Only"], horizontal=True)

    with col2:
        st.subheader("Real-Time Telemetry")
        # Placeholders for live metrics
        ocr_card = st.empty()
        blur_metric = st.empty()
        defect_alert = st.empty()

    # --- Video Loop Logic ---
    start_btn = st.button("▶️ Start Inspection")
    
    if start_btn:
        # Use a demo video or webcam (0)
        cap = cv2.VideoCapture("demo_video.mp4" if input_source == "Upload Video" else 0) 
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("End of video stream.")
                break
                
            # Resize for dashboard performance
            frame = cv2.resize(frame, (640, 360))
            
            # Run AI
            enhanced_frame, data = run_ai_pipeline(frame, enable_enhancement=(enable_deblur or enable_lowlight))
            
            # --- UPDATE UI COMPONENTS ---
            
            # 1. Video Player
            if view_mode == "Split Screen (Raw vs Enhanced)":
                # Stack images horizontally
                combined = np.hstack((frame, enhanced_frame))
                video_placeholder.image(combined, channels="BGR", caption="Left: RAW Input | Right: RailVision Enhanced", use_column_width=True)
            else:
                video_placeholder.image(enhanced_frame, channels="BGR", caption="Final AI Output", use_column_width=True)
            
            # 2. OCR Card (Styled)
            ocr_card.markdown(f"""
            <div class="metric-card">
                <h3>🆔 Wagon ID: <span style="color:#4caf50">{data['wagon_id']}</span></h3>
                <p>Confidence: {data['confidence']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 3. Blur Metric (Gauge)
            blur_val = data['blur_score']
            blur_color = "red" if blur_val > 0.6 else "green"
            blur_metric.markdown(f"""
            **Motion Blur Level:**
            <div style="width:100%; background-color:#ddd; border-radius:5px;">
                <div style="width:{blur_val*100}%; background-color:{blur_color}; height:10px; border-radius:5px;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # 4. Defect Alert
            if data['defects'] != "None":
                defect_alert.error(f"⚠️ DEFECT DETECTED: {data['defects']}")
            else:
                defect_alert.success("✅ Wagon Status: CLEAR")

            # 5. Log Data to Session State (for Analytics Tab)
            if "log_data" not in st.session_state:
                st.session_state["log_data"] = []
            
            st.session_state["log_data"].append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Wagon ID": data['wagon_id'],
                "Defect": data['defects'],
                "Blur": data['blur_score']
            })

# --- TAB 2: ANALYTICS HUB ---
with tab2:
    st.subheader("Post-Operation Analytics")
    
    if "log_data" in st.session_state and len(st.session_state["log_data"]) > 0:
        df = pd.DataFrame(st.session_state["log_data"])
        
        # 1. Summary Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Wagons Scanned", len(df))
        m2.metric("Defects Found", len(df[df["Defect"] != "None"]))
        m3.metric("Avg Blur Score", f"{df['Blur'].mean():.2f}")
        
        # 2. Charts
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Defect Distribution")
            chart_data = df["Defect"].value_counts().reset_index()
            chart_data.columns = ["Type", "Count"]
            c = alt.Chart(chart_data).mark_bar().encode(
                x='Type', y='Count', color='Type'
            )
            st.altair_chart(c, use_container_width=True)
            
        with c2:
            st.markdown("### Blur Levels Over Time")
            st.line_chart(df["Blur"])
            
        # 3. Detailed Data Table
        st.dataframe(df)
        
        # Export Button
        st.download_button("📥 Download Inspection Report", df.to_csv(), "railvision_report.csv")
    else:
        st.info("Start the inspection in 'Live Inspector' to generate data.")

# --- TAB 3: SYSTEM HEALTH ---
with tab3:
    st.subheader("Edge Device Telemetry (Jetson AGX)")
    colA, colB = st.columns(2)
    
    with colA:
        st.progress(72, text="GPU Usage (CUDA Core Load)")
        st.progress(45, text="RAM Usage (14GB / 32GB)")
    
    with colB:
        st.metric("Inference Latency", "12ms", "-2ms")
        st.metric("FPS", "42", "+4")
    
    st.code("""
    # Model Loading Status
    [OK] DeblurGAN-v2 (FP16 optimized)
    [OK] Zero-DCE (TensorRT engine)
    [OK] PaddleOCR (v3.0)
    [OK] YOLOv8 (Small)
    """, language="bash")
