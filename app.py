import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import altair as alt

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="RailVision Command Center",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Mode CSS
st.markdown("""
<style>
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #4caf50;
    }
    .stAlert {
        padding: 10px;
        border-radius: 5px;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS (The Fail-Safe Engine)
# ==========================================

def generate_synthetic_frame(text_overlay, noise_level=0):
    """Generates a visual simulation if video upload fails."""
    # Create black background
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    
    # Add noise (simulating low light grain)
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
    
    # Add a moving element (simulating a train passing)
    x_pos = int((time.time() * 300) % 800) - 100
    # Draw Wagon Body
    cv2.rectangle(img, (x_pos, 50), (x_pos+300, 310), (50, 50, 50), -1) 
    # Draw Door
    cv2.rectangle(img, (x_pos+20, 100), (x_pos+280, 260), (30, 30, 30), -1) 
    
    # Add Text
    cv2.putText(img, text_overlay, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return img

def image_enhancement_simulation(frame, cam_type):
    """Simulates the specific model behavior for the demo."""
    height, width = frame.shape[:2]
    
    # 1. Create the "Dirty" Input
    dirty_frame = frame.copy()
    
    if "Low Light" in cam_type:
        # Simulate Darkness
        dirty_frame = (dirty_frame * 0.3).astype(np.uint8)
    elif "Motion Blur" in cam_type:
        # Simulate Blur
        dirty_frame = cv2.GaussianBlur(dirty_frame, (15, 15), 0)
        
    # 2. Create the "Clean" Output
    clean_frame = frame.copy()
    
    # Add Overlay
    cv2.putText(clean_frame, "AI ENHANCED (TensorRT)", (20, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return dirty_frame, clean_frame

# ==========================================
# 3. SIDEBAR CONFIG
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/high-speed-train.png", width=60)
    st.title("RailVision Config")
    
    st.subheader("📷 Camera Feed Selection")
    cam_select = st.selectbox(
        "Active Sensor:", 
        ["Side Camera (Motion Blur)", "Top Camera", "Undercarriage (Low Light)"]
    )
    
    st.divider()
    
    st.subheader("🧠 Model Pipeline")
    st.caption("Active Models (Jetson AGX)")
    st.checkbox("DeblurGAN-v2", value=True, disabled=True)
    st.checkbox("Zero-DCE (Night)", value=True, disabled=True)
    st.checkbox("PaddleOCR", value=True, disabled=True)
    st.checkbox("YOLOv8-Small", value=True, disabled=True)
    
    st.divider()
    st.info(f"**Status:** RUNNING\n\n**FPS:** {np.random.randint(38, 45)}\n\n**Latency:** 18ms")

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
st.title("🚄 RailVision: Edge AI Inspection System")
st.markdown(f"**Current View:** {cam_select} | **Speed:** 72 km/h")

# TABS
tab_live, tab_analytics, tab_arch = st.tabs(["🔴 Live Inspection", "📊 Analytics Hub", "🛠️ Architecture"])

# --- TAB 1: LIVE INSPECTOR ---
with tab_live:
    col_video, col_data = st.columns([0.65, 0.35])
    
    with col_video:
        st.subheader("Real-Time Enhancement")
        image_spot = st.empty()
        
    with col_data:
        st.subheader("Live Telemetry")
        m1, m2 = st.columns(2)
        m1.metric("Wagon ID", "WR-8472")
        m2.metric("Conf.", "98.2%")
        
        st.markdown("---")
        st.markdown("**Defect Status:**")
        defect_spot = st.empty()
        
        st.markdown("---")
        st.markdown("**Enhancement Metrics:**")
        blur_chart_spot = st.empty()

    # ANIMATION LOOP
    start = st.toggle("▶️ Activate Edge AI System", value=False)
    
    if start:
        # Attempt to load video, fail gracefully to synthetic generator
        cap = cv2.VideoCapture("demo_video.mp4")
        
        while True:
            frame_source = "synthetic"
            
            if cap.isOpened():
                ret, raw_frame = cap.read()
                if ret:
                    frame_source = "video"
                    raw_frame = cv2.resize(raw_frame, (640, 360))
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                    continue
            
            if frame_source == "synthetic":
                raw_frame = generate_synthetic_frame("Simulating Input...", noise_level=10)
            
            # Process Frame
            dirty, clean = image_enhancement_simulation(raw_frame, cam_select)
            
            # Display
            combined = np.hstack((dirty, clean))
            image_spot.image(combined, channels="BGR", caption="Input (Degraded) vs. Output (Restored)")
            
            # Updates
            if np.random.rand() > 0.90:
                defect_spot.error("⚠️ CRITICAL: BRAKE BEAM CRACK")
            else:
                defect_spot.success("✅ SYSTEM NORMAL")
                
            blur_val = pd.DataFrame({"Blur": [np.random.uniform(0.1, 0.4)]})
            blur_chart_spot.bar_chart(blur_val, height=100)
            
            time.sleep(0.05)

# --- TAB 2: ANALYTICS ---
with tab_analytics:
    st.subheader("Post-Operation Report")
    data = pd.DataFrame({
        "Wagon Type": ["BOXN", "Bcn", "Tanker", "Flatbed", "BOXN"],
        "Defects": [5, 2, 0, 1, 4],
        "Avg Confidence": [95, 92, 98, 89, 94]
    })
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Defect Counts")
        st.bar_chart(data.set_index("Wagon Type")["Defects"])
    
    with c2:
        st.markdown("### Blur Correction")
        chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["DeblurGAN", "Zero-DCE", "Raw"])
        st.line_chart(chart_data)

# --- TAB 3: ARCHITECTURE ---
with tab_arch:
    st.header("Technical Stack")
    st.info("Deployment: NVIDIA Jetson AGX Orin (TensorRT Optimized)")
    st.code("""
    # Hardware Acceleration Pipeline
    Input (RTSP) -> CUDA Buffer -> TensorRT Engine -> Output
    Throughput: 42 FPS @ 1080p
    Latency: < 20ms
    """, language="bash")
