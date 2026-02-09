import streamlit as st
import cv2
import numpy as np
import pyaudio
import librosa
import threading
import time
from ultralytics import YOLO
import sentinel_hub  # Integration
import sentinel_auth # Authentication

# --- Configuration ---
# YOLO_MODEL = "yolov8n.pt" # dynamic now
CROWD_DENSITY_HIGH = 5
AUDIO_RATE = 22050
AUDIO_CHUNK = 1024

# --- Helper Logic ---
def analyze_audio_chunk(stream):
    try:
        # Read without blocking too long?
        if stream.get_read_available() >= AUDIO_CHUNK:
            data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            y = np.frombuffer(data, dtype=np.float32)
            
            rms = np.mean(librosa.feature.rms(y=y))
            cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=AUDIO_RATE))
            
            # Simple Thresholds
            if rms > 0.05 and cent > 2000:
                return "PANIC"
    except:
        pass
    return "NORMAL"

class VideoIngest:
    """
    Robust Video Capture that runs in a separate thread.
    Handles auto-reconnection for RTMP/RTSP streams.
    """
    def __init__(self, source):
        self.source = source
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.start()

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            # 1. Connect
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                time.sleep(2.0) # Retry delay
                continue
            
            # 2. Consume
            while not self.stopped and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break # Stream dropped/ended
                
                # Keep only the latest frame
                with self.lock:
                    self.frame = frame
            
            # 3. Disconnect & Retry
            cap.release()
            if not self.stopped:
                time.sleep(1.0) # Brief pause before reconnect

    def read(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.stopped = True

def get_risk_level(audio_status, person_count):
    if audio_status == "PANIC" and person_count >= CROWD_DENSITY_HIGH:
        return "HIGH", "red"
    elif audio_status == "PANIC" or person_count >= CROWD_DENSITY_HIGH:
        return "MEDIUM", "orange"
    else:
        return "LOW", "green"

# --- Authentication UI ---
def login_screen():
    st.title("🔐 Sentinel Login")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if sentinel_auth.verify_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        with st.form("signup_form"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            submitted_signup = st.form_submit_button("Sign Up")
            
            if submitted_signup:
                if new_pass != confirm_pass:
                    st.error("Passwords do not match")
                elif len(new_pass) < 4:
                    st.error("Password too short")
                else:
                    if sentinel_auth.create_user(new_user, new_pass):
                        st.success("Account created! Please login.")
                    else:
                        st.error("Username already exists")

# --- Main Dashboard Logic ---
def sentinel_dashboard():
    # --- Sidebar Settings ---
    # Logout Button
    with st.sidebar:
        st.write(f"User: **{st.session_state.username}**")
        if st.button("Logout", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        st.divider()

    # 1. Input Source
    st.sidebar.subheader("Input Source")
    input_type = st.sidebar.selectbox("Source Type", ["Live Webcam", "Phone Camera (IP Webcam)", "Upload File"])
    
    ipcam_url = ""
    
    if input_type == "Phone Camera (IP Webcam)":
        ipcam_url = st.sidebar.text_input("IP Webcam URL", "http://192.168.1.xxx:8080/video")
        st.sidebar.caption("Download 'IP Webcam' app on phone. Start server. Enter correct IP.")

    # 2. AI Settings (Authority Config)
    st.sidebar.divider()
    st.sidebar.subheader("AI Configuration")
    model_type = st.sidebar.selectbox(
        "YOLO Pose Model", 
        ["Fast (Nano)", "Balanced (Small)", "High Accuracy (Large)"], 
        index=0
    )
    
    # Hardcoded "Best" Defaults
    conf_thresh = 0.25
    img_size = 640
    iou_thresh = 0.45
    
    model_map = {
        "Fast (Nano)": "yolov8n-pose.pt",
        "Balanced (Small)": "yolov8s-pose.pt",
        "High Accuracy (Large)": "yolov8l-pose.pt"
    }
    selected_model = model_map[model_type]
    
    # ROI Defaults (Disabled/Full Screen)
    use_roi = False
    roi_top, roi_bottom, roi_left, roi_right = 0.0, 1.0, 0.0, 1.0

    # --- Header ---
    st.title("🛡️ Sentinel Authority Dashboard")
    st.markdown("**Real-time Fusion of Computer Vision & Audio Analysis**")

    # --- Loading Resources ---
    @st.cache_resource
    def load_model(model_name):
        return YOLO(model_name)
    
    model = load_model(selected_model)
    hub = sentinel_hub.get_hub()

    # --- Execution Logic ---
    if input_type in ["Live Webcam", "Phone Camera (IP Webcam)"]:
        start_btn = st.button("Start System", type="primary")
        stop_btn = st.button("Stop System")
        
        if start_btn: st.session_state.running = True
        if stop_btn: st.session_state.running = False

        if st.session_state.running:
            # Layout Selection (Authority Only)
            video_placeholder = st.empty()
            c1, c2, c3, c4 = st.columns(4)
            m_people = c1.empty()
            m_skel = c2.empty()
            m_audio = c3.empty()
            m_risk = c4.empty()
            
            # Init Setup
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32, channels=1, rate=AUDIO_RATE, input=True, frames_per_buffer=AUDIO_CHUNK)
            
            cap = None      # For standard cv2 (Webcam/File)
            ingest = None   # For Robust Network Stream
            
            if input_type in ["Live Webcam", "Upload File"]:
                src = 0 if input_type == "Live Webcam" else None
                if input_type == "Live Webcam": cap = cv2.VideoCapture(0)
                
            elif input_type == "Phone Camera (IP Webcam)":
                # Use Robust Ingest for network stream
                ingest = VideoIngest(ipcam_url)
            
            panic_counter = 0

            while st.session_state.running:
                
                # --- Frame Capture ---
                frame = None
                
                if ingest:
                    # Robust Stream
                    frame = ingest.read()
                    if frame is None:
                        # Stream connects asynchronously. Show spinner or wait.
                        # Don't break loop, just wait for connection.
                        time.sleep(0.1)
                        continue 
                        
                elif cap:
                    ret, frame_read = cap.read()
                    if ret: frame = frame_read
                    else:
                        st.error("Camera disconnected.")
                        break
                    
                if frame is None:
                    continue # Skip processing if no frame yet
                
                # --- ROI Logic ---
                h, w, _ = frame.shape
                # Define User Zone in pixels
                z_y1, z_y2 = int(h * roi_top), int(h * roi_bottom)
                z_x1, z_x2 = int(w * roi_left), int(w * roi_right)
                
                # Draw ROI box on frame for Authority (Only if enabled)
                if use_roi:
                    cv2.rectangle(frame, (z_x1, z_y1), (z_x2, z_y2), (255, 200, 0), 2)
                
                # Inference
                results = model(frame, verbose=False, classes=[0], conf=conf_thresh, imgsz=img_size, iou=iou_thresh)
                annotated_frame = frame
                
                # Count people INSIDE the ROI only
                person_count = 0
                skeleton_detected = "NO"
                
                for r in results:
                    annotated_frame = r.plot()
                    if r.boxes is not None:
                        for box in r.boxes:
                            # Check center point
                            bx1, by1, bx2, by2 = box.xyxy[0]
                            cx, cy = (bx1+bx2)/2, (by1+by2)/2
                            
                            # Filter
                            if (z_x1 <= cx <= z_x2) and (z_y1 <= cy <= z_y2):
                                person_count += 1
                                
                    if r.keypoints is not None and len(r.keypoints) > 0:
                        skeleton_detected = "YES"

                # Audio Analysis
                status_audio = analyze_audio_chunk(stream)
                if status_audio == "PANIC": panic_counter += 1
                else: panic_counter = max(0, panic_counter - 1)
                final_audio = "PANIC" if panic_counter > 2 else "NORMAL"

                # Fusion
                risk, color = get_risk_level(final_audio, person_count)
                
                # Hub Log
                hub.log_result(person_count, final_audio, risk)
                hub.broadcast_frame(annotated_frame)

                # --- Display ---
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Authority Display
                video_placeholder.image(frame_rgb, channels="RGB", width="stretch")
                
                m_people.metric("People Count", person_count)
                m_skel.metric("Skeleton Data", skeleton_detected)
                
                m_audio.metric("Audio Audio", final_audio, delta_color="inverse")
                m_risk.metric("Crowd Risk", risk, delta_color="inverse" if risk=="LOW" else "normal")

            # Cleanup
            if cap: cap.release()
            if ingest: ingest.stop()
            stream.stop_stream()
            stream.close()
            p.terminate()

    elif input_type == "Upload File":
        st.subheader("📂 File Analysis")
        st.info("File upload mode runs in Authority-style visualization by default.")
        
        col1, col2 = st.columns(2)
        v_file = col1.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        a_file = col2.file_uploader("Upload Audio (Optional)", type=['wav', 'mp3'])
        
        if v_file and st.button("Analyze File"):
            # Save and Process Logic (Reuse simpler version or Refactor)
            # For brevity, implementing a simplified version here mirroring the live loop logic
            v_path = f"temp_uploads/{v_file.name}"
            with open(v_path, "wb") as f: f.write(v_file.getbuffer())
            
            y_audio = None
            if a_file:
                a_path = f"temp_uploads/{a_file.name}"
                with open(a_path, "wb") as f: f.write(a_file.getbuffer())
                y_audio, sr_audio = librosa.load(a_path, sr=AUDIO_RATE)
            
            cap = cv2.VideoCapture(v_path)
            st_vid = st.empty()
            st_metrics = st.empty()
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                results = model(frame, verbose=False, classes=[0], conf=conf_thresh)
                annotated_frame = results[0].plot()
                count = len(results[0].boxes)
                
                # Simple Audio Check from file
                stat_aud = "NORMAL"
                if y_audio is not None:
                    # Time based check... (Simplified for now)
                    pass
                
                # Display
                st_vid.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                st_metrics.markdown(f"**People:** {count} | **Audio:** {stat_aud}")
                hub.broadcast_frame(annotated_frame)
                frame_idx += 1

            cap.release()

# --- Main/Entry Point ---
def main():
    st.set_page_config(page_title="Sentinel Integrated System", layout="wide")
    
    # Init Session State
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'running' not in st.session_state:
        st.session_state.running = False
        
    # Router
    if st.session_state.authenticated:
        sentinel_dashboard()
    else:
        login_screen()

if __name__ == "__main__":
    main()
