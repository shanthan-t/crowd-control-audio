import streamlit as st
import cv2
import numpy as np
import pyaudio
import librosa
import threading
import time
from ultralytics import YOLO
import sentinel_hub  # Integration

# --- Configuration ---
# YOLO_MODEL = "yolov8n.pt" # dynamic now
CROWD_DENSITY_HIGH = 5
AUDIO_RATE = 22050
AUDIO_CHUNK = 1024

# --- State Management ---
if 'audio_status' not in st.session_state:
    st.session_state.audio_status = "NORMAL"
if 'running' not in st.session_state:
    st.session_state.running = False

# --- Audio Thread ---
def audio_listener():
    """
    Background thread to listen to microphone and update session state.
    Note: Streamlit session state is not thread-safe in the usual way, 
    so we use a global or a mutable object if we want to share data? 
    Actually, threads spawned by Streamlit re-run the script. 
    It's tricky.
    
    Better approach for Streamlit Live Loop:
    Run the logic INSIDE the main loop frame-by-frame.
    Audio needs to be non-blocking.
    
    We will use PyAudio non-blocking callback or just read small chunks in the loop.
    For this prototype, let's try to read audio in the loop.
    """
    pass 

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

# --- Main App ---
def main():
    st.set_page_config(page_title="Sentinel Integrated System", layout="wide")
    
    # --- Sidebar Settings ---
    st.sidebar.title("Sentinel Config")
    
    # 1. View Mode
    view_mode = st.sidebar.radio("View Mode", ["Authority Dashboard", "Public Safety View"])
    
    # 2. Input Source
    st.sidebar.divider()
    st.sidebar.subheader("Input Source")
    input_type = st.sidebar.selectbox("Source Type", ["Live Webcam", "RTMP Stream", "RTSP Stream", "HTTP Snapshot", "Upload File"])
    
    rtsp_url = ""
    rtmp_url = ""
    http_url = ""
    snapshot_interval = 0.5
    
    if input_type == "RTMP Stream":
        rtmp_url = st.sidebar.text_input("RTMP URL", "rtmp://192.168.1.xxx/live/stream")
    elif input_type == "RTSP Stream":
        rtsp_url = st.sidebar.text_input("RTSP URL", "rtsp://admin:pass@192.168.1.xxx:554/cam/realmonitor?channel=1&subtype=0")
    elif input_type == "HTTP Snapshot":
        http_url = st.sidebar.text_input("Snapshot URL", "http://admin:pass@192.168.1.xxx/cgi-bin/snapshot.cgi")
        snapshot_interval = st.sidebar.slider("Poll Interval (sec)", 0.1, 2.0, 0.5)

    # 3. AI Settings (Only in Authority Mode)
    conf_thresh = 0.25
    img_size = 640
    iou_thresh = 0.45
    selected_model = "yolov8n-pose.pt"
    
    # ROI Defaults
    roi_top, roi_bottom, roi_left, roi_right = 0.0, 1.0, 0.0, 1.0

    if view_mode == "Authority Dashboard":
        st.sidebar.divider()
        st.sidebar.subheader("AI Parameters")
        model_type = st.sidebar.selectbox(
            "YOLO Pose Model", 
            ["Nano Pose (Fast)", "Small Pose (Balanced)", "Medium Pose (Accuracy)", "Large Pose (High Acc)", "Huge Pose (Best Acc)"], 
            index=0
        )
        conf_thresh = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.25, 0.05)
        img_size = st.sidebar.select_slider("Inference Resolution (px)", options=[640, 960, 1280], value=640)
        iou_thresh = st.sidebar.slider("NMS IOU Threshold", 0.1, 1.0, 0.45, 0.05)
        
        model_map = {
            "Nano Pose (Fast)": "yolov8n-pose.pt",
            "Small Pose (Balanced)": "yolov8s-pose.pt",
            "Medium Pose (Accuracy)": "yolov8m-pose.pt",
            "Large Pose (High Acc)": "yolov8l-pose.pt",
            "Huge Pose (Best Acc)": "yolov8x-pose.pt"
        }
        selected_model = model_map[model_type]
        
        # 4. Geofencing / ROI
        st.sidebar.divider()
        st.sidebar.subheader("Geofencing (ROI)")
        use_roi = st.sidebar.checkbox("Enable Zone Filtering")
        if use_roi:
            col_r1, col_r2 = st.sidebar.columns(2)
            roi_top = col_r1.slider("Top %", 0.0, 1.0, 0.0)
            roi_bottom = col_r1.slider("Bottom %", 0.0, 1.0, 1.0)
            roi_left = col_r2.slider("Left %", 0.0, 1.0, 0.0)
            roi_right = col_r2.slider("Right %", 0.0, 1.0, 1.0)

    # --- Header ---
    if view_mode == "Authority Dashboard":
        st.title("🛡️ Sentinel Authority Dashboard")
        st.markdown("**Real-time Fusion of Computer Vision & Audio Analysis**")
    else:
        st.title("📢 Public Safety Alert System")
        st.markdown("**Live Crowd Guidance & Status**")

    # --- Loading Resources ---
    @st.cache_resource
    def load_model(model_name):
        return YOLO(model_name)
    
    model = load_model(selected_model)
    hub = sentinel_hub.get_hub()

    # --- Execution Logic ---
    if input_type in ["Live Webcam", "RTMP Stream", "RTSP Stream", "HTTP Snapshot"]:
        start_btn = st.button("Start System", type="primary")
        stop_btn = st.button("Stop System")
        
        if start_btn: st.session_state.running = True
        if stop_btn: st.session_state.running = False

        if st.session_state.running:
            # Layout Selection
            if view_mode == "Authority Dashboard":
                video_placeholder = st.empty()
                c1, c2, c3, c4 = st.columns(4)
                m_people = c1.empty()
                m_skel = c2.empty()
                m_audio = c3.empty()
                m_risk = c4.empty()
            else:
                # Public View Layout
                status_header = st.empty()
                instruction_text = st.empty()
                video_placeholder = st.empty() # Optional: Show video? Maybe smaller.
            
            # Init Setup
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32, channels=1, rate=AUDIO_RATE, input=True, frames_per_buffer=AUDIO_CHUNK)
            
            cap = None      # For standard cv2 (Webcam/File)
            ingest = None   # For Robust Network Stream
            
            if input_type in ["Live Webcam", "Upload File"]:
                # Use standard blocking capture for local devices/files
                src = 0 if input_type == "Live Webcam" else None # Handle file later if needed
                if input_type == "Live Webcam": cap = cv2.VideoCapture(0)
                # (File upload logic is handled separately usually, but for now we keep webcam basic)
                
            elif input_type in ["RTSP Stream", "RTMP Stream"]:
                # Use Robust Ingest
                url = rtsp_url if input_type == "RTSP Stream" else rtmp_url
                # Clean URL logic
                if input_type == "RTSP Stream" and url.isdigit(): url = int(url)
                ingest = VideoIngest(url)
            
            panic_counter = 0

            while st.session_state.running:
                
                # --- Frame Capture ---
                frame = None
                
                if input_type == "HTTP Snapshot":
                    # ... Existing HTTP Logic ...
                    try:
                        import requests
                        resp = requests.get(http_url, timeout=2.0)
                        if resp.status_code == 200:
                            arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                            frame = cv2.imdecode(arr, -1)
                        else: time.sleep(0.5)
                    except: time.sleep(0.5)
                    time.sleep(snapshot_interval)
                    
                elif ingest:
                    # Robust Stream
                    frame = ingest.read()
                    if frame is None:
                        # Stream connects asynchronously. Show spinner or wait.
                        # Don't break loop, just wait for connection.
                        time.sleep(0.1)
                        # Optional: st.toast("Connecting to stream...")
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
                
                # Draw ROI box on frame for Authority
                if view_mode == "Authority Dashboard":
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
                
                if view_mode == "Authority Dashboard":
                    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    m_people.metric("People (In Zone)", person_count)
                    m_skel.metric("Skeleton", skeleton_detected)
                    m_audio.metric("Audio", final_audio, delta_color="inverse")
                    m_risk.markdown(f"### Risk: :{color}[{risk}]")
                else: # Public View
                    # Big Status
                    if risk == "LOW":
                        status_header.markdown(f"<h1 style='text-align: center; color: green; font-size: 80px;'>✅ SAFE</h1>", unsafe_allow_html=True)
                        instruction_text.markdown(f"<h2 style='text-align: center;'>Area is clear. Proceed normally.</h2>", unsafe_allow_html=True)
                    elif risk == "MEDIUM":
                        status_header.markdown(f"<h1 style='text-align: center; color: orange; font-size: 80px;'>⚠️ CAUTION</h1>", unsafe_allow_html=True)
                        instruction_text.markdown(f"<h2 style='text-align: center;'>High density detected. Please slow down.</h2>", unsafe_allow_html=True)
                    else: # HIGH
                        status_header.markdown(f"<h1 style='text-align: center; color: red; font-size: 80px;'>🚨 DANGER</h1>", unsafe_allow_html=True)
                        instruction_text.markdown(f"<h2 style='text-align: center;'>EMERGENCY DETECTED. FOLLOW EXITS CALMLY.</h2>", unsafe_allow_html=True)
                    
                    # Smaller video for public? Or just status?
                    # Typically public view doesn't show raw surveillance.
                    # But for demo, let's show it smaller or dimmed.
                    video_placeholder.image(frame_rgb, channels="RGB", width=400)

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

if __name__ == "__main__":
    main()
