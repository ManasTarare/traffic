import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import easyocr
import re, time, os
from PIL import Image
import tempfile
import torch

# =====================================================
# CONFIG
# =====================================================
st.set_page_config("AI Traffic Challan System", layout="wide", page_icon="🚦")
st.title("🚦 AI Traffic Violation & Challan System")

IMG_SIZE = 64
DEDUP_SECONDS = 30
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
os.makedirs("snapshots", exist_ok=True)

# =====================================================
# INDIAN STATE CODES
# =====================================================
INDIAN_STATE_CODES = {
    "AP","AR","AS","BR","CG","GA","GJ","HR","HP","JH","KA","KL",
    "MP","MH","MN","ML","MZ","NL","OD","PB","RJ","SK","TN","TS",
    "TR","UK","UP","WB",
    "AN","CH","DD","DL","JK","LA","LD","PY"
}

# =====================================================
# SESSION STATE
# =====================================================
if "challans" not in st.session_state:
    st.session_state.challans = []

if "final_challans" not in st.session_state:
    st.session_state.final_challans = []

if "processed_plates" not in st.session_state:
    st.session_state.processed_plates = {}

if "prices" not in st.session_state:
    st.session_state.prices = {
        "Helmet Violation": 500,
        "Triple Seat": 1000,
        "Wrong Side": 1500
    }

if "place" not in st.session_state:
    st.session_state.place = "Unknown Location"

if "live_running" not in st.session_state:
    st.session_state.live_running = False

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    yolo = YOLO("yolov8s.pt")
    plate_model = YOLO("best.pt")
    helmet_model = tf.keras.models.load_model("helmet_classifier.h5")
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    return yolo, plate_model, helmet_model, reader

yolo, plate_model, helmet_model, reader = load_models()

# =====================================================
# PLATE CLEANING
# =====================================================
def clean_plate(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(text) < 8:
        return ""
    state = text[:2].replace('0', 'O')
    if state not in INDIAN_STATE_CODES:
        return ""
    mid = text[2:-4].replace('O', '0').replace('I', '1')
    last = text[-4:].replace('O', '0')
    return state + mid + last

# =====================================================
# UTILS
# =====================================================
def overlap(p, b):
    px1, py1, px2, py2 = p
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(px1, bx1), max(py1, by1)
    ix2, iy2 = min(px2, bx2), min(py2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = (px2 - px1) * (py2 - py1)
    return inter / (area + 1e-6)

def recently_seen(plate):
    now = time.time()
    last = st.session_state.processed_plates.get(plate, 0)
    if now - last < DEDUP_SECONDS:
        return True
    st.session_state.processed_plates[plate] = now
    return False

def enhance_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return [gray, enhanced, thresh]

def hard_ocr(reader, plate_img):
    texts = []
    for img in enhance_plate(plate_img):
        res = reader.readtext(img, detail=0)
        for t in res:
            c = clean_plate(t)
            if c:
                texts.append(c)
    return max(set(texts), key=texts.count) if texts else ""

# =====================================================
# FRAME PROCESSING
# =====================================================
def process_frame(frame):
    results = yolo(frame, conf=0.4, verbose=False)[0]
    bikes, persons = [], []

    for b in results.boxes:
        label = results.names[int(b.cls[0])]
        box = tuple(map(int, b.xyxy[0]))
        if label == "motorcycle":
            bikes.append(box)
        elif label == "person":
            persons.append(box)

    for bike in bikes:
        bx1, by1, bx2, by2 = bike
        riders = [p for p in persons if overlap(p, bike) > 0.3]
        violations = []

        if len(riders) > 2:
            violations.append("Triple Seat")

        for r in riders:
            rx1, ry1, rx2, ry2 = r
            head_h = (ry2 - ry1) // 3
            head = frame[ry1:ry1 + head_h, rx1:rx2]
            if head.size > 0:
                head = cv2.resize(head, (IMG_SIZE, IMG_SIZE)) / 255.0
                score = helmet_model.predict(np.expand_dims(head, 0), verbose=0)[0][0]
                if score < 0.7:
                    violations.append("Helmet Violation")
                    break

        if not violations:
            continue

        bike_crop = frame[by1:by2, bx1:bx2]
        if bike_crop.size == 0:
            continue

        plate_text = "UNKNOWN"
        plate_candidates = []

        plate_results = plate_model(bike_crop, conf=0.25, iou=0.4, verbose=False)[0]
        if plate_results.boxes:
            boxes = plate_results.boxes
            scores = boxes.conf.cpu().numpy()
            for idx in scores.argsort()[::-1][:3]:
                box = boxes[idx].xyxy[0].cpu().numpy().astype(int)
                px1, py1, px2, py2 = box
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(bike_crop.shape[1], px2), min(bike_crop.shape[0], py2)
                p_crop = bike_crop[py1:py2, px1:px2]
                if p_crop.size == 0:
                    continue
                text = hard_ocr(reader, p_crop)
                if text:
                    plate_candidates.append(text)

        if plate_candidates:
            plate_text = max(set(plate_candidates), key=plate_candidates.count)

        if plate_text != "UNKNOWN" and recently_seen(plate_text):
            continue

        timestamp = time.strftime("%d-%m-%Y %H:%M:%S")
        cv2.putText(bike_crop, timestamp, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        snap = f"snapshots/violation_{int(time.time()*1000)}.jpg"
        cv2.imwrite(snap, cv2.cvtColor(bike_crop, cv2.COLOR_RGB2BGR))

        st.session_state.challans.append({
            "image": snap,
            "plate": plate_text,
            "violations": list(set(violations)),
            "time": timestamp
        })

# =====================================================
# UI
# =====================================================
tab1, tab2, tab3 = st.tabs(["🎥 Detection", "🧾 Challans", "🔐 Admin"])

# ---------------- TAB 1 ----------------
with tab1:
    src = st.radio("Source", ["Image", "Video", "Live Camera"])
    frame_box = st.empty()

    if src == "Live Camera":
        cam_url = st.text_input("Camera / RTSP URL", value="0")
        c1, c2 = st.columns(2)
        if c1.button("▶ Start Live"):
            st.session_state.live_running = True
        if c2.button("⏹ Stop Live"):
            st.session_state.live_running = False

        if st.session_state.live_running:
            cam = int(cam_url) if cam_url.isdigit() else cam_url
            cap = cv2.VideoCapture(cam)
            f = 0
            while cap.isOpened() and st.session_state.live_running:
                ret, frame = cap.read()
                if not ret:
                    break
                f += 1
                if f % 5 != 0:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                process_frame(rgb)
                frame_box.image(rgb, width=900)
            cap.release()

    else:
        upload = st.file_uploader("Upload file", type=["jpg", "png", "mp4", "avi"])
        if upload:
            if src == "Image":
                img = np.array(Image.open(upload).convert("RGB"))
                frame_box.image(img, width=900)
                if st.button("Process Image"):
                    process_frame(img)
            else:
                tmp = tempfile.NamedTemporaryFile(delete=False)
                tmp.write(upload.read())
                cap = cv2.VideoCapture(tmp.name)
                if st.button("Start Video"):
                    i = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if i % 5 == 0:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            process_frame(rgb)
                            frame_box.image(rgb, width=900)
                        i += 1
                    cap.release()

# ---------------- TAB 2 ----------------
with tab2:
    if not st.session_state.challans:
        st.info("No challans yet.")
    else:
        for i, c in enumerate(st.session_state.challans):
            col1, col2 = st.columns([1, 2])
            col1.image(c["image"], width=250)

            plate_in = col2.text_input("Number Plate", c["plate"], key=f"p{i}")
            if plate_in:
                cleaned = clean_plate(plate_in)
                if cleaned:
                    c["plate"] = cleaned

            col2.write(f"Violations: {', '.join(c['violations'])}")
            col2.write(f"Time: {c['time']}")
            col2.write(f"Place: {st.session_state.place}")
            total = sum(st.session_state.prices[v] for v in c["violations"])
            col2.write(f"Total Fine: ₹{total}")

            b1, b2 = col2.columns(2)
            if b1.button("Confirm & Send", key=f"s{i}"):
                st.session_state.final_challans.append({
                    "plate": c["plate"],
                    "violations": c["violations"],
                    "time": c["time"],
                    "place": st.session_state.place,
                    "amount": total,
                    "image": c["image"]
                })
                st.session_state.challans.pop(i)
                st.rerun()

            if b2.button("Remove", key=f"r{i}"):
                st.session_state.challans.pop(i)
                st.rerun()

# ---------------- TAB 3 ----------------
with tab3:
    pwd = st.text_input("Admin Password", type="password")
    if pwd == ADMIN_PASSWORD:
        st.session_state.place = st.text_input("Location", st.session_state.place)
        for k in st.session_state.prices:
            st.session_state.prices[k] = st.number_input(k, st.session_state.prices[k], step=100)

        colA, colB, colC = st.columns(3)
        if colA.button("Remove ALL Pending"):
            st.session_state.challans.clear()
            st.rerun()
        if colB.button("Remove ALL Issued"):
            st.session_state.final_challans.clear()
            st.rerun()
        if colC.button("FULL RESET"):
            st.session_state.challans.clear()
            st.session_state.final_challans.clear()
            st.session_state.processed_plates.clear()
            st.rerun()
