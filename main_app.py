import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import easyocr
import re, time, os
from collections import Counter
from PIL import Image

# =====================================================
# CONFIG
# =====================================================
st.set_page_config("AI Traffic Challan System", layout="wide")
st.title("🚦 AI Traffic Violation & Challan System")

IMG_SIZE = 64
os.makedirs("snapshots", exist_ok=True)

# =====================================================
# SESSION STATE INIT
# =====================================================
if "logs" not in st.session_state:
    st.session_state.logs = []

if "challans" not in st.session_state:
    st.session_state.challans = []

if "fines" not in st.session_state:
    st.session_state.fines = {
        "Helmet Violation": 500,
        "Triple Seat": 1000,
        "Wrong Side": 1500
    }

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

# =====================================================
# SIDEBAR – PROCESSING STEPS
# =====================================================
st.sidebar.title("🧠 Processing Steps")

def log(msg):
    st.session_state.logs.append(msg)
    st.sidebar.write(f"• {msg}")

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    log("Loading YOLOv8s (bike + person)")
    yolo = YOLO("yolov8s.pt")

    log("Loading number plate model (best.pt)")
    plate_model = YOLO("best.pt")

    log("Loading helmet classifier")
    helmet_model = tf.keras.models.load_model("helmet_classifier.h5")

    log("Loading EasyOCR")
    reader = easyocr.Reader(['en'], gpu=False)

    return yolo, plate_model, helmet_model, reader

yolo, plate_model, helmet_model, reader = load_models()

# =====================================================
# INDIAN PLATE LOGIC
# =====================================================
INDIAN_STATE_CODES = {
    "AP","AR","AS","BR","CG","GA","GJ","HR","HP","JH","KA","KL",
    "MP","MH","MN","ML","MZ","NL","OD","PB","RJ","SK","TN","TS",
    "TR","UK","UP","WB","AN","CH","DD","DL","JK","LA","LD","PY"
}

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

def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

# =====================================================
# HELPERS
# =====================================================
def overlap(p, b):
    px1,py1,px2,py2 = p
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(px1,bx1), max(py1,by1)
    ix2,iy2 = min(px2,bx2), min(py2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    area = (px2-px1)*(py2-py1)
    return inter/(area+1e-6)

def helmet_score(head):
    head = cv2.resize(head, (IMG_SIZE, IMG_SIZE)) / 255.0
    return helmet_model.predict(
        np.expand_dims(head, 0), verbose=0
    )[0][0]

# =====================================================
# PLATE EXTRACTION
# =====================================================
def extract_plate(bike_crop):
    log("Running number plate detection")

    h, w, _ = bike_crop.shape
    counter = Counter()

    results = plate_model(bike_crop, conf=0.2, verbose=False)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        log(f"Plate boxes detected: {len(results.boxes)}")
        for box in results.boxes.xyxy:
            x1,y1,x2,y2 = map(int, box)
            crop = bike_crop[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            proc = preprocess_plate(crop)
            texts = reader.readtext(proc, detail=0,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            log(f"OCR texts: {texts}")
            for t in texts:
                p = clean_plate(t)
                if p:
                    counter[p] += 1
    else:
        log("YOLO failed → heuristic plate box")
        hx1, hx2 = int(w*0.2), int(w*0.8)
        hy1, hy2 = int(h*0.6), int(h*0.9)
        crop = bike_crop[hy1:hy2, hx1:hx2]
        proc = preprocess_plate(crop)
        texts = reader.readtext(proc, detail=0,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        log(f"OCR texts (heuristic): {texts}")
        for t in texts:
            p = clean_plate(t)
            if p:
                counter[p] += 1

    if counter:
        plate = counter.most_common(1)[0][0]
        log(f"Final plate: {plate}")
        return plate

    log("Plate extraction failed")
    return "UNKNOWN"

# =====================================================
# CORE PIPELINE
# =====================================================
def process_frame(frame):
    log("Frame received")

    results = yolo(frame, conf=0.4, verbose=False)[0]
    bikes, persons = [], []

    for b in results.boxes:
        label = results.names[int(b.cls[0])]
        box = tuple(map(int, b.xyxy[0]))
        if label == "motorcycle":
            bikes.append(box)
        elif label == "person":
            persons.append(box)

    log(f"Bikes detected: {len(bikes)}")
    log(f"Persons detected: {len(persons)}")

    for bike in bikes:
        bx1,by1,bx2,by2 = bike
        riders = [p for p in persons if overlap(p, bike) > 0.3]
        log(f"Riders associated: {len(riders)}")

        violations = []

        if len(riders) > 2:
            violations.append("Triple Seat")
            log("Triple seat detected")

        for r in riders:
            rx1,ry1,rx2,ry2 = r
            head = frame[ry1:ry1+(ry2-ry1)//3, rx1:rx2]
            if head.size and helmet_score(head) < 0.5:
                violations.append("Helmet Violation")
                log("Helmet violation detected")
                break

        if not violations:
            log("No violations → skipping challan")
            continue

        bike_crop = frame[by1:by2, bx1:bx2]
        plate = extract_plate(bike_crop)

        path = f"snapshots/{int(time.time()*1000)}.jpg"
        cv2.imwrite(path, bike_crop)

        st.session_state.challans.append({
            "image": path,
            "plate": plate,
            "violations": violations
        })

# =====================================================
# UI
# =====================================================
tab1, tab2, tab3 = st.tabs(
    ["🎥 Detection", "🧾 Challan", "🔐 Admin"]
)

# ------------------ DETECTION ------------------
with tab1:
    upload = st.file_uploader("Upload image",
                              type=["jpg","png","jpeg"])

    if upload:
        frame = np.array(Image.open(upload).convert("RGB"))
        st.image(frame, width=350)

        if st.button("🚨 Run Detection"):
            st.session_state.logs.clear()
            process_frame(frame)
            st.success("Detection complete")

# ------------------ CHALLAN ------------------
with tab2:
    if not st.session_state.challans:
        st.info("No pending challans")
    else:
        cols = st.columns(3)
        remove_idx = None

        for i, ch in enumerate(st.session_state.challans):
            with cols[i % 3]:
                st.image(ch["image"], use_column_width=True)
                st.write("🚗 Plate:", ch["plate"])

                v1 = st.checkbox("Helmet Violation",
                                 value="Helmet Violation" in ch["violations"],
                                 key=f"h{i}")
                v2 = st.checkbox("Triple Seat",
                                 value="Triple Seat" in ch["violations"],
                                 key=f"t{i}")
                v3 = st.checkbox("Wrong Side", key=f"w{i}")

                place = st.text_input("Place", key=f"p{i}")
                time_ = st.text_input("Time", key=f"time{i}")

                selected = []
                if v1: selected.append("Helmet Violation")
                if v2: selected.append("Triple Seat")
                if v3: selected.append("Wrong Side")

                total = sum(st.session_state.fines[v] for v in selected)
                st.write(f"💰 **Total Fine: ₹{total}**")

                if st.button("✅ Send Challan", key=f"s{i}"):
                    remove_idx = i
                    st.success("Challan sent")

        if remove_idx is not None:
            st.session_state.challans.pop(remove_idx)
            st.rerun()

# ------------------ ADMIN ------------------
with tab3:
    st.subheader("🔐 Admin Panel")

    # ---- init admin creds in session ----
    if "admin_user" not in st.session_state:
        st.session_state.admin_user = ""

    if "admin_pass" not in st.session_state:
        st.session_state.admin_pass = ""

    if not st.session_state.admin_logged_in:
        st.text_input(
            "Username",
            key="admin_user"
        )
        st.text_input(
            "Password",
            type="password",
            key="admin_pass"
        )

        if st.button("Login"):
            if (
                st.session_state.admin_user == "admin"
                and st.session_state.admin_pass == "admin123"
            ):
                st.session_state.admin_logged_in = True
                st.success("Admin logged in")
                st.rerun()
            else:
                st.error("Invalid username or password")

    else:
        st.success("Logged in as admin")

        st.subheader("💰 Fine Rates")
        for k in st.session_state.fines:
            st.session_state.fines[k] = st.number_input(
                k,
                value=st.session_state.fines[k],
                step=100
            )

        if st.button("🗑 Hard Reset Pending Challans"):
            st.session_state.challans.clear()
            st.warning("All pending challans cleared")
