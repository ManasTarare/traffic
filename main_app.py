import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ultralytics import YOLO

# ---- Pylance builtins workaround (editor-only) ----
from builtins import max, min, int, len, map, float, bytearray

# ===================== LOAD MODELS =====================
@st.cache_resource
def load_models():
    person_bike_model = YOLO("yolov8s.pt")
    helmet_classifier = tf.keras.models.load_model("helmet_classifier.h5")
    return person_bike_model, helmet_classifier

person_bike_model, helmet_classifier = load_models()

IMG_SIZE = 64
HELMET_THRESHOLD = 0.5  # < 0.5 = HELMET, >= 0.5 = NO HELMET

# ===================== UTILS =====================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter)

# ===================== UI =====================
st.title("🚦 Helmet Violation Detection")
st.write("Detects **bike riders without helmets** using YOLO + CNN")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ===================== DETECTION =====================
    results = person_bike_model(frame, conf=0.4)[0]

    persons = []
    bikes = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "person":
            persons.append((x1, y1, x2, y2))
        elif label == "motorcycle":
            bikes.append((x1, y1, x2, y2))

    # ===================== FIND RIDERS =====================
    riders = []

    for bike in bikes:
        best_person = None
        best_overlap = 0

        for person in persons:
            px1, py1, px2, py2 = person
            person_lower = (px1, py1 + (py2-py1)//2, px2, py2)
            overlap = iou(person_lower, bike)

            if overlap > best_overlap:
                best_overlap = overlap
                best_person = person

        if best_person and best_overlap > 0.05:
            riders.append((bike, best_person))

    helmet_count = 0
    no_helmet_count = 0
    violations = []

    # ===================== HELMET CHECK =====================
    for bike_box, person_box in riders:
        x1, y1, x2, y2 = person_box
        head_y2 = y1 + int(0.4 * (y2 - y1))
        head_crop = frame[y1:head_y2, x1:x2]

        if head_crop.size == 0:
            continue

        head_crop = cv2.resize(head_crop, (IMG_SIZE, IMG_SIZE))
        head_crop = head_crop / 255.0
        head_crop = np.expand_dims(head_crop, axis=0)

        helmet_prob = helmet_classifier.predict(head_crop, verbose=0)[0][0]

        if helmet_prob < HELMET_THRESHOLD:
            helmet_count += 1
            label = "HELMET"
            color = (0, 255, 0)
        else:
            no_helmet_count += 1
            label = "NO HELMET"
            color = (255, 0, 0)
            violations.append((bike_box, person_box))

        cv2.rectangle(frame_rgb, person_box[:2], person_box[2:], color, 2)
        cv2.putText(
            frame_rgb,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # Draw bikes
    for bike in bikes:
        cv2.rectangle(frame_rgb, bike[:2], bike[2:], (255, 255, 0), 2)

    # ===================== DISPLAY =====================
    st.subheader("Result")
    st.image(frame_rgb, channels="RGB", use_column_width=True)

    st.subheader("Counts")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bikes", len(bikes))
    col2.metric("Persons", len(persons))
    col3.metric("Helmet", helmet_count)
    col4.metric("No Helmet", no_helmet_count)

    st.metric("🚨 Violations", len(violations))
