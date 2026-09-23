
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import cv2
import numpy as np
import io
from datetime import datetime

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SnapSort AI",
    page_icon="🔍",
    layout="wide"
)

# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# =========================================================
# DETECTION HISTORY
# =========================================================

if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

# =========================================================
# HEADER
# =========================================================

st.title("🔍 SnapSort AI")
st.subheader("Intelligent Visual Understanding System")

st.write(
    "Detect, count, understand, protect and analyze visual information using AI."
)

st.divider()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ SnapSort Controls")

    privacy_mode = st.toggle(
        "🔐 Privacy Mode",
        value=False,
        help="Blur detected people for privacy."
    )

    st.divider()

    st.subheader("📊 Session History")

    st.metric(
        "Analyses",
        len(st.session_state.detection_history)
    )

    if st.button("🗑️ Clear History"):

        st.session_state.detection_history = []

        st.success("Detection history cleared.")

# =========================================================
# INPUT MODE
# =========================================================

mode = st.radio(
    "Choose input method:",
    ["📁 Upload Image", "📷 Use Camera"],
    horizontal=True
)

image = None

if mode == "📁 Upload Image":

    file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if file:
        image = Image.open(file).convert("RGB")

else:

    camera = st.camera_input("Take a picture")

    if camera:
        image = Image.open(camera).convert("RGB")

# =========================================================
# PROCESS IMAGE
# =========================================================

if image is not None:

    # -----------------------------------------------------
    # YOLO DETECTION
    # -----------------------------------------------------

    results = model(image, conf=0.25)

    boxes = results[0].boxes

    # -----------------------------------------------------
    # PRIVACY PROCESSING
    # -----------------------------------------------------

    original = np.array(image)

    processed_image = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    if privacy_mode:

        for box in boxes:

            class_id = int(box.cls[0])

            object_name = model.names[class_id]

            if object_name == "person":

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                x1 = max(0, x1)
                y1 = max(0, y1)

                x2 = min(
                    processed_image.shape[1],
                    x2
                )

                y2 = min(
                    processed_image.shape[0],
                    y2
                )

                region = processed_image[
                    y1:y2,
                    x1:x2
                ]

                if region.size > 0:

                    blurred = cv2.GaussianBlur(
                        region,
                        (51, 51),
                        0
                    )

                    processed_image[
                        y1:y2,
                        x1:x2
                    ] = blurred

        displayed_image = cv2.cvtColor(
            processed_image,
            cv2.COLOR_BGR2RGB
        )

    else:

        displayed_image = results[0].plot()

    # =====================================================
    # IMAGE DISPLAY
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📷 Input Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        if privacy_mode:
            st.subheader("🔐 Privacy Protected")
        else:
            st.subheader("🤖 AI Detection")

        st.image(
            displayed_image,
            use_container_width=True
        )

    st.divider()

    # =====================================================
    # OBJECT ANALYSIS
    # =====================================================

    if len(boxes) == 0:

        st.warning(
            "No objects detected. Try another image."
        )

    else:

        object_names = []
        confidences = []

        for box in boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            object_name = model.names[class_id]

            object_names.append(object_name)

            confidences.append(confidence)

        counts = Counter(object_names)

        total_objects = len(object_names)

        unique_objects = len(set(object_names))

        average_confidence = (
            sum(confidences) /
            len(confidences)
        )

        # =================================================
        # SAVE TO HISTORY
        # =================================================

        history_entry = {
            "time": datetime.now().strftime(
                "%H:%M:%S"
            ),
            "objects": total_objects,
            "types": unique_objects,
            "confidence": average_confidence,
            "detected": ", ".join(
                f"{name} × {count}"
                for name, count in counts.items()
            )
        }

        # Prevent duplicate history entries during
        # Streamlit reruns for the same image
        if (
            not st.session_state.detection_history
            or st.session_state.detection_history[-1]["detected"]
            != history_entry["detected"]
        ):

            st.session_state.detection_history.append(
                history_entry
            )

        # =================================================
        # ANALYTICS
        # =================================================

        st.subheader("📊 Detection Analytics")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "🎯 Objects Detected",
                total_objects
            )

        with c2:

            st.metric(
                "🏷️ Object Types",
                unique_objects
            )

        with c3:

            st.metric(
                "📈 Avg Confidence",
                f"{average_confidence * 100:.1f}%"
            )

        st.divider()

        # =================================================
        # OBJECT COUNT
        # =================================================

        st.subheader("📦 Object Count")

        for name, count in counts.items():

            st.write(
                f"### 🔹 {name.title()} × {count}"
            )

        st.divider()

        # =================================================
        # SMART SCENE
        # =================================================

        st.subheader("🧠 Smart Scene Understanding")

        object_description = ", ".join(
            f"{count} {name}"
            for name, count in counts.items()
        )

        if "person" in counts:

            people = counts["person"]

            people_text = (
                "1 person"
                if people == 1
                else f"{people} people"
            )

            scene_summary = (
                f"The image contains {people_text} "
                f"and the following detected objects: "
                f"{object_description}. "
                "The system has analyzed the scene "
                "using AI-based object detection."
            )

        else:

            scene_summary = (
                f"The image contains "
                f"{object_description}. "
                "No people were detected."
            )

        st.info(scene_summary)

        st.divider()

        # =================================================
        # DETECTION DETAILS
        # =================================================

        st.subheader("🎯 Detection Details")

        for i, box in enumerate(boxes):

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            object_name = model.names[class_id]

            st.write(
                f"**{i + 1}. {object_name.title()}**"
            )

            st.progress(confidence)

            st.caption(
                f"Confidence: {confidence * 100:.2f}%"
            )

        # =================================================
        # PDF REPORT
        # =================================================

        st.divider()

        st.subheader("📄 AI Detection Report")

        def create_report():

            buffer = io.BytesIO()

            pdf = canvas.Canvas(
                buffer,
                pagesize=A4
            )

            width, height = A4

            y = height - 50

            pdf.setFont(
                "Helvetica-Bold",
                20
            )

            pdf.drawString(
                50,
                y,
                "SnapSort AI - Detection Report"
            )

            y -= 40

            pdf.setFont(
                "Helvetica",
                11
            )

            pdf.drawString(
                50,
                y,
                "AI Model: YOLO11n"
            )

            y -= 20

            pdf.drawString(
                50,
                y,
                "Framework: Ultralytics"
            )

            y -= 20

            pdf.drawString(
                50,
                y,
                f"Privacy Mode: "
                f"{'Enabled' if privacy_mode else 'Disabled'}"
            )

            y -= 30

            pdf.setFont(
                "Helvetica-Bold",
                13
            )

            pdf.drawString(
                50,
                y,
                "Detection Summary"
            )

            y -= 25

            pdf.setFont(
                "Helvetica",
                11
            )

            pdf.drawString(
                50,
                y,
                f"Total Objects: {total_objects}"
            )

            y -= 20

            pdf.drawString(
                50,
                y,
                f"Object Types: {unique_objects}"
            )

            y -= 20

            pdf.drawString(
                50,
                y,
                f"Average Confidence: "
                f"{average_confidence * 100:.2f}%"
            )

            y -= 35

            pdf.setFont(
                "Helvetica-Bold",
                13
            )

            pdf.drawString(
                50,
                y,
                "Detected Objects"
            )

            y -= 25

            pdf.setFont(
                "Helvetica",
                11
            )

            for name, count in counts.items():

                pdf.drawString(
                    60,
                    y,
                    f"{name.title()}: {count}"
                )

                y -= 20

            y -= 20

            pdf.setFont(
                "Helvetica-Bold",
                13
            )

            pdf.drawString(
                50,
                y,
                "Scene Understanding"
            )

            y -= 25

            pdf.setFont(
                "Helvetica",
                10
            )

            words = scene_summary.split()

            line = ""

            for word in words:

                if len(line + " " + word) > 90:

                    pdf.drawString(
                        60,
                        y,
                        line
                    )

                    y -= 15

                    line = word

                else:

                    line += " " + word

            if line:

                pdf.drawString(
                    60,
                    y,
                    line
                )

            y -= 40

            pdf.setFont(
                "Helvetica-Oblique",
                9
            )

            pdf.drawString(
                50,
                y,
                "Generated by SnapSort AI"
            )

            pdf.save()

            buffer.seek(0)

            return buffer

        report = create_report()

        st.download_button(
            label="⬇️ Download AI Report",
            data=report,
            file_name="SnapSort_AI_Report.pdf",
            mime="application/pdf"
        )

# =========================================================
# DETECTION HISTORY
# =========================================================

st.divider()

st.subheader("📊 Detection History")

if len(st.session_state.detection_history) == 0:

    st.info(
        "No detection history yet. Analyze an image to create your first entry."
    )

else:

    for i, entry in enumerate(
        reversed(st.session_state.detection_history)
    ):

        with st.expander(
            f"Analysis {len(st.session_state.detection_history) - i} "
            f"— {entry['time']}"
        ):

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Objects",
                    entry["objects"]
                )

            with c2:
                st.metric(
                    "Object Types",
                    entry["types"]
                )

            with c3:
                st.metric(
                    "Confidence",
                    f"{entry['confidence'] * 100:.1f}%"
                )

            st.write(
                f"**Detected:** {entry['detected']}"
            )

# =========================================================
# AI SYSTEM
# =========================================================

st.divider()

st.subheader("⚡ AI System")

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Model",
        "YOLO11n"
    )

with c2:

    st.metric(
        "Framework",
        "Ultralytics"
    )

with c3:

    st.metric(
        "Vision Mode",
        "Object Detection"
    )

with c4:

    st.metric(
        "Privacy",
        "ON 🔐" if privacy_mode else "OFF"
    )

st.divider()

st.caption(
    "🔍 SnapSort AI | Intelligent Visual Understanding System"
)

st.caption(
    "⚡ Privacy-focused AI vision"
)
