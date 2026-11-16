import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load your trained model
model = YOLO("best.pt")

# Page configuration
st.set_page_config(
    page_title="Object Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Object Detection System")
st.write("Upload an image to detect and classify objects using our trained model.")

# Image uploader
uploaded_file = st.file_uploader(
    "Select an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(image, caption="Original Image", use_container_width=True)

    if st.button("Detect Objects", type="primary"):

        with st.spinner("Running model prediction..."):

            # Run prediction
            results = model.predict(image)

            # Draw bounding boxes
            detected_image = results[0].plot()

            # Convert BGR to RGB
            detected_image = detected_image[:, :, ::-1]

        st.success("Detection completed!")

        # Display results
        st.subheader("Detection Results")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

        with col2:
            st.image(
                detected_image,
                caption="Detected Objects",
                use_container_width=True
            )

        # Display detected classes
        st.subheader("Detected Classes")

        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                st.write(
                    f"**{class_name}** — Confidence: {confidence:.2%}"
                )