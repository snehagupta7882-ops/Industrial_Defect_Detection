from image_processing import resize_image, normalize_image
import streamlit as st
import cv2
import numpy as np

st.title("Industrial Defect Detection")
st.subheader("Multi-Material Quality Inspection")
st.write("Capture a product image for inspection.")
st.header("Product Inspection")

# Step 1: Capture image from camera
image = st.camera_input("Capture Product Image")

if image is not None:

    # Step 2-4: Bytes -> NumPy -> OpenCV image
    image_bytes = image.getvalue()
    numpy_image = np.frombuffer(image_bytes, np.uint8)
    opencv_image = cv2.imdecode(numpy_image, cv2.IMREAD_COLOR)

    if opencv_image is None:
        st.error("Image could not be decoded. Please try again.")
        st.stop()

    # Step 5: Dimensions
    height, width, channels = opencv_image.shape

    # Step 6: BGR -> RGB
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # Step 7: Display image
    st.image(rgb_image, caption="Captured Product Image", width="stretch")

    # Step 8: Image info
    st.subheader("Image Information")
    st.write("Width:", width, "pixels")
    st.write("Height:", height, "pixels")
    st.write("Channels:", channels)
    st.success("Image processed successfully!")

    # Step 9: Select ROI
    st.subheader("Select Product Region")

    x1 = st.slider("Left Position", 0, width - 2, width // 4)
    x2 = st.slider("Right Position", x1 + 1, width, 3 * width // 4)
    y1 = st.slider("Top Position", 0, height - 2, height // 4)
    y2 = st.slider("Bottom Position", y1 + 1, height, 3 * height // 4)

    cropped_image = rgb_image[y1:y2, x1:x2]

    st.subheader("Cropped Product Image")
    st.image(cropped_image, caption="Selected Region of Interest", width="stretch")

    # Step 10: Resize
    resized_image = resize_image(cropped_image, (224, 224))

    st.subheader("Resized Image")
    st.image(resized_image, width=300)
    st.write("New Image Shape:", resized_image.shape)

    # Step 11: Normalize
    normalized_image = normalize_image(resized_image)

    st.subheader("Normalized Image")
    st.image(normalized_image, caption="Normalized Image", width=300)
    st.write("Minimum Pixel Value:", normalized_image.min())
    st.write("Maximum Pixel Value:", normalized_image.max())

    # Step 12: Noise reduction
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    st.subheader("Noise Reduced Image")
    st.image(blurred_image, caption="Gaussian Blur Applied", width=300)

    # Step 13: Compare
    st.subheader("Image Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.image(resized_image, caption="Original Resized Image", width="stretch")
    with col2:
        st.image(blurred_image, caption="Gaussian Blur Image", width="stretch")