import streamlit as st
import numpy as np
from PIL import Image

st.title("Average RGB Calculator")
st.write("Upload an image to calculate its average RGB values.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    arr = np.array(img)

    avg_r = float(arr[:, :, 0].mean())
    avg_g = float(arr[:, :, 1].mean())
    avg_b = float(arr[:, :, 2].mean())

    avg_color = (int(avg_r), int(avg_g), int(avg_b))
    hex_color = "#{:02X}{:02X}{:02X}".format(*avg_color)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    st.subheader("Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Red", f"{avg_r:.1f}")
    col2.metric("Green", f"{avg_g:.1f}")
    col3.metric("Blue", f"{avg_b:.1f}")

    st.write(f"**Hex:** `{hex_color}`")

    st.markdown(
        f"""
        <div style="
            width: 100%;
            height: 80px;
            background-color: {hex_color};
            border-radius: 8px;
            border: 1px solid #ccc;
        "></div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Average color swatch — {hex_color}")
