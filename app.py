import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import plotly.express as px
import pandas as pd
import time
import os
from streamlit_echarts import st_echarts

# Set Page Configuration
st.set_page_config(
    page_title="Nike Shoe Authenticity Detector",
    page_icon="👟",
    layout="centered",
)

# Custom CSS for Dark Theme and Better UI
st.markdown("""
    <style>
        body {
            background-color: #1e1e2f;
            color: white;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #FF6347;
        }
        .sub-title {
            text-align: center;
            font-size: 18px;
            color: #bbb;
        }
        .upload-box {
            text-align: center;
            background-color: #222;
            padding: 15px;
            border-radius: 10px;
        }
        .result-box {
            text-align: center;
            background-color: #2a2a3d;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(255, 99, 71, 0.5);
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<p class='title'>👟 Nike Shoe Authenticity Detector</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Upload an image of a Nike shoe to check if it is original or counterfeit.</p>", unsafe_allow_html=True)

# Initialize the Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="0IwFXQMr4i2FQ4HpThQd"
)

# File Upload
st.markdown("<div class='upload-box'>📤 <b>Upload a Nike Shoe Image</b></div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Uploaded Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the Uploaded Image Temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show Progress Animation
    st.markdown("🔍 **Analyzing Image...**")
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.02)
        progress_bar.progress(percent_complete + 1)

    # Call API for Inference
    result = CLIENT.infer("temp_image.jpg", model_id="counterfeit-nike-shoes-detection/2")

    # Display JSON Response (Hidden)
    with st.expander("📝 Raw API Response"):
        st.json(result)

    # Extract Prediction Data
    if "predictions" in result and len(result["predictions"]) > 0:
        prediction = result["predictions"][0]
        class_name = prediction["class"]
        confidence = prediction["confidence"] * 100

        # Display Results in a Styled Box
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.success(f"✅ **Detected:** {class_name}")
        st.info(f"📊 **Confidence Level:** {confidence:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        # **Circular Gauge Chart**
        st.markdown("### 🎯 Confidence Level")
        options = {
            "series": [{
                "type": "gauge",
                "progress": {"show": True},
                "axisLine": {"lineStyle": {"width": 20}},
                "pointer": {"length": "80%", "width": 5},
                "detail": {"formatter": f"{confidence:.2f}%"},
                "data": [{"value": confidence, "name": "Confidence"}]
            }]
        }
        st_echarts(options=options, height="300px")

        # **Confidence Level Bar Chart**
        df = pd.DataFrame({
            "Metric": ["Confidence", "Remaining"],
            "Percentage": [confidence, 100 - confidence]
        })

        # Create Bar Chart
        bar_fig = px.bar(
            df,
            x="Percentage",
            y="Metric",
            text="Percentage",
            orientation='h',
            color="Metric",
            color_discrete_map={"Confidence": "green", "Remaining": "red"},
        )

        bar_fig.update_traces(texttemplate='%{text:.2f}%', textposition='inside')
        bar_fig.update_layout(
            title="📊 Confidence Breakdown",
            xaxis_title="Percentage",
            yaxis_title="",
            template="plotly_dark"
        )

        st.plotly_chart(bar_fig, use_container_width=True)

    else:
        st.error("❌ No shoe detected. Please try another image.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#bbb;'>Developed by Md Saad</p>", unsafe_allow_html=True)

# Delete temp image after processing
if os.path.exists("temp_image.jpg"):
    os.remove("temp_image.jpg")
