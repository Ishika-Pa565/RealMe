"""
app.py - Streamlit Web Application for RealMe Deepfake Detection.
Upload any face image, get an instant fakeness score and confidence rating.
"""

import io
import sys
from pathlib import Path
from PIL import Image
import streamlit as st

# Setup system path to access Code modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict import load_model, predict_image

# Page configuration
st.set_page_config(
    page_title="RealMe - Deepfake Detection in Social Media Content",
    page_icon="🛡️",
    layout="centered",
)


@st.cache_resource
def get_cached_detector():
    """Caches the PyTorch model in memory for fast repeated inference."""
    weights_path = PROJECT_ROOT / "models" / "best.pth"
    if not weights_path.exists():
        return None
    return load_model(weights_path=weights_path, device="cpu")


# Title and Header
st.markdown(
    "<h1 style='text-align: center; color: #1E3A8A;'>DEEP FAKE DETECTION IN SOCIAL MEDIA CONTENT</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #64748B; font-size: 1.1em;'>RealMe: AI-Powered Deepfake Image Analysis</p>",
    unsafe_allow_html=True,
)

# Detailed overview section
with st.expander("ℹ️ Understanding Deepfakes & How RealMe Works", expanded=False):
    st.write(
        """
        **Deepfakes** are synthetic media generated using deep learning algorithms (such as Generative Adversarial Networks) 
        where a person's likeness, facial expressions, or features are digitally swapped or altered. While they offer creative 
        potential in visual effects and education, they can also be weaponized for disinformation, identity fraud, and impersonation.
        
        **RealMe** leverages a fine-tuned **EfficientNet-B0** deep neural network with transfer learning to detect subtle 
        micro-artifacts, blending inconsistencies, and unnatural lighting anomalies across facial regions.
        """
    )

# Model status check
detector = get_cached_detector()
weights_file = PROJECT_ROOT / "models" / "best.pth"

if not weights_file.exists():
    st.warning(
        "⚠️ **Model Weights Missing (`models/best.pth`):**\n\n"
        "No trained weights file was found in the `models/` directory.\n"
        "- If you have trained your model on Google Colab or Kaggle, place `best.pth` into `RealMe/models/`.\n"
        "- Alternatively, you can run a quick local smoke-test to create temporary test weights:\n"
        "  `python Code/train.py --smoke-test`"
    )

st.markdown("### 📤 Upload Image for Inspection")
uploaded_file = st.file_uploader(
    "Choose a facial image to analyze...",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG (Max size: 10MB)",
)

if uploaded_file is not None:
    # Validate file size (10 MB limit)
    file_bytes = uploaded_file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        st.error("Error: Uploaded file exceeds the 10 MB limit.")
    else:
        # In-memory PIL conversion (never saved to disk)
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            try:
                st.image(image, caption="Uploaded Image", width="stretch")
            except TypeError:
                st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.subheader("Analysis Verdict")
            if detector is None:
                # Fallback to model loaded without custom weights if file not present
                temp_model = load_model(device="cpu")
                fake_prob = predict_image(image, temp_model, device="cpu")
                st.info("ℹ️ Using initialized baseline weights (untrained).")
            else:
                fake_prob = predict_image(image, detector, device="cpu")

            percentage = fake_prob * 100.0
            is_fake = fake_prob >= 0.5
            label = "Fake" if is_fake else "Real"
            color = "#DC2626" if is_fake else "#16A34A"

            st.markdown(
                f"<div style='background-color: {color}15; border: 2px solid {color}; border-radius: 10px; padding: 15px; text-align: center; margin-bottom: 15px;'>"
                f"<h2 style='color: {color}; margin: 0;'>The image is {label}</h2>"
                f"<h3 style='margin: 8px 0 0 0; color: #334155;'>Fakeness Confidence: {percentage:.1f}%</h3>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Progress meter for fakeness probability
            st.write("**Fakeness Probability Meter:**")
            st.progress(float(fake_prob))

            if is_fake:
                st.markdown(
                    "**Analysis Details:** Our deepfake detection model classified this image as **Fake** based on characteristic "
                    "synthetic manipulation patterns. These indicators include frequency discrepancies, blending boundary artifacts, "
                    "unnatural eye/skin reflection inconsistencies, or generative texture flaws."
                )
            else:
                st.markdown(
                    "**Analysis Details:** Our deepfake detection model classified this image as **Real**. The image exhibits coherent "
                    "biological texture, natural lighting distributions, and consistent anatomical landmark proportions typical of unmanipulated photos."
                )

        # Cautionary disclaimer (FR5)
        st.info(
            "⚠️ **Important Disclaimer:** The fakeness score represents statistical model confidence based on learned visual patterns. "
            "It should be used as an assistive tool and does not constitute absolute or legal proof."
        )

# Historical training performance section
st.markdown("---")
st.subheader("📊 Model Performance Benchmarks")
st.markdown(
    """
    - **Target In-Domain Accuracy**: ~95%
    - **Backbone Architecture**: EfficientNet-B0 (Transfer Learning)
    - **Loss Function**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)
    - **Optimization**: AdamW with Cosine Annealing Learning Rate
    """
)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #64748B; font-size: 0.9em;'>
        <b>RealMe Deepfake Detection Project</b> | Developed with PyTorch & Streamlit<br>
        <i>Based on reference architecture: Ramya-Mahi / DeepFake-Detection-and-Prevention-A-Comprehensive-approach-using-AI</i>
    </div>
    """,
    unsafe_allow_html=True,
)
