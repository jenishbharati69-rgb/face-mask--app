import streamlit as st
import numpy as np
from PIL import Image
import io
import os

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
#  STYLES
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }
  .hero h1 { font-size: 2.2rem; font-weight: 700; color: #e2e8f0; margin: 0 0 .4rem; }
  .hero p  { color: #94a3b8; font-size: 1rem; margin: 0; }

  .result-card {
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-top: 1.4rem;
    text-align: center;
    animation: fadeIn .4s ease;
  }
  @keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; } }

  .with-mask    { background: linear-gradient(135deg,#064e3b,#065f46); border:1px solid #10b981; }
  .without-mask { background: linear-gradient(135deg,#7f1d1d,#991b1b); border:1px solid #ef4444; }
  .incorrect    { background: linear-gradient(135deg,#78350f,#92400e); border:1px solid #f59e0b; }

  .result-label { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; }
  .result-sub   { color: #cbd5e1; font-size: .9rem; margin-top: .4rem; }
  .confidence   { font-size: 1rem; font-weight: 600; margin-top: .8rem; }

  .prob-bar-wrap { margin-top: 1.5rem; }
  .prob-row      { display:flex; align-items:center; margin:.35rem 0; gap:.7rem; }
  .prob-label    { width:180px; text-align:right; font-size:.85rem; color:#94a3b8; }
  .prob-bar-bg   { flex:1; height:10px; background:#1e293b; border-radius:99px; overflow:hidden; }
  .prob-bar-fill { height:100%; border-radius:99px; transition:width .6s ease; }
  .prob-pct      { width:42px; font-size:.8rem; color:#64748b; }

  .upload-hint { color:#64748b; font-size:.85rem; text-align:center; margin-top:.5rem; }

  .stButton>button {
    width:100%; border-radius:10px; height:2.8rem;
    background: linear-gradient(90deg,#3b82f6,#6366f1);
    color:white; font-weight:600; font-size:1rem; border:none;
    box-shadow: 0 2px 12px rgba(99,102,241,.4);
  }
  .stButton>button:hover { filter: brightness(1.1); }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  MODEL
#  The notebook (facemaskdetection_vijaykumar.ipynb) trains on:
#    vijaykumar1799/face-mask-detection  (Kaggle)
#  and saves as:  face_mask_classifier.keras
#
#  Keras assigns class indices alphabetically:
#    0 → mask_weared_incorrect
#    1 → with_mask
#    2 → without_mask
# ──────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["mask_weared_incorrect", "with_mask", "without_mask"]

CLASS_LABELS = {
    "with_mask":             ("😷 Mask On",           "with-mask",    "#10b981"),
    "without_mask":          ("🚫 No Mask",           "without-mask", "#ef4444"),
    "mask_weared_incorrect": ("⚠️ Incorrectly Worn", "incorrect",    "#f59e0b"),
}

CLASS_ADVICE = {
    "with_mask":             "Great job! Your mask is properly worn.",
    "without_mask":          "Please put on a mask to protect yourself and others.",
    "mask_weared_incorrect": "Adjust your mask to fully cover your nose and mouth.",
}

CLASS_NICE = {
    "with_mask":             "With Mask",
    "without_mask":          "Without Mask",
    "mask_weared_incorrect": "Incorrect Wear",
}

IMG_SIZE = (256, 256)   # must match training image_size=(256,256)


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Loads face_mask_classifier.keras (saved by the notebook).
    Falls back to .h5 if that exists instead.
    """
    try:
        import tensorflow as tf
        for model_path in ["face_mask_classifier.keras", "face_mask_classifier.h5"]:
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                return model, None
        return None, (
            "Model file not found.\n"
            "Place `face_mask_classifier.keras` in the same folder as this script, then restart."
        )
    except ImportError:
        return None, "TensorFlow is not installed. Run:  pip install tensorflow"
    except Exception as e:
        return None, str(e)


def preprocess(image: Image.Image) -> np.ndarray:
    """Resize to 256×256, normalise to [0,1], add batch dim."""
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image):
    """Returns (top_class_name, {class_name: probability})."""
    arr   = preprocess(image)
    preds = model.predict(arr, verbose=0)[0]          # shape (3,)
    probs = {name: float(preds[i]) for i, name in enumerate(CLASS_NAMES)}
    top   = max(probs, key=probs.get)
    return top, probs


# ──────────────────────────────────────────────────────────────────────────────
#  UI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>😷 Face Mask Detector</h1>
  <p>Upload a photo — the CNN will classify it into one of three categories</p>
</div>
""", unsafe_allow_html=True)

# Load model
model, model_err = load_model()

if model_err:
    st.warning(f"⚠️ **Model not loaded:** {model_err}")
    st.info("""
**How to fix:**
1. Run all cells in `facemaskdetection_vijaykumar.ipynb` on Google Colab
2. Download the `face_mask_classifier.keras` file it produces
3. Place it in the same folder as `mask_detector.py`
4. Run:  `streamlit run mask_detector.py`

The app will run in **demo mode** until the model is loaded.
    """)
    DEMO_MODE = True
else:
    DEMO_MODE = False
    st.success("✅ Model loaded — ready to classify!")

# File uploader
uploaded = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)
st.markdown(
    '<p class="upload-hint">JPG · PNG · WEBP — works best with a single face photo</p>',
    unsafe_allow_html=True,
)

if uploaded:
    image = Image.open(io.BytesIO(uploaded.read()))

    # Show image centred
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded image", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Classify Image"):
        with st.spinner("Analysing..."):
            if DEMO_MODE:
                top   = "with_mask"
                probs = {
                    "with_mask":             0.912,
                    "without_mask":          0.053,
                    "mask_weared_incorrect": 0.035,
                }
            else:
                top, probs = predict(model, image)

        label_str, css_cls, bar_color = CLASS_LABELS[top]
        advice   = CLASS_ADVICE[top]
        conf_pct = probs[top] * 100

        # ── Result card ───────────────────────────────────────────────────────
        st.markdown(f"""
<div class="result-card {css_cls}">
  <div class="result-label">{label_str}</div>
  <div class="result-sub">{advice}</div>
  <div class="confidence" style="color:{bar_color}">Confidence: {conf_pct:.1f}%</div>
</div>
""", unsafe_allow_html=True)

        # ── Probability bars ──────────────────────────────────────────────────
        bar_colors = {
            "with_mask":             "#10b981",
            "without_mask":          "#ef4444",
            "mask_weared_incorrect": "#f59e0b",
        }
        st.markdown('<div class="prob-bar-wrap">', unsafe_allow_html=True)
        for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
            pct   = prob * 100
            color = bar_colors[cls]
            st.markdown(f"""
<div class="prob-row">
  <span class="prob-label">{CLASS_NICE[cls]}</span>
  <div class="prob-bar-bg">
    <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
  </div>
  <span class="prob-pct">{pct:.1f}%</span>
</div>
""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if DEMO_MODE:
            st.caption("⚠️ Demo mode — place `face_mask_classifier.keras` here for real predictions.")

# ──────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#475569;font-size:.8rem;">'
    'CNN trained on the '
    '<a href="https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection" '
    'target="_blank" style="color:#6366f1;">vijaykumar1799 Face Mask Detection</a> dataset · '
    '3 classes: With Mask · Without Mask · Incorrectly Worn</p>',
    unsafe_allow_html=True,
)