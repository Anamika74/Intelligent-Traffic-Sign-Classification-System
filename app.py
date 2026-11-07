"""Indian Traffic Sign Recognition: Streamlit interface.

This file only draws the interface. All model logic lives in src/.
Run with:  streamlit run app.py
"""
import hashlib
import io
from pathlib import Path
import importlib
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.model import (DEFAULT_MODEL, MODEL_REGISTRY, get_device, load_class_names,
                       load_model, pretty_name)
from src.predict import run_inference

# Load voice support dynamically so static analyzers do not require the optional
# module to be present in every installation.
try:
    _voice = importlib.import_module("src.voice")
except ModuleNotFoundError as err:
    if err.name not in {"src", "src.voice"}:
        raise
    _voice = None

if _voice is None:
    class LocalVoice:
        @staticmethod
        def speak(_message):
            pass

    def alert_message(_label, name):
        return name

    def browser_speech(_message, autoplay=False):
        pass

    def should_alert(_session_state, _key):
        return False
else:
    LocalVoice = _voice.LocalVoice
    alert_message = _voice.alert_message
    browser_speech = _voice.browser_speech
    should_alert = _voice.should_alert

st.set_page_config(page_title="Indian Traffic Sign Recognition", page_icon="🚦", layout="wide")

SAMPLE_DIR = Path(__file__).parent / "assets" / "sample_images"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_FILES = sorted(p for p in SAMPLE_DIR.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


# ----------------------------------------------------------------------------
# Cached resources: the model is loaded once per server, results once per image.
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_model(model_name: str):
    cfg = MODEL_REGISTRY[model_name]
    device = get_device()
    model = load_model(cfg["path"], cfg["num_classes"], device)
    class_names = load_class_names(cfg["classes_file"], cfg["num_classes"])
    return model, class_names, device


@st.cache_data(show_spinner=False)
def cached_inference(model_name: str, image_bytes: bytes):
    model, class_names, device = get_model(model_name)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return run_inference(model, image, class_names, device)


# ----------------------------------------------------------------------------
# Sidebar: settings only. Results stay in the main area.
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", list(MODEL_REGISTRY), index=list(MODEL_REGISTRY).index(DEFAULT_MODEL))
    threshold = st.slider("Minimum confidence", 0.30, 0.95, 0.60, 0.05,
                          help="Predictions below this are flagged as unreliable and no voice alert is played.")
    voice_on = st.toggle("Voice alerts", value=True)
    if _voice is None:
        st.caption("Voice alerts are unavailable because src.voice could not be loaded.")
    st.caption("Switch models to compare them. The German-trained model scores 95% on "
               "German signs but only 54% on Indian versions of signs both countries share.")

st.title("Indian Traffic Sign Recognition")
st.caption("ResNet18 classifier with Grad-CAM explanations, trained to study the "
           "domain shift between European and Indian road signs.")

tab_predict, tab_experiments, tab_about = st.tabs(["Predict", "Experiments", "About"])

# ----------------------------------------------------------------------------
# Tab 1: Predict
# ----------------------------------------------------------------------------
with tab_predict:
    source = st.radio("Image source", ["Upload an image", "Choose a sample"], horizontal=True)
    image_bytes = None

    if source == "Upload an image":
        uploaded = st.file_uploader("Upload a cropped traffic sign (JPG or PNG)",
                                    type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
    else:
        if not SAMPLE_FILES:
            st.info("No sample images yet. Add a few cropped signs to `assets/sample_images/`.")
        else:
            cols = st.columns(min(len(SAMPLE_FILES), 6))
            for col, path in zip(cols, SAMPLE_FILES[:6]):
                with col:
                    st.image(str(path), width=120)
                    if st.button("Use", key=f"use_{path.name}", width="stretch"):
                        st.session_state["sample_path"] = str(path)
            chosen = st.session_state.get("sample_path")
            if chosen and Path(chosen).exists():
                image_bytes = Path(chosen).read_bytes()

    if image_bytes is None:
        st.info("Upload an image or pick a sample to see the prediction.")
    else:
        try:
            Image.open(io.BytesIO(image_bytes)).verify()
        except (UnidentifiedImageError, OSError):
            st.error("That file could not be read as an image. Try a JPG or PNG.")
            st.stop()

        try:
            get_model(model_name)
        except FileNotFoundError as err:
            st.error(str(err))
            st.stop()

        pred = cached_inference(model_name, image_bytes)
        confident = pred.confidence >= threshold
        shown_name = pretty_name(pred.label)

        left, right = st.columns(2)
        with left:
            st.subheader("Input image")
            st.image(image_bytes, width=160)
        with right:
            st.subheader("Where the model looked")
            st.image(pred.overlay, width=160)
            st.caption("Red and yellow areas influenced the prediction most.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted sign", shown_name if confident else "Not sure")
        m2.metric("Confidence", f"{pred.confidence:.1%}")
        m3.metric("Inference time", f"{pred.latency_ms:.0f} ms",
                  help="Forward pass plus Grad-CAM on this machine.")

        if pred.confidence >= 0.85:
            st.success(f"High confidence: {shown_name}")
        elif confident:
            st.warning(f"Moderate confidence: {shown_name}. Check the heatmap before trusting it.")
        else:
            st.error(f"Low confidence (best guess: {shown_name}). This may not be a supported sign, "
                     "or the image may be too blurry or cropped.")

        st.subheader("Top 3 predictions")
        top_df = pd.DataFrame({"Sign": [pretty_name(n) for n, _ in pred.top_k],
                               "Probability": [p for _, p in pred.top_k]}).set_index("Sign")
        st.bar_chart(top_df, horizontal=True)

        # Voice: speak once per new image, only when the model is confident.
        if voice_on and confident:
            key = (model_name, hashlib.md5(image_bytes).hexdigest())
            fire = should_alert(st.session_state, key)
            message = alert_message(pred.label, shown_name)
            browser_speech(message, autoplay=fire)
            if fire:
                LocalVoice.speak(message)

# ----------------------------------------------------------------------------
# Tab 2: Experiments (numbers come from notebooks/kaggle_retrain_and_evaluate.ipynb outputs)
# ----------------------------------------------------------------------------
with tab_experiments:
    st.subheader("Improving the model")
    st.caption("Three random seeds each, same train/validation split, test set used once per run. "
               "Values are mean ± spread across seeds.")
    compare = pd.DataFrame({
        "Setup": ["Original recipe (3 epochs, light augmentation)",
                  "Improved recipe, ImageNet start",
                  "Improved recipe, German-signs start"],
        "Accuracy (%)": ["88.1 ± 0.2", "96.5 ± 0.4", "96.2 ± 0.5"],
        "Macro-F1 (%)": ["67.4 ± 0.7", "93.7 ± 0.5", "92.8 ± 0.6"],
    })
    st.dataframe(compare, hide_index=True, width="stretch")
    st.bar_chart(pd.DataFrame({"Macro-F1 (%)": [67.4, 93.7, 92.8]}, index=compare["Setup"]), horizontal=True)
    st.markdown(
        "- The improved recipe adds flip-free augmentation, class weighting, label smoothing, "
        "a learning-rate schedule and 20 epochs.\n"
        "- Starting from German-sign weights or from ImageNet weights gives about the same result.\n"
        "- Macro-F1 counts every sign class equally, so it exposes weakness on rare signs "
        "that plain accuracy hides."
    )

    st.subheader("Does a German-trained model work on Indian signs?")
    shift = pd.DataFrame({
        "Model and test images": ["German model on German test set",
                                  "German model on Indian versions of 19 shared signs (562 images)",
                                  "Indian-trained model on the same Indian images"],
        "Accuracy (%)": [95.0, 54.4, 99.8],
    })
    st.dataframe(shift, hide_index=True, width="stretch")
    st.bar_chart(shift.set_index("Model and test images"), horizontal=True)
    st.markdown(
        "- Signs that look the same in both countries (stop, give way, cycle crossing) transfer well.\n"
        "- Speed limits, curves and traffic signals transfer badly, so local training data matters."
    )

# ----------------------------------------------------------------------------
# Tab 3: About
# ----------------------------------------------------------------------------
with tab_about:
    st.subheader("How it works")
    st.markdown(
        "1. The image is resized to 224×224 and normalised.\n"
        "2. A ResNet18 predicts one of the traffic sign classes.\n"
        "3. Grad-CAM uses the last convolutional block (`layer4`) to show which pixels mattered.\n"
        "4. Predictions below the confidence threshold are flagged instead of announced."
    )
    st.subheader("Limitations")
    st.markdown(
        "- Trained on cropped signs, so full road scenes need a sign detector first.\n"
        "- Blurry, tiny or heavily occluded signs reduce accuracy.\n"
        "- Signs outside the training classes will still receive a guess, which is why "
        "the confidence check matters."
    )
    st.caption("Built by Anamika Sharma · [GitHub](https://github.com/Anamika74/Intelligent-Traffic-Sign-Classification-System)")