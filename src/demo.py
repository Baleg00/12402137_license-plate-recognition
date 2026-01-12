from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import easyocr
from PIL import Image

from training import UNetSmall, load_model
from helpers import *
from eval_ocr import *


@st.cache_resource
def get_reader(langs: tuple[str, ...], use_gpu: bool) -> easyocr.Reader:
    return easyocr.Reader(list(langs), gpu=use_gpu)


st.set_page_config(page_title="Plate Segmentation + OCR", layout="wide")
st.title("License Plate Segmentation + OCR (Python Web App)")

with st.sidebar:
    st.header("Model")
    ckpt_path = st.text_input(
        "Checkpoint path (.pth/.pt)", value="checkpoints/best_cbam.pth"
    )
    attention = st.selectbox("Attention", options=["none", "se", "cbam"], index=2)
    base_ch = st.number_input("base_ch", min_value=8, max_value=128, value=32, step=8)
    input_h = st.number_input(
        "Input H", min_value=128, max_value=1024, value=512, step=32
    )
    input_w = st.number_input(
        "Input W", min_value=128, max_value=1024, value=512, step=32
    )
    thr = st.slider(
        "Segmentation threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05
    )

    st.header("OCR")
    langs = st.multiselect(
        "EasyOCR languages", options=["en", "de", "fr", "es", "it"], default=["en"]
    )
    ocr_height = st.number_input(
        "OCR resize height", min_value=24, max_value=96, value=48, step=8
    )

    st.header("Run")
    run_btn = st.button("Run inference", type="primary")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

# Read uploaded image
pil = Image.open(uploaded).convert("RGB")
img_rgb = np.array(pil)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# Show input immediately
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.subheader("Input")
    st.image(img_rgb, use_container_width=True)

if not run_btn:
    st.stop()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = device.type == "cuda"

ckpt = Path(ckpt_path)
if not ckpt.is_file():
    st.error(f"Checkpoint not found: {ckpt}")
    st.stop()

with st.spinner("Loading model..."):
    net = UNetSmall(in_ch=3, base_ch=int(base_ch), attention=attention).to(device)
    net = load_model(net, ckpt, device=device, strict=True)
    net.eval()

with st.spinner("Running segmentation..."):
    mask = predict_mask(
        net, img_bgr, device, input_hw=(int(input_h), int(input_w)), thr=float(thr)
    )

overlay = overlay_mask(img_rgb, mask)

with c2:
    st.subheader("Overlay")
    st.image(overlay, use_container_width=True)

plate = rectify_and_crop_plate(img_bgr, mask)

if plate is None:
    with c3:
        st.subheader("Rectified crop")
        st.warning("No plate crop could be extracted.")
    with c4:
        st.subheader("OCR prediction")
        st.write("(none)")
    st.stop()

plate_rgb = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)

with c3:
    st.subheader("Rectified crop")
    st.image(plate_rgb, use_container_width=True)

ocr_img = enhance_for_ocr(plate, out_h=int(ocr_height))

with st.spinner("Running EasyOCR..."):
    reader = get_reader(tuple(langs), use_gpu=use_gpu)
    results = reader.readtext(ocr_img, detail=1, paragraph=False)

pred = ""
conf = 0.0
if results:
    best = max(results, key=lambda x: float(x[2]))
    pred = normalize_plate_text(best[1])
    conf = float(best[2])

with c4:
    st.subheader("OCR prediction")
    st.write(f"**{pred or '(empty)'}**")
    st.caption(f"confidence={conf:.3f}")

st.markdown("---")
st.subheader("OCR input (post-processed)")
st.image(ocr_img, clamp=True, use_container_width=False)
