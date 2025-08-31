import os
import io
import json
import base64
import requests
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, redirect, url_for, flash

# ---------------------------
# Config
# ---------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "supersecretkey_for_dev")
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Endpoints & keys (use environment variables in production)
# Endpoints & keys (always set via environment variables)
CLASSIFICATION_ENDPOINT = os.getenv("CLASSIFICATION_ENDPOINT")
CLASSIFICATION_KEY = os.getenv("CLASSIFICATION_KEY")

SEGMENTATION_ENDPOINT = os.getenv("SEGMENTATION_ENDPOINT")
SEGMENTATION_KEY = os.getenv("SEGMENTATION_KEY")

# TogetherAI (optional)
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY")
TOGETHER_AI_MODEL = os.getenv("TOGETHER_AI_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")

# HTTP headers for Azure endpoints
def azure_headers(key):
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }

# ---------------------------
# Utilities
# ---------------------------
def read_image_as_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def pil_to_base64_png(img_pil):
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def overlay_mask_on_image(image_pil, mask_arr, alpha=0.5):
    """
    image_pil: PIL image (RGB or L)
    mask_arr: 2D numpy array with values in [0,1] or [0,255]
    returns PIL PNG image (RGBA) of overlay
    """
    if image_pil.mode != "RGB":
        image_rgb = image_pil.convert("RGB")
    else:
        image_rgb = image_pil

    # normalize mask to 0..255
    if mask_arr.max() <= 1.0:
        mask_uint8 = (mask_arr * 255).astype(np.uint8)
    else:
        mask_uint8 = mask_arr.astype(np.uint8)

    # create colored mask (red)
    mask_img = Image.fromarray(mask_uint8).convert("L").resize(image_rgb.size, resample=Image.NEAREST)
    color_mask = Image.new("RGBA", image_rgb.size, (255, 0, 0, 0))
    color_mask.putalpha(mask_img)  # alpha channel = mask

    base = image_rgb.convert("RGBA")
    overlayed = Image.alpha_composite(base, color_mask)
    return overlayed

def try_post_json_with_fallback(endpoint, key, json_payload, files_payload=None, timeout=30):
    """
    Try sending JSON payload first; if the endpoint returns an error indicating unsupported input,
    try multipart/form-data upload if files_payload is provided.
    """
    headers = azure_headers(key)
    try:
        resp = requests.post(endpoint, json=json_payload, headers=headers, timeout=timeout)
    except Exception as e:
        return {"error": f"Request failed: {e}", "status_code": None}

    # try parse json
    try:
        resp_json = resp.json()
    except Exception:
        resp_json = {"raw_text": resp.text}

    # If it's successful (200..299) return
    if resp.ok:
        return {"response": resp_json, "status_code": resp.status_code}

    # If fallback files provided, try sending multipart
    if files_payload:
        try:
            # do not include json_content-type header for multipart
            headers2 = {"Authorization": f"Bearer {key}"} if key else {}
            resp2 = requests.post(endpoint, files=files_payload, headers=headers2, timeout=timeout)
            try:
                resp2_json = resp2.json()
            except Exception:
                resp2_json = {"raw_text": resp2.text}

            if resp2.ok:
                return {"response": resp2_json, "status_code": resp2.status_code}
            else:
                return {"error": "Both JSON and multipart failed", "json_response": resp_json, "multipart_response": resp2_json, "status_code": resp2.status_code}
        except Exception as e:
            return {"error": f"Multipart fallback failed: {e}", "status_code": None}

    return {"error": "Request failed", "response": resp_json, "status_code": resp.status_code}

def fetch_tumor_definition(tumor_type):
    if not TOGETHER_AI_API_KEY:
        return ""
    prompt = f"Give a concise medical definition of the brain tumor type: {tumor_type}. Don't mention AI."
    headers = {
        "Authorization": f"Bearer {TOGETHER_AI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": TOGETHER_AI_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        r = requests.post(TOGETHER_AI_ENDPOINT, json=data, headers=headers, timeout=20)
        if r.ok:
            js = r.json()
            # Together AI returns choices -> message -> content (may vary)
            return js.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        return ""
    except Exception:
        return ""

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        f = request.files["file"]
        if f.filename == "":
            flash("No selected file")
            return redirect(request.url)

        # Save uploaded file
        filename = f"{int(__import__('time').time())}_{f.filename}"
        safe_path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(safe_path)

        # encode to base64 for sending to endpoints
        img_b64 = read_image_as_base64(safe_path)
        json_payload = {"image": img_b64}

        # Classification
        clf = try_post_json_with_fallback(
            CLASSIFICATION_ENDPOINT,
            CLASSIFICATION_KEY,
            json_payload,
            files_payload={"file": open(safe_path, "rb")}
        )

        if "error" in clf and not clf.get("response"):
            # show error but continue to try segmentation
            flash(f"Classification error: {clf.get('error')}")
            clf_result = {}
        else:
            clf_result = clf.get("response", {})

        # Segmentation
        seg = try_post_json_with_fallback(
            SEGMENTATION_ENDPOINT,
            SEGMENTATION_KEY,
            json_payload,
            files_payload={"file": open(safe_path, "rb")}
        )

        if "error" in seg and not seg.get("response"):
            flash(f"Segmentation error: {seg.get('error')}")
            seg_result = {}
        else:
            seg_result = seg.get("response", {})

        # Attempt to parse classification: support multiple shapes
        predicted_label = None
        confidence = None
        # common responses: {"prediction": "glioma", "confidence": 0.93} or {"label": "glioma"} or {"predictions": [...]}
        if isinstance(clf_result, dict):
            predicted_label = clf_result.get("prediction") or clf_result.get("label") or clf_result.get("predicted_label")
            confidence = clf_result.get("confidence") or clf_result.get("score") or clf_result.get("probability")

            # Some classification implementations return {"outputs": [{"label":..., "confidence":...}]}
            if predicted_label is None and "outputs" in clf_result and isinstance(clf_result["outputs"], list) and clf_result["outputs"]:
                predicted_label = clf_result["outputs"][0].get("label")
                confidence = clf_result["outputs"][0].get("confidence")

        # Parse segmentation mask:
        mask_image_b64 = None
        mask_array = None

        if isinstance(seg_result, dict):
            # if endpoint returned base64 PNG mask
            if "mask" in seg_result and isinstance(seg_result["mask"], str):
                try:
                    mask_bytes = base64.b64decode(seg_result["mask"])
                    mask_pil = Image.open(io.BytesIO(mask_bytes)).convert("L")
                    overlay = overlay_mask_on_image(Image.open(safe_path).convert("RGB"), np.array(mask_pil)/255.0)
                    mask_image_b64 = pil_to_base64_png(overlay)
                except Exception:
                    # maybe mask was returned as list or array
                    pass

            # if endpoint returned numeric mask array
            elif "mask" in seg_result and isinstance(seg_result["mask"], list):
                try:
                    arr = np.array(seg_result["mask"])
                    # arr may have shape (1,H,W) or (H,W)
                    if arr.ndim == 3 and arr.shape[0] == 1:
                        arr = arr[0]
                    # normalize to 0..1
                    if arr.max() > 1:
                        arr = arr / 255.0
                    overlay = overlay_mask_on_image(Image.open(safe_path).convert("RGB"), arr)
                    mask_image_b64 = pil_to_base64_png(overlay)
                except Exception:
                    pass

            # if endpoint returned {"mask_base64": "..."} or {"mask_png": "..."}
            elif "mask_base64" in seg_result:
                try:
                    mask_bytes = base64.b64decode(seg_result["mask_base64"])
                    mask_pil = Image.open(io.BytesIO(mask_bytes)).convert("RGBA")
                    mask_image_b64 = pil_to_base64_png(mask_pil)
                except Exception:
                    pass

        # fallback: if segmentation failed, create placeholder overlay (original image)
        if mask_image_b64 is None:
            # just send original image as base64 for preview
            mask_image_b64 = read_image_as_base64(safe_path)

        # Tumor definition via TogetherAI (optional)
        tumor_def = ""
        if predicted_label:
            tumor_def = fetch_tumor_definition(predicted_label)

        return render_template(
            "result.html",
            filename=filename,
            predicted_label=predicted_label,
            confidence=confidence,
            tumor_def=tumor_def,
            overlay_image=mask_image_b64
        )

    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for('static', filename=f"uploads/{filename}"), code=301)


# Simple health route
@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # production server (gunicorn) will be used in App Service; debug only when running locally
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
