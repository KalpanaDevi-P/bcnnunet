from flask import Flask, render_template, request, redirect, url_for, session, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import numpy as np
import cv2
import os
from fpdf import FPDF
import requests

# ---------------------------
# 0. Config
# ---------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Together AI API
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY", "dummy_key")  # put in Azure secrets
TOGETHER_AI_ENDPOINT = "https://api.together.xyz/v1/chat/completions"
TOGETHER_AI_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# GitHub Release links (replace <user>/<repo>/v1.0.0 with your release info)
MODEL_URLS = {
    "clf": "https://github.com/KalpanaDevi-P/bcnnunet/releases/download/v1.0.0/brain_tumor_mobilenetv2_final.h5",
    "seg": "https://github.com/KalpanaDevi-P/bcnnunet/releases/download/v1.0.0/unet_brain_tumor_final.pth"
}
MODEL_PATHS = {
    "clf": "brain_tumor_mobilenetv2_final.h5",
    "seg": "unet_brain_tumor_final.pth"
}
CLASSES_PATH = "class_indices.json"


def download_model(url, path):
    """Download model file if not already present"""
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {path}")


def fetch_tumor_definition(tumor_type):
    """Fetches a short medical description of a tumor from TogetherAI"""
    headers = {
        "Authorization": f"Bearer {TOGETHER_AI_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"Give a concise medical definition of the brain tumor type: {tumor_type}. Dont mention anywhere as AI generated."
    data = {
        "model": TOGETHER_AI_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(TOGETHER_AI_ENDPOINT, headers=headers, json=data)
        response_json = response.json()
        if response.status_code == 200 and "choices" in response_json:
            return response_json["choices"][0]["message"]["content"]
        return f"No definition returned for {tumor_type}."
    except Exception as e:
        return f"Error fetching definition: {e}"


# ---------------------------
# 1. Load Classification Model
# ---------------------------
download_model(MODEL_URLS["clf"], MODEL_PATHS["clf"])
clf_model = load_model(MODEL_PATHS["clf"])

with open(CLASSES_PATH, "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}


# ---------------------------
# 2. Define UNet Segmentation
# ---------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.dec2 = CBR(384, 128)
        self.dec1 = CBR(192, 64)
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.up(e4)
        if d3.size()[2:] != e3.size()[2:]:
            d3 = F.interpolate(d3, size=e3.size()[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = F.interpolate(d3, size=e2.size()[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = F.interpolate(d2, size=e1.size()[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = torch.sigmoid(self.out(d1))
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
download_model(MODEL_URLS["seg"], MODEL_PATHS["seg"])
seg_model = UNet().to(device)
seg_model.load_state_dict(torch.load(MODEL_PATHS["seg"], map_location=device))
seg_model.eval()

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# ---------------------------
# 3. Routes (unchanged from your code)
# ---------------------------
# keep your upload(), classification(), segmentation(), download_report() exactly same
# ---------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
