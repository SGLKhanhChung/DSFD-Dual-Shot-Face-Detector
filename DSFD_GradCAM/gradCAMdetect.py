import torch
import cv2
import numpy as np
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from face_detection.dsfd.face_ssd import SSD  # Đảm bảo đúng path

# === Load mô hình ===
model_path = "RetinaFace_mobilenet025.pth"
resnet152_model_config = {'mbox': [3, 6, 6, 6, 6, 6], 'variance': [0.1, 0.2]}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ssd_model = SSD(resnet152_model_config)
state_dict = torch.load(model_path, map_location="cpu")
ssd_model.load_state_dict(state_dict, strict=False)
ssd_model.eval().to(device)

# === Feature extractor để dùng với Grad-CAM ===
class SSDFeatureOnly(torch.nn.Module):
    def __init__(self, ssd_model):
        super().__init__()
        self.layer1 = ssd_model.layer1
        self.layer2 = ssd_model.layer2
        self.layer3 = ssd_model.layer3
        self.layer4 = ssd_model.layer4

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

feature_model = SSDFeatureOnly(ssd_model).to(torch.float32).to(device)
target_layer = feature_model.layer4[-1]
cam = EigenCAM(model=feature_model, target_layers=[target_layer])

# === Tiền xử lý ảnh ===
def preprocess_image(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return tensor.to(device)

# === Hàm chính: chạy Grad-CAM lên bbox ===
def run_gradcam_on_bbox_and_overlay(image, bbox):
    x0, y0, x1, y1 = bbox

    # Cắt vùng mặt từ ảnh gốc
    cropped = image[y0:y1, x0:x1]
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        raise ValueError("Invalid bounding box.")

    rgb_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else cropped
    input_tensor = preprocess_image(rgb_crop)

    grayscale_cam = cam(input_tensor=input_tensor)[0]
    #grayscale_cam_resized = cv2.resize(grayscale_cam, (x1 - x0, y1 - y0))


    # Biến ảnh về [0, 1] float32 để overlay
    rgb_crop_norm = rgb_crop.astype(np.float32) / 255.0
    cam_overlay = show_cam_on_image(rgb_crop_norm, grayscale_cam, use_rgb=True)

    # Gắn overlay trở lại ảnh gốc
    image_out = image.copy()
    image_out[y0:y1, x0:x1] = cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR)

    return image_out