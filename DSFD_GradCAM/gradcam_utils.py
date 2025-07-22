import torch, cv2, numpy as np
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from face_detection.dsfd.face_ssd import SSD

# Khởi tạo GradCAM (chỉ cần gọi 1 lần)
def init_gradcam(model_path, config):
    ssd = SSD(config)
    sd = torch.load(model_path, map_location="cpu")
    ssd.load_state_dict(sd, strict=False)
    ssd.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Dùng feature extractor
    class F(torch.nn.Module):
        def __init__(self, s):
            super().__init__()
            for i in range(1, 5):
                setattr(self, f"layer{i}", getattr(s, f"layer{i}"))
        def forward(self, x):
            for i in range(1,5): x = getattr(self, f"layer{i}")(x)
            return x
    feat = F(ssd).to(torch.float32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    cam = EigenCAM(model=feat, target_layers=[feat.layer4[-1]])
    return cam

def overlay_gradcam_on_bbox(image, bbox, cam, alpha=0.4):
    x0,y0,x1,y1 = bbox
    crop = image[y0:y1, x0:x1]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) /255.0
    # Grad‑CAM
    inp = torch.tensor(((rgb - np.array([0.485,0.456,0.406]))/[0.229,0.224,0.225]).transpose(2,0,1)[None]).float().to(cam.device)
    gray = cam(input_tensor=inp)[0]
    gray = cv2.resize(gray, (x1-x0, y1-y0))
    heat = cv2.applyColorMap((gray*255).astype(np.uint8), cv2.COLORMAP_JET)[...,::-1]/255.0
    overlay = np.clip((1-alpha)*rgb + alpha*heat,0,1)
    image[y0:y1, x0:x1] = (overlay*255).astype(np.uint8)[...,::-1]
    return image
