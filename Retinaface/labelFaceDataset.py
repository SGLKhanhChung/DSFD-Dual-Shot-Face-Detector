import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN

# Cấu hình
image_folder = r'C:\Users\Admin01\Desktop\DSFD-Dual-Shot-Face-Detector\Retinaface\data\VNFace\train\images'         # Thư mục ảnh đầu vào
output_file = r'C:\Users\Admin01\Desktop\DSFD-Dual-Shot-Face-Detector\Retinaface\data\VNFace\train\label.txt'  # File kết quả

mtcnn = MTCNN(keep_all=True)  # Detect multiple faces per image

with open(output_file, 'w') as f:
    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        path = os.path.join(image_folder, filename)
        img = Image.open(path).convert('RGB')

        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

        if boxes is None:
            continue

        f.write(f"# {filename}\n")
        for box, mark, prob in zip(boxes, landmarks, probs):
            x1, y1, x2, y2 = box
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)

            f.write(f"{x} {y} {w} {h} ")
            for (lx, ly) in mark:
                f.write(f"{lx:.3f} {ly:.3f} 0.0 ")
            f.write(f"{prob:.2f}\n")
