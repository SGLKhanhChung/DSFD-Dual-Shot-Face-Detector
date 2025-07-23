from facenet_pytorch import MTCNN
from PIL import Image
import os

image_folder = r'C:\Users\Admin01\Desktop\DSFD-Dual-Shot-Face-Detector\Retinaface\data\VNFace\train\images'
output_txt = r'C:\Users\Admin01\Desktop\DSFD-Dual-Shot-Face-Detector\Retinaface\data\VNFace\train\label.txt'

mtcnn = MTCNN(keep_all=True)  # Detect multiple faces per image

with open(output_txt, 'w') as f:
    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        path = os.path.join(image_folder, filename)
        img = Image.open(path).convert('RGB')

        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

        if boxes is None:
            continue

        f.write(f"# {path}\n")
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
