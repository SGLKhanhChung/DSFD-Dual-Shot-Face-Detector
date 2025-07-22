import glob
import os
import cv2
import time
import face_detection
from gradcam_utils import init_gradcam, overlay_gradcam_on_bbox


def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)
        


if __name__ == "__main__":
    impaths = "images"
    impaths = glob.glob(os.path.join(impaths, "*.jpg"))
    detector = face_detection.build_detector(
        "DSFDDetector",
        max_resolution=1080
    )
    resnet152_model_config = {
    'mbox': [3, 6, 6, 6, 6, 6],  # hoặc cấu hình thực tế bạn đã dùng
    'variance': [0.1, 0.2]
    }
    cam = init_gradcam("RetinaFace_mobilenet025.pth", resnet152_model_config)
    
    for impath in impaths:
        if impath.endswith("out.jpg"): continue
        im = cv2.imread(impath)
        print("Processing:", impath)
        t = time.time()
        dets = detector.detect(
            im[:, :, ::-1]
        )[:, :4]
        print(f"Detection time: {time.time()- t:.3f}")
        for bbox in dets:
            x0, y0, x1, y1 = [int(_) for _ in bbox]
            im = overlay_gradcam_on_bbox(im, (int(x0),int(y0),int(x1),int(y1)), cam)
        draw_faces(im, dets)
        imname = os.path.basename(impath).split(".")[0]
        output_path = os.path.join(
            os.path.dirname(impath),
            f"{imname}_out.jpg"
        )

        cv2.imwrite(output_path, im)
        