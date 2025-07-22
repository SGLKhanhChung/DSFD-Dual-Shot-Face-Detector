import glob
import os
import cv2
import time
import face_detection
import gradCAMdetect as grad


def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        im=grad.run_gradcam_on_bbox_and_overlay(im,(x0, y0, x1, y1))
        #with open("bbox_output.txt", "w") as f:
            #f.write(f"{x0} {y0} {x1} {y1}\n")
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)


if __name__ == "__main__":
    impaths = "images"
    impaths = glob.glob(os.path.join(impaths, "*.jpeg"))
    detector = face_detection.build_detector(
        "DSFDDetector",
        max_resolution=1080
    )
    for impath in impaths:
        if impath.endswith("out.jpeg"): continue
        im = cv2.imread(impath)
        print("Processing:", impath)
        t = time.time()
        dets = detector.detect(
            im[:, :, ::-1]
        )[:, :4]
        print(f"Detection time: {time.time()- t:.3f}")
        draw_faces(im, dets)
        imname = os.path.basename(impath).split(".")[0]
        output_path = os.path.join(
            os.path.dirname(impath),
            f"{imname}_out.jpeg"
        )

        cv2.imwrite(output_path, im)
        