import argparse
import glob
import os

from dotenv import load_dotenv
import cv2
import numpy as np
import torch
from pdf2image import convert_from_path
from ultralytics import YOLO
from tqdm import tqdm

from plots import draw_bboxes
from plots.constants import MIN_COLOR_VAL, MAX_COLOR_VAL, DEFAULT_COLOR


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(prog="PDF Formula Detection")

    parser.add_argument("folder")
    parser.add_argument("filetype")

    args = parser.parse_args()
    folder, filetype = args.folder, args.filetype

    model = YOLO(os.environ["MODEL_PATH"])
    predictions = []

    if filetype == "pdf":
        files = glob.glob(os.path.join(folder, "**", f"*.{filetype}"), recursive=True)
        images = []

        for file in tqdm(files, desc="Converting PDFs to JPGs"):
            images += convert_from_path(file)

        with torch.no_grad():
            predictions = model.predict(images, device="mps", stream=True)

    if filetype == "mp4":
        video = glob.glob(os.path.join(folder, f"*.{filetype}"))[0]

        with torch.no_grad():
            predictions = model.predict(video, device="mps", stream=True)

    for prediction in predictions:
        image = prediction.orig_img
        scores = prediction.boxes.conf.cpu().numpy()
        bboxes = prediction.boxes
        window_name = f"Image: {len(bboxes)} formulas"

        if filetype != "mp4":
            color = np.random.randint(MIN_COLOR_VAL, MAX_COLOR_VAL, (3,)).tolist()
        else:
            color = DEFAULT_COLOR

        image = draw_bboxes(
            img=image,
            scores=scores,
            bboxes=bboxes.xyxy.cpu().numpy(),
            color=color
        )

        cv2.imshow(window_name, image)
        cv2.waitKey(int(filetype == "mp4"))
        cv2.destroyAllWindows()
