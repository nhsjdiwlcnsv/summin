import cv2
import numpy as np

from plots.constants import FONT, FONT_SCALE, FONT_THICKNESS, BORDER_THICKNESS


def draw_bboxes(img, bboxes, scores, color):
    bboxes = bboxes.astype(int)

    for i, bbox in enumerate(bboxes):
        label: str = f"Formula: {scores[i]:.3f}"
        label_y: int = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
        (label_w, label_h), label_b = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

        # Bounding box
        img = cv2.rectangle(
            img=img,
            pt1=(bbox[0], bbox[1]),
            pt2=(bbox[2], bbox[3]),
            color=color, thickness=BORDER_THICKNESS
        )

        # Bounding box background
        sub_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]  # subimage to add background to
        color_rect = np.full_like(sub_img, color)  # color rectangle size of subimage
        background = cv2.addWeighted(sub_img, 0.95, color_rect, 1, 1)
        img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = background

        # Text background
        img = cv2.rectangle(
            img=img,
            pt1=(bbox[0] - BORDER_THICKNESS + 1, bbox[1]),
            pt2=(bbox[0] + label_w, label_y - label_h),
            color=color, thickness=-1
        )

        # Text
        img = cv2.putText(
            img=img, text=label,
            org=(bbox[0], label_y + BORDER_THICKNESS + 1),
            fontFace=FONT, fontScale=FONT_SCALE,
            color=(255, 255, 255), thickness=FONT_THICKNESS
        )

    return img
