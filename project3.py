import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Step 1:
import cv2
import numpy as np
from ultralytics import YOLO  # moved up so everything imports first

# 1. Load image
img = cv2.imread("motherboard_image.JPEG")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Blur to reduce small details
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# 3. Canny edges
edges = cv2.Canny(blur, 30, 120)

# 4. CLOSE GAPS in edges so the board becomes one big contour
kernel_edges = np.ones((21, 21), np.uint8)
edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_edges)

# 5. Find only external contours on the *closed* edge image
contours, _ = cv2.findContours(
    edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Get the largest contour = whole PCB
largest = max(contours, key=cv2.contourArea)

# 6. Create a filled mask from that contour (exact PCB outline)
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.drawContours(mask, [largest], -1, 255, thickness=-1)

# 7. Extract PCB using cleaned mask
extracted = cv2.bitwise_and(img, img, mask=mask)

cv2.imwrite("pcb_extracted.png", extracted)


# ---------- Step 2 + Step 3 wrapped in main() for Windows ----------

def main():
    # Step 2:
    model = YOLO("yolo11n.pt")  # nano model

    data = "data/data.yaml"

    model.train(
        data=data,          # YAML with train/val paths and class names
        epochs=150,         # must be < 200 as per instructions
        batch=4,            # tune based on GPU memory
        imgsz=960,         # recommended >= 900
        name="pcb_prediction",  # run/model name
        workers=0,          # IMPORTANT on Windows to avoid multiprocessing spawn error
    )

    # Step 3:
    model.predict("data/evaluation/ardmega.jpg", save=True, workers=0)
    model.predict("data/evaluation/arduno.jpg", save=True, workers=0)
    model.predict("data/evaluation/rasppi.jpg", save=True, workers=0)


if __name__ == "__main__":
    main()
