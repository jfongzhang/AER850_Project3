import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Step 1:
import cv2
import numpy as np
from ultralytics import YOLO  

img = cv2.imread("motherboard_image.JPEG")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (7, 7), 0)

edges = cv2.Canny(blur, 30, 120)
cv2.imwrite("pcb_edges.png", edges)

kernel = np.ones((21, 21), np.uint8)
edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("pcb_edges_closed.png", edges_closed)

contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)

mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
cv2.imwrite("pcb_mask.png", mask)

extracted = cv2.bitwise_and(img, img, mask=mask)
cv2.imwrite("pcb_extracted.png", extracted)


def main():
    # Step 2:
    model = YOLO("yolo11n.pt") 

    data = "data/data.yaml"

    model.train(
        data=data,          
        epochs=150,       
        batch=4,           
        imgsz=960,        
        name="pcb_prediction",  
        workers=0,  
    )

    # Step 3:
    model.predict("data/evaluation/ardmega.jpg", save=True, workers=0)
    model.predict("data/evaluation/arduno.jpg", save=True, workers=0)
    model.predict("data/evaluation/rasppi.jpg", save=True, workers=0)


if __name__ == "__main__":
    main()
