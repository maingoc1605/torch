import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import imshow, imshow2, imshow3, mixup
from ultralytics import YOLO
import cv2
import numpy as np
model = YOLO("yolov8m-seg.pt")
from datetime import datetime
'''
dst = cv2.imread(r"C:\maintn\Pictures\image_original.jpg")
src = cv2.imread(r"C:\maintn\Pictures\car.jpg")

src_mask = cv2.imread(r"C:maintn\Pictures\mask.png", cv2.IMREAD_GRAYSCALE)
src_mask= cv2.bitwise_not(src_mask)
predict = model.predict("\maintn\Pictures\car.jpg" , save = False, save_txt = True)
cv2.imshow("map",((predict[0].masks.masks[0].numpy() * 255).astype("uint8")))
mask = (predict[0].masks.masks[0].numpy() * 255).astype("uint8")
mask= cv2.bitwise_not(mask)
print(src_mask)
w,h,_ = dst.shape
print(dst.shape)
dim = (w,h)
dst_resized = cv2.resize(dst, dim, interpolation = cv2.INTER_AREA)
src_resized = cv2.resize(src, dim, interpolation = cv2.INTER_AREA)
src_mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
img = mixup(dst_resized,src_resized)
imshow(img)
imshow3(dst, src, src_mask)
alpha = cv2.cvtColor(src_mask.copy(), cv2.COLOR_GRAY2RGB)
alpha = alpha.astype(np.float32) / 255.0
#imshow3(dst, src, src_mask)
print(alpha.shape)
print(dst_resized.shape)
#output_noclone = src_resized * alpha #+ dst_resized * (1 - alpha)
output_noclone = src_resized * (1-alpha) + dst_resized * (alpha)
output_noclone = output_noclone.astype(np.uint8)

imshow(output_noclone, 0.5)
'''
def mix(dst_path, src_path, model):
    dst = cv2.imread(dst_path)
    src = cv2.imread(src_path)
    predict = model.predict(src_path, save=False, save_txt=True)
    mask = (predict[0].masks.masks[0].numpy() * 255).astype("uint8")
    mask = cv2.bitwise_not(mask)
    w, h, _ = dst.shape
    dim = (w, h)
    dst_resized = cv2.resize(dst, dim, interpolation=cv2.INTER_AREA)
    src_resized = cv2.resize(src, dim, interpolation=cv2.INTER_AREA)
    src_mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)
    imshow3(dst, src, src_mask)
    alpha = cv2.cvtColor(src_mask.copy(), cv2.COLOR_GRAY2RGB)
    alpha = alpha.astype(np.float32) / 255.0
    output_noclone = src_resized * (1 - alpha) + dst_resized * (alpha)
    output_noclone = output_noclone.astype(np.uint8)
    return output_noclone
dst_path = r"C:\Users\maintn\Pictures\image_original.jpg "
src_path = r"C:\Users\maintn\Pictures\car.jpg"
mix(dst_path, src_path, model)


