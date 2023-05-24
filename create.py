import cv2
import numpy as np


def convert_roi_poly_to_rec(img, points_array):
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    points = np.array([points_array])
    # method 1 smooth region
    # cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # method 2 not so smooth region
    cv2.fillPoly(mask, points, (255))
    res = cv2.bitwise_and(img,img,mask = mask)
    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    # ## crate the white background of the same size of original image
    # wbg = np.ones_like(img, np.uint8)*255
    # cv2.bitwise_not(wbg,wbg, mask=mask)
    # # overlap the resulted cropped image on the white background
    # dst = wbg+res

    return cropped, rect # r

path_stream = "rtsp://admin:Ab123456@cam8ocd.cameraddns.net:556/Streaming/Channels/1"
cap = cv2.VideoCapture(path_stream)
image = None
while True:
    ok, image = cap.read()
    if ok:
        break
# path_image = "/home/mypc/taidv/project/camera_ai_traffic/datasets/Test/2023_01_08/cap_VID_20230108_120049_00:00:08_01.jpg"
# image = cv2.imread(path_image)
print(image.shape)
cv2.imwrite("image_original.jpg", image)
a = [[1618, 27], [575, 2150], [3815, 2141], [3826, 1710], [2869, 246], [2993, 6]]
cropped, rect  = convert_roi_poly_to_rec(image, a)
print(cropped.shape)
cv2.imwrite("image_cropped_roi_polygon.jpg", cropped)
print(rect)
