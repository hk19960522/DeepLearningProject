import cv2
import os
import numpy as np


def MakeImage(size, ratio):
    img = np.zeros(size, np.uint8)

    for i in range(0, size[0]):
        for j in range(0, size[1]):
            for k in range(0, size[2]):
                img[i, j, k] = ratio * 255
    return img


def MakeVideo(img_list, size, fps=30):
    videoWriter = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)

    for img in img_list:
        videoWriter.write(img)
    videoWriter.release()
    pass

img_list = []
for i in range(0, 30 * 30):
    img_list.append(MakeImage((500, 500, 3), i/(30.0*30.0)))
    print(f'img {i}')
MakeVideo(img_list, (500, 500), 30)