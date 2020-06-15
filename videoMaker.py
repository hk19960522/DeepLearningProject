import cv2
import os
import numpy as np
import math


def read_file(path='datasets/hotel/test/biwi_hotel.txt'):
    data = []
    with open(path, 'r') as f:
        for line in f:
            d = line.strip().split()
            data.append([float(i) for i in d])
    # print(data)
    return np.asarray(data)


def find_bound(data):
    x_min = np.amin(data[:, 2])
    x_max = np.amax(data[:, 2])
    y_min = np.amin(data[:, 3])
    y_max = np.amax(data[:, 3])
    print(f'X: {x_min} ~ {x_max}\n'
          f'y: {y_min} ~ {y_max}')

    bound = np.amax(np.abs([x_min, x_max, y_min, y_max])) * 1.2
    print(f'{math.ceil(bound)}\n')
    return math.ceil(bound)


def make_image(size, bound, pos):
    img = np.zeros(size, np.uint8)
    img.fill(255)
    unit_size = (size[0] / (2.0*bound), size[1] / (2.0*bound))

    #print((int(pos[0] * unit_size[0]), int(pos[1] * unit_size[0])))
    cv2.circle(img, (int((pos[0]+bound) * unit_size[0]), int((pos[1]+bound) * unit_size[0])), 2, (255, 0, 0), -1)
    return img


def MakeVideo(img_list, size, fps=30):
    videoWriter = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)

    for img in img_list:
        videoWriter.write(img)
    videoWriter.release()
    pass

"""
img_list = []
for i in range(0, 30 * 30):
    img_list.append(MakeImage((500, 500, 3), i/(30.0*30.0)))
    print(f'img {i}')
MakeVideo(img_list, (500, 500), 30)
"""

data = read_file()
print(data.shape)
bound = find_bound(data)

frames = np.unique(data[:, 0]).tolist()
frame_data = []

#[frame, ID, x, y]
for frame in frames:
    frame_data.append(data[frame == data[:, 0], :])

print(f'{frames[0]}, {frames[-1]}, {len(frames)}')

cont_frame = []
cur_frame = frames[0]
start_idx = 0
diff = 0
for i in range(1, len(frames)):
    if i - start_idx == 1:
        diff = frames[i] - frames[start_idx]
        continue

    if frames[i] - frames[i-1] != diff:
        cont_frame.append([start_idx, i-1])
        start_idx = i

print(cont_frame)




"""
img_size = (500, 500, 3)

img_list = []
for i in range(0, 30 * 30):
    img_list.append(make_image(img_size, bound, [(i-450)/900.0*bound, (i-450)/900.0*bound]))
    print(f'img {i}')
MakeVideo(img_list, (500, 500), 30)

"""

