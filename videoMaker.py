import cv2
import os
import numpy as np
import math


def read_file(path='datasets/hotel/test/biwi_hotel.txt'):
    #path = 'datasets/univ/test/students003.txt'
    data = []
    with open(path, 'r') as f:
        for line in f:
            d = line.strip().split()
            data.append([float(i) for i in d])
    # print(data)
    return np.asarray(data)


def find_bound(data):
    x_min = np.amin(data[:, 0])
    x_max = np.amax(data[:, 0])
    y_min = np.amin(data[:, 1])
    y_max = np.amax(data[:, 1])
    print(f'X: {x_min} ~ {x_max}\n'
          f'y: {y_min} ~ {y_max}')

    bound = np.amax(np.abs([x_min, x_max, y_min, y_max]))
    print(f'{math.ceil(bound)}\n')
    return math.ceil(bound)


def make_image(size, bound, pos_data):
    img = np.zeros(size, np.uint8)
    img.fill(255)
    unit_size = (size[0] / (2.0*bound), size[1] / (2.0*bound))

    #print((int(pos[0] * unit_size[0]), int(pos[1] * unit_size[0])))
    '''
    for pos in pos_data:
        cv2.circle(img,
                   (int((pos[0] + bound) * unit_size[0]), int((pos[1] + bound) * unit_size[0])),
                   4, (255, 0, 0), -1)    
    '''
    bound = bound * 0.5
    for ped_data in pos_data:
        for idx, obs_pos in enumerate(ped_data[0]):
            if idx == 0:
                continue
            cv2.line(img,
                     (int((obs_pos[0] + bound) * unit_size[0]),
                      int((obs_pos[1] + bound) * unit_size[0])),
                     (int((ped_data[0][idx-1][0] + bound) * unit_size[0]),
                      int((ped_data[0][idx-1][1] + bound) * unit_size[0])),
                     (255, 0, 0), 4)
        for idx, pred_pos in enumerate(ped_data[1]):
            if idx == 0:
                continue
            cv2.line(img,
                     (int((pred_pos[0] + bound) * unit_size[0]),
                      int((pred_pos[1] + bound) * unit_size[0])),
                     (int((ped_data[1][idx-1][0] + bound) * unit_size[0]),
                      int((ped_data[1][idx-1][1] + bound) * unit_size[0])),
                     (0, 255, 0), 4)

    # cv2.circle(img, (int((pos[0]+bound) * unit_size[0]), int((pos[1]+bound) * unit_size[0])), 2, (255, 0, 0), -1)
    return img


def MakeVideo(img_list, size, fps=30, file_name = 'output'):
    videoWriter = cv2.VideoWriter(file_name + '.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)

    for img in img_list:
        videoWriter.write(img)
    videoWriter.release()
    pass

def GetFrameData(frame_data, cont_frame, bound, size, repeat = 3):
    # frmae_data : one dimension

    img_list = []
    start_idx = 0

    obs_len = 8
    pred_len = 8

    pre_start_idx = 0
    for (idx, frame) in enumerate(frame_data):
        start_idx = 0

        while 0 <= start_idx < len(cont_frame) and not (cont_frame[start_idx][0] <= frame[0, 0] <= cont_frame[start_idx][1]):
            start_idx += 1
        if start_idx < 0 or start_idx >= len(cont_frame):
            continue

        if start_idx != pre_start_idx:
            MakeVideo(img_list, size[0:2], 30, 'output' + str(pre_start_idx))
            img_list = []
            pre_start_idx = start_idx

        data = frame[:, 1:]
        frame_id = frame[0, 0]

        peds_data = []
        for ped_data in data:
            ped_id = ped_data[0]
            obs_data = []
            pred_data = []
            for i in range(-obs_len, 1):
                f_id = idx + i
                if f_id < 0:
                    continue

                if cont_frame[start_idx][0] <= frame_data[f_id][0, 0] <= cont_frame[start_idx][1]:
                    # in same data range
                    pos = frame_data[f_id][frame_data[f_id][:, 1] == ped_id, 2:]
                    if len(pos) != 0:
                        obs_data.append(pos[0])

            for i in range(0, pred_len+1):
                f_id = idx + i
                if f_id >= len(frame_data):
                    continue

                if cont_frame[start_idx][0] <= frame_data[f_id][0, 0] <= cont_frame[start_idx][1]:
                    # in same data range
                    pos = frame_data[f_id][frame_data[f_id][:, 1] == ped_id, 2:]
                    if len(pos) != 0:
                        pred_data.append(pos[0])
                    if idx == 0:
                        print(f'Now pos:{ped_data[1:]}, After {i}\'s pos: {pos}')
            '''
            if idx == 0:
                print(f'{pred_data}\n{obs_data}')
            '''
            p_data = [obs_data, pred_data]
            peds_data.append(p_data)

        for i in range(repeat):
            img_list.append(make_image(size, bound, peds_data))

    # MakeVideo(img_list, size[0:2], 30)
    MakeVideo(img_list, size[0:2], 30, 'output' + str(pre_start_idx))
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
size = (800, 800, 3)

frames = np.unique(data[:, 0]).tolist()
frame_data = []

#[frame, ID, x, y]
for frame in frames:
    frame_data.append(data[frame == data[:, 0], :])


print(f'type: {(data[0 == data[:, 0], :]).shape}, value: {data[0 == data[:, 0], :]}')
print(f'{0}, {frames[-1]}, {len(frames)}')

cont_frame = []
cur_frame = frames[0]
start_idx = 0
diff = 0
for i in range(1, len(frames)):
    if i - start_idx == 1:
        diff = frames[i] - frames[start_idx]
        continue

    if frames[i] - frames[i-1] != diff:
        cont_frame.append([frames[start_idx], frames[i-1]])
        start_idx = i

if len(cont_frame) == 0:
    cont_frame.append([frames[0], frames[-1]])
print(cont_frame)

input('next...')
m = np.array(frame_data)
print(m.shape)
print(f'Frame Data index 0 : {m[0:5]}')

# get all data position
all_pos = np.zeros([])

for idx, data in enumerate(frame_data):
    if idx == 0:
        all_pos = data[:, 2:4]
    else:
        all_pos = np.concatenate((all_pos, data[:, 2:4]), axis=0)

bound = find_bound(all_pos)
print(f'Bound : {bound}')

GetFrameData(frame_data, cont_frame, bound, size)



"""
img_size = (500, 500, 3)

img_list = []
for i in range(0, 30 * 30):
    img_list.append(make_image(img_size, bound, [(i-450)/900.0*bound, (i-450)/900.0*bound]))
    print(f'img {i}')
MakeVideo(img_list, (500, 500), 30)

"""

