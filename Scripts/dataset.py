import logging
import os
import math
import torch
from torch.utils import data
import numpy as np


def get_loader(name, type):
    dataset_path = get_dataset_path(name, type)
    dataset = TrajectoryDataSet(path=dataset_path)
    loader = data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=data_collate
    )
    return dataset, loader

def data_collate(data):
    (obs_seq_list, pred_seq_list, obs_rel_seq_list, pred_rel_seq_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    '''
    format:  seq_len, batch, input_size
    default: pred/obs len, ped nums, 2(x-y-coord)
    '''
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_rel_traj = torch.cat(obs_rel_seq_list, dim=0).permute(2, 0, 1)
    pred_rel_traj = torch.cat(pred_rel_seq_list, dim=0).permute(2, 0, 1)

    seq_start_end = torch.LongTensor(seq_start_end)

    out = [
        obs_traj, pred_traj,
        obs_rel_traj, pred_rel_traj,
        seq_start_end
    ]

    return tuple(out)

def get_dataset_path(name, type):
    # path = '../datasets/'+name+'/' + type
    path = 'datasets/' + name + '/' + type
    if not os.path.isdir(path):
        logging.error(f'{path} is not exist...\nend program...')
        exit()
    return path


def read_file(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            d = line.strip().split()
            data.append([float(i) for i in d])
    # print(data)
    return np.asarray(data)


class TrajectoryDataSet(data.Dataset):
    def __init__(self,
                 path,
                 obs_len=8, pred_len=8):
        # TODO
        self.path = path
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.sequence_len = obs_len + pred_len
        self.skip = 1

        # region read data
        if not os.path.isdir(self.path):
            logging.error(f'{self.path} is not exist...\nend program...')
            exit()

        files = os.listdir(self.path)
        files = [os.path.join(self.path, f) for f in files]

        all_data = []
        num_peds_in_seq = []
        all_sequences = []
        all_rel_sequences = []
        for f in files:
            data = read_file(f)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            # check how many sequences in this data
            num_sequences = int(
                math.ceil((len(frames) - self.sequence_len + 1) / self.skip))
            '''
            print(f'Frame nums: {len(frames)}')
            print(num_sequences)
            '''
            for idx in range(0, num_sequences+1):
                # print(idx)
                curr_seq_data = np.concatenate(
                    frame_data[idx: idx+self.sequence_len], axis=0)
                # find all pedestrian in current sequence
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.sequence_len))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.sequence_len))

                num_peds = 0
                for ped_id in peds_in_curr_seq:
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                    # check whether complete
                    front_padding = frames.index(curr_ped_seq[0, 0]) - idx
                    end_padding = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if end_padding - front_padding != self.sequence_len:
                        # current pedestrian don't have enough data
                        '''
                        print(f'Not Complete : {end_padding - front_padding}\n'
                              f'{curr_ped_seq.shape}')
                        '''
                        continue

                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    # relative position
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    # new pedestrian to be data
                    ped_idx = num_peds
                    curr_seq[ped_idx, :, front_padding:end_padding] = curr_ped_seq
                    curr_seq_rel[ped_idx, :, front_padding:end_padding] = rel_curr_ped_seq
                    num_peds += 1

                if num_peds >= 1:
                    num_peds_in_seq.append(num_peds)
                    all_sequences.append(curr_seq[:num_peds])
                    all_rel_sequences.append(curr_seq_rel[:num_peds])

                '''
                print(f'Now frame idx: {idx}\n Now ped num in seq: {num_peds}\n'
                      f'Now seq: {curr_seq[:num_peds]}')
                '''
                # print(curr_seq_data)

        self.num_sequences = len(all_sequences)
        all_sequences = np.concatenate(all_sequences, axis=0)
        all_rel_sequences = np.concatenate(all_rel_sequences, axis=0)
        # endregion

        # convert numpy data to torch data
        self.obs_traj = torch.from_numpy(
            all_sequences[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            all_sequences[:, :, self.obs_len:]).type(torch.float)
        self.obs_rel_traj = torch.from_numpy(
            all_rel_sequences[:, :, :self.obs_len]).type(torch.float)
        self.pred_rel_traj = torch.from_numpy(
            all_rel_sequences[:, :, self.obs_len:]).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        print(self.num_sequences)
        print(len(self.seq_start_end))
        # print(self.obs_traj.shape)

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_rel_traj[start:end, :], self.pred_rel_traj[start:end, :],
        ]
        return out

    def __len__(self):
        return self.num_sequences

    def read_data(self):
        pass


# dataset_path = get_dataset_path('test1', 'test')
dataset_path = get_dataset_path('test1', 'test')
dataset = TrajectoryDataSet(path=dataset_path)
loader = data.DataLoader(
    dataset=dataset,
    batch_size=3,
    shuffle=True,
    collate_fn=data_collate
)

i = 0
for idx, d in enumerate(loader):
    # print(d)
    [obs_traj, pred_traj, obs_rel_traj, pred_rel_traj, seq_start_end] = d
    i += 1
    print(seq_start_end)

