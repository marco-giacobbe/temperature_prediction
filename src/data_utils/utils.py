import pandas as pd
import numpy as np
import torch

def create_inout_seqs(input_data, in_w, out_w):
    inout_seq = []
    L = len(input_data)
    for i in range(L-in_w-out_w+1):
        train_seq = np.append(input_data[i:i+in_w], out_w * [0])
        train_label = input_data[i:i+in_w+out_w]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(np.array(inout_seq))

def get_data(datapath, device, in_w, out_w):
    train_data = pd.read_csv(datapath.format("train"), header=0, index_col=0, skiprows=4)["temp"].to_numpy()
    test_data = pd.read_csv(datapath.format("eval"), header=0, index_col=0, skiprows=4)["temp"].to_numpy()
    train_sequence = create_inout_seqs(train_data, in_w, out_w).to(device)
    test_sequence = create_inout_seqs(test_data, in_w, out_w).to(device)
    return train_sequence, test_sequence

def get_batch(source, i, bsz, out_w):
    seq_len = min(bsz, len(source) - 1 - i)
    data = source[i:i+seq_len]
    data = data.permute(1, 0, 2)
    return data[0][:, :-out_w], data[1][:, -out_w:]