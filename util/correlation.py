import time
import logging
import torch

def correlation_loop(s_sequence):
    time_start = time.time()
    cors = 0
    len_seq = len(s_sequence)
    if len_seq == 0:
        return 0
    len_lst = len(s_sequence[0])
    for i in range(len_seq):
        for j in range(len_seq):
            start = 0
            if i == j:
                start = 1
            for t in range(start, len_lst):
                s = 0
                for k in range(len_lst):
                    s += s_sequence[i][k] * s_sequence[j][(k + t) % len_lst]
                s = abs(s)
                if cors < s:
                    cors = s
    time_end = time.time()
    logging.debug('Time cost: {}'.format(time_end-time_start))
    return cors

def correlation_cpu(s_sequence):
    time_start = time.time()
    cors = 0
    len_seq = len(s_sequence)
    if len_seq == 0:
        return 0
    len_lst = len(s_sequence[0])
    A = torch.tensor(s_sequence, dtype=torch.float)
    arange1 = torch.arange(len_lst).view((len_lst, 1)).repeat((1, len_lst))
    indices = torch.arange(len_lst)
    arange2 = (arange1 - indices) % len_lst
    for i in range(len_seq):
        T = torch.tensor(s_sequence[i], dtype=torch.float)
        B = T.repeat(len_lst, 1)
        B.transpose_(int(0), int(1))
        B = torch.gather(B, int(0), arange2)
        C = torch.matmul(A, B)
        C[i, 0] = float(0)
        t = int(max([C.max(), -C.min()]))
        if cors < t:
            cors = t
    time_end = time.time()
    logging.debug('Correlation compute (CPU) time cost: {}'.format(time_end-time_start))
    return cors

def correlation_list_gpu(s_sequence):
    time_start = time.time()
    cors = []
    len_seq = len(s_sequence)
    if len_seq == 0:
        return 0
    len_lst = len(s_sequence[0])
    A = torch.tensor(s_sequence, dtype=torch.float).cuda()
    arange1 = torch.arange(len_lst).cuda().view((len_lst, 1)).repeat((1, len_lst))
    indices = torch.arange(len_lst).cuda()
    arange2 = (arange1 - indices) % len_lst
    for i in range(len_seq):
        T = torch.tensor(s_sequence[i], dtype=torch.float).cuda()
        B = T.repeat(len_lst, 1)
        B.transpose_(int(0), int(1))
        B = torch.gather(B, int(0), arange2)
        C = torch.matmul(A, B)
        C[i, 0] = float(0)
        C = torch.abs(C)
        cor = torch.max(C, 1).values
        cors.append(cor)
    ret = torch.stack(cors, 1).cpu()
    time_end = time.time()
    logging.debug('Correlation compute (GPU) time cost: {}'.format(time_end-time_start))
    return ret

def correlation_gpu(s_sequence):
    time_start = time.time()
    cors = 0
    len_seq = len(s_sequence)
    if len_seq == 0:
        return 0
    len_lst = len(s_sequence[0])
    A = torch.tensor(s_sequence, dtype=torch.float).cuda()
    arange1 = torch.arange(len_lst).cuda().view((len_lst, 1)).repeat((1, len_lst))
    indices = torch.arange(len_lst).cuda()
    arange2 = (arange1 - indices) % len_lst
    for i in range(len_seq):
        T = torch.tensor(s_sequence[i], dtype=torch.float).cuda()
        B = T.repeat(len_lst, 1)
        B.transpose_(int(0), int(1))
        B = torch.gather(B, int(0), arange2)
        C = torch.matmul(A, B)
        C[i, 0] = float(0)
        t = int(max([C.max(), -C.min()]))
        if cors < t:
            cors = t
    time_end = time.time()
    logging.debug('Correlation compute (GPU) time cost: {}'.format(time_end-time_start))
    return cors
