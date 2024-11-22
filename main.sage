load('sequence/ec.sage')

from sage.all import *
import datetime
import logging
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from util.correlation import correlation_list_gpu
from util.balance import balance_rate, distribution_calc


def save_result(idx, res, final=False):
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath = 'results/' + ('final/' if final else '') + 'result_' + time_str + ('_final' if final else '') + '.csv'
    df = pd.DataFrame(res, index=idx)
    df.to_csv(filepath)
    

def evaluate_ec_corr_distribution(n, order_range=None):
    if order_range is None:
        order_range = [2^n, 2^n]
    ec = EC()
    seq, ord = ec.gen_of_order(n, order_range)
    auto_dist, non_auto_dist, dist = evaluate_corr_distribution(seq)
    return auto_dist, non_auto_dist, dist, ord
    

def evaluate_corr_distribution(seq):
    cors = correlation_list_gpu(seq)
    auto = torch.diagonal(cors)
    non_auto = cors - torch.diag_embed(auto)
    auto_dist = distribution_calc(auto)
    non_auto_dist = distribution_calc(torch.flatten(non_auto))
    dist = distribution_calc(torch.flatten(cors))
    non_auto_dist[0] -= len(seq)
    return auto_dist, non_auto_dist, dist


def evaluate_ec(n, order_range=None, file=None, useAllEC=False):
    if order_range is None:
        order_range = [2^n, 2^n]
    ec = EC()
    seq, ord = ec.gen_of_order(n, order_range, file, useAllEC)
    cors = correlation_list_gpu(seq)
    if file: print('Correlations:', file=file)
    if file: print('\n'.join(['\t'.join([str(int(x)) for x in e]) for e in cors]), file=file)
    cor = int(torch.max(cors))
    if file: print('Maximum correlation:', cor, file=file)
    return cor, ord


def evaluate_ec_balence(n, order_range=None):
    if order_range is None:
        order_range = [2^n, 2^n]
    ec = EC()
    seq, ord = ec.gen_of_order(n, order_range)
    r = balance_rate(seq)
    return r


def plt_save_distribution(dist, figure_path, label=''):
    x = range(len(dist))
    plt.bar(x, dist, label=label)
    plt.legend()
    plt.savefig(figure_path)
    plt.clf()


def save_distribution_ec(n, dir):
    rates = []
    for i in range(4, n + 1):
        qi = 2^i
        x = [float(e / qi) for e in range(qi + 1)]
        rate = evaluate_ec_balence(i)
        width = 1 / qi
        plt.xlabel('bias rate')
        plt.ylabel('sequence num')
        plt.bar(x, rate, label=str(i), width=width)
        plt.legend()
        plt.savefig(os.path.join(dir, '{}.png'.format(i)))
        plt.clf()
        rates.append(rate)
    pd.DataFrame(rates, index=range(4, n + 1)).to_csv(os.path.join(dir, 'result.csv'))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    for n in range(3, 5):
        t = 2^((n + 1) // 2)
        cor, ord = evaluate_ec(n)
        print('correlation:', cor)
        print('order of E:', ord)
