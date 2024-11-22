import os
import psutil
import logging

def log_mem(message=None):
    if message != None:
        message = '({})'.format(message)
    else:
        message = ''
    logging.debug('Memory usage {}: {:.4f} GB'.format(message, psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))