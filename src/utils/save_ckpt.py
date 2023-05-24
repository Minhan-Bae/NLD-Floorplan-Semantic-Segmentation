import torch
import logging

def save_checkpoint(state, filename = ''):
    torch.save(state, filename)
    logging.info(f'\nSave ckpt : {filename}')