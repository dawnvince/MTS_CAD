from ast import parse
from multiprocessing import cpu_count
import os
import numpy as np
import random
import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary, Callback
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers import TensorBoardLogger

from model import MMoE


seed = 2023
if not seed:
    seed = random.randint(1, 10000)
print("seed is %d" %seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

logger = TensorBoardLogger(name="logs", save_dir="./")
print("gpu is available: ", torch.cuda.is_available())
if torch.cuda.is_available(): dev = "gpu"
else: dev = "cpu"

def train_test_model(hparams, seed=None):
    print("Loading Model...")
    model = MMoE(hparams, seed)
    print("Model Built...")

    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, verbose=True, mode='min'
    )
    
    cpkt_callback = ModelCheckpoint(
        monitor='val_loss', save_top_k=1, mode='min'
    )
    
    callback = [cpkt_callback, early_stop]
    trainer = Trainer(max_epochs=int(10) , callbacks=callback, logger=logger,\
        devices=1, accelerator=dev
    )
    
    trainer.fit(model, train_dataloaders=model.mydataloader(train='train'), val_dataloaders=model.mydataloader(train="train"))
    print("=========Train over============")
    
    test_result = trainer.test(model, dataloaders=model.mydataloader(train="test"))


def arg_parse_():
    parser = ArgumentParser()
    
    # dataset hparams
    
    parser.add_argument('--dataset' ,type=str, default="SMD", choices=["SMD", "SWaT", "WADI"])
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--window', default=16, type=int)
    
    # n_multiv means the dimension of metrics
    parser.add_argument('--n_multiv', type=int)
    parser.add_argument('--horize', default=1, type=int)
    parser.add_argument('--batch_size', type=int, default=128)
    
    # model hparams
    parser.add_argument('--num_experts', default=5, type=int)
    parser.add_argument('--n_kernel', type=int, default=16)
    parser.add_argument('--experts_out', type=int, default=128)
    parser.add_argument('--experts_hidden', type=int, default=256)
    parser.add_argument('--towers_hidden', type=int, default=32)
    parser.add_argument('--criterion', default="l2", type=str, choices=["l1", "l2"])
    parser.add_argument('--exp_dropout', type=float, default=0.2)
    parser.add_argument('--tow_dropout', type=float, default=0.1)
    parser.add_argument('--conv_dropout', type=float, default=0.1)
    parser.add_argument('--sg_ratio', type=float, default=0.7)
    parser.add_argument('--lr', type=float, default=0.001)
 
    
    return parser
    
    
if __name__ == "__main__":
    
    parser = arg_parse_()
    hparams = parser.parse_args()
    print(hparams)
    
    train_test_model(hparams, seed)
    
