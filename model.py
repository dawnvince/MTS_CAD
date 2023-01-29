import os
from random import shuffle
from turtle import forward
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset import MTSDataset
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


import warnings
warnings.filterwarnings("ignore")

class Expert(nn.Module):
    def __init__(self, n_kernel, window, n_multiv, hidden_size, output_size, drop_out):
        super(Expert, self).__init__()
        self.conv = nn.Conv2d(1, n_kernel, (window, 1))
        self.dropout = nn.Dropout(drop_out)
        self.fc1 = nn.Linear(n_kernel * n_multiv, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x):
        x = x.unsqueeze(dim=1).contiguous()
        x = F.relu(self.conv(x))
        x = self.dropout(x)
        
        out = torch.flatten(x, start_dim=1).contiguous()
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    
class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, drop_out):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(drop_out)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class MMoE(pl.LightningModule):
    def __init__(self, hparams, seed=None):
        super(MMoE, self).__init__()
        self.hp = hparams
        self.seed = seed
        self.n_multiv = hparams.n_multiv
        self.n_kernel = hparams.n_kernel
        self.window = hparams.window
        self.num_experts = hparams.num_experts
        self.experts_out = hparams.experts_out
        self.experts_hidden = hparams.experts_hidden
        self.towers_hidden = hparams.towers_hidden
        
        # task num = n_multiv
        self.tasks = hparams.n_multiv
        self.criterion = hparams.criterion
        self.exp_dropout = hparams.exp_dropout
        self.tow_dropout = hparams.tow_dropout
        self.conv_dropout = hparams.conv_dropout
        self.lr = hparams.lr

        self.softmax = nn.Softmax(dim=1)
        
        self.experts = nn.ModuleList([Expert(self.n_kernel, self.window, self.n_multiv, self.experts_hidden, self.experts_out, self.exp_dropout) \
            for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(self.window, self.num_experts), requires_grad=True) \
            for i in range(self.tasks)])
        self.share_gate = nn.Parameter(torch.randn(self.window, self.num_experts), requires_grad=True)
        self.towers = nn.ModuleList([Tower(self.experts_out, 1, self.towers_hidden, self.tow_dropout) \
            for i in range(self.tasks)])
        
    
    def forward(self, x):
        experts_out = [e(x) for e in self.experts]
        experts_out_tensor = torch.stack(experts_out)
        
        gates_out = [self.softmax((x[:,:,i] @ self.w_gates[i]) * (1 - self.hp.sg_ratio) + (x[:,:,i] @ self.share_gate) * self.hp.sg_ratio) for i in range(self.tasks)]
        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_out_tensor for g in gates_out]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]
        
        tower_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        tower_output = torch.stack(tower_output, dim=0).permute(1,2,0)
        
        final_output = tower_output
        return final_output
    
    def loss(self, labels, predictions):
        if self.criterion == "l1":
            loss = F.l1_loss(predictions, labels)
        elif self.criterion == "l2":
            loss = F.mse_loss(predictions, labels)
        return loss
    
    def training_step(self, data_batch, batch_i):
        x, y = data_batch
        
        y_hat_ = self.forward(x)   
        
        loss_val = self.loss(y, y_hat_)
        self.log("val_loss", loss_val)
        output = OrderedDict({
            'loss': loss_val,
            'y' :y,
            'y_hat':y_hat_
        })
        return output
        
    def validation_step(self, data_batch, batch_i):
        x, y = data_batch
        
        y_hat_ = self.forward(x)

        loss_val = self.loss(y, y_hat_)
        
        self.log("val_loss", loss_val, on_step=False, on_epoch=True)
        output = OrderedDict({
            'val_loss': loss_val,
            'y' :y,
            'y_hat':y_hat_
        })
        return output

    def test_step(self, data_batch, batch_i):
        x, y = data_batch
        
        y_hat_ = self.forward(x)
        
        
        loss_val = self.loss(y, y_hat_)
        output = OrderedDict({
            'val_loss': loss_val,
            'y' :y,
            'y_hat':y_hat_
        })
        return output
    
    def cal_loss(self, y, y_hat):
        output = torch.sub(y, y_hat)
        output = torch.abs(output)
        if self.criterion == "l2":
            output = output.pow(2)
        
        mean_output = torch.mean(output, dim=1)
        max_output, _ = torch.max(output, dim=1)
        return mean_output, max_output
    
    def validation_step_end(self, outputs):
        y = outputs['y'].squeeze(1)
        y_hat = outputs['y_hat'].squeeze(1)
        loss_val, loss_max = self.cal_loss(y, y_hat)
        return [y, y_hat, loss_val]
    
    def validation_epoch_end(self, outputs):
        print("==============validation epoch end===============")
        y = torch.cat(([output[0] for output in outputs]),0)  
        y_hat = torch.cat(([output[1] for output in outputs]),0)  
        val_loss = torch.cat(([output[2] for output in outputs]), 0)
        np.set_printoptions(suppress=True)
            
    def test_step_end(self, outputs):
        y = outputs['y'].squeeze(1)
        y_hat = outputs['y_hat'].squeeze(1)
        loss_val, loss_max = self.cal_loss(y, y_hat)
        return [y, y_hat, loss_val, loss_max]
    
    def test_epoch_end(self, outputs):
        print("==============test epoch end===============")
        y = torch.cat(([output[0] for output in outputs]),0)  
        y_hat = torch.cat(([output[1] for output in outputs]),0)  
        val_loss = torch.cat(([output[2] for output in outputs]), 0)
        val_max = torch.cat(([output[3] for output in outputs]), 0)
        np.set_printoptions(suppress=True)
        
        if self.on_gpu:
            y = y.cpu()
            y_hat = y_hat.cpu()
            val_loss = val_loss.cpu()
            val_max = val_max.cpu()
        
        try:
            save_data_path = "./%s" %(self.hp.dataset)
            if not os.path.exists(save_data_path):
                os.makedirs(save_data_path)
            if not os.path.exists(save_data_path + "/"+self.hp.data_name):
                os.makedirs(save_data_path + "/" +self.hp.data_name)
            np.savetxt(save_data_path + "/"+self.hp.data_name+"/y_label.txt", np.array(val_loss), delimiter='\n', fmt='%.8f')
            np.save(save_data_path + "/"+self.hp.data_name+"/y.npy", np.array(y))
            np.save(save_data_path + "/"+self.hp.data_name+"/y_hat.npy", np.array(y_hat))
            print("y hat shape is " + str(np.array(y_hat).shape))
        except Exception as e:
            print(e)
    
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
    
    def mydataloader(self, train, test_name=None, batch_s=0):
        set_type = train
        print(set_type + "data loader called...")
        train_sampler = None
        batch_size = self.hp.batch_size
        
        if batch_s == 0:
            batch_size = self.hp.batch_size
        else:
            batch_size = batch_s
        
        if test_name:
            dataset = MTSDataset(window=self.window, horizon=self.hp.horize, \
                data_name=test_name, set_type=set_type, dataset=self.hp.dataset)
        else:
            dataset = MTSDataset(window=self.window, horizon=self.hp.horize, \
                data_name=self.hp.data_name, set_type=set_type, dataset=self.hp.dataset)
            
        
        try:
            if self.on_gpu:
                train_sampler = DistributedSampler(dataset, rank=self.trainer.local_rank)
                batch_size = batch_size // self.trainer.world_size
        except Exception as e:
            print(e)
            print("=============GPU Setting ERROR================")
            
        if set_type == "train":
            shuffle_ = True
        else:
            shuffle_ = False
        
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle_,
            sampler=train_sampler,
            persistent_workers=False
        )
        
        return loader