# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 10:40:24 2025

@author: estumpe
"""

import os
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sklearn.metrics import roc_curve, auc
from collections import defaultdict
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# importing the unsupervised domain adaptation methods
import pytorch_adapt as pa
import pytorch_adapt.containers    # load pa.containers
import pytorch_adapt.layers        # load pa.layers
import pytorch_adapt.hooks         # load pa.hooks
import pytorch_adapt.weighters     # load pa.weighters
import pytorch_adapt.utils         # load pa.utils#
import joblib
from skimage.measure import regionprops, label
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from contextlib import nullcontext

import time

from numpy.lib.stride_tricks import sliding_window_view

#%% The samples


def return_split(name):
    
    if name == "bottom_gall_train":
        
        samples = [
            "exp00_setID0000", #plant 1
            "exp00_setID0002", #plant 1
            "exp00_setID0004", #plant 1
            "exp00_setID0006", #plant 1
            "exp00_setID0008", #plant 1
            "exp01_setID0000", # plant 1
            "exp02_setID0000", # plant 2
            "exp02_setID0002", # plant 2
            "exp02_setID0004", # plant 2
            "exp02_setID0006", # plant 2
            "exp02_setID0008", # plant 2
            "exp02_setID0010", # plant 2
            "exp03_setID0018", #  plant 3
            "exp03_setID0020", #  plant 3
            "exp03_setID0022", #  plant 3
            "exp03_setID0024", #  plant 3
            "exp03_setID0028", #  plant 3
            "exp01_setID0002", #plant 1
            "exp01_setID0004", #plant 1
            "exp01_setID0006", #plant 1
            "exp02_setID0012", # plant 2
            "exp02_setID0014", # plant 2
            "exp02_setID0016", # plant 2
            "exp03_setID0030", # plant 3
            "exp03_setID0032", #  plant 3
            "exp03_setID0034"  #  plant 3
        ]
        
    elif name == "bottom_gall_test":
        
        samples = [
            "exp01_setID0008", #plant 1
            "exp01_setID0010", #plant 1
            "exp01_setID0012", #plant 1
            "exp02_setID0018", # plant 2
            "exp02_setID0020", # plant 2
            "exp02_setID0022", # plant 2
            "exp03_setID0036", #  plant 3
            "exp03_setID0038", #  plant 3
            "exp03_setID0040", #  plant 3
            ]   

    elif name == "top_gall_train":
        
        samples = [
            "exp00_setID0001", #plant 1
            "exp00_setID0003", #plant 1
            "exp00_setID0005", #plant 1
            "exp00_setID0007", #plant 1
            "exp00_setID0009", #plant 1
            "exp01_setID0001", #plant 1
            "exp02_setID0001", # plant 2
            "exp02_setID0003", # plant 2
            "exp02_setID0005", # plant 2
            "exp02_setID0007", # plant 2
            "exp02_setID0009", # plant 2
            "exp02_setID0011", # plant 2
            "exp03_setID0019", #  plant 3
            "exp03_setID0021", #  plant 3
            "exp03_setID0023", #  plant 3
            "exp03_setID0025", #  plant 3
            "exp03_setID0029", #  plant 3
            "exp01_setID0003", #plant 1
            "exp01_setID0005", #plant 1
            "exp01_setID0007", #plant 1
            "exp02_setID0013", # plant 2
            "exp02_setID0015", # plant 2
            "exp02_setID0017", # plant 2
            "exp03_setID0031", #  plant 3
            "exp03_setID0033", #  plant 3
            "exp03_setID0035"  #  plant 3
            ]
    
    elif name == "top_gall_test":
        
        samples = [
            "exp01_setID0009", #plant 1
            "exp01_setID0011", #plant 1
            "exp01_setID0013", #plant 1
            "exp02_setID0019", # plant 2
            "exp02_setID0021", # plant 2
            "exp02_setID0023", # plant 2
            "exp03_setID0037", #  plant 3
            "exp03_setID0039", #  plant 3
            "exp03_setID0041", #  plant 3
                    ]    
                
    elif name == "bottom_gall_trainv2":
        
        samples = [
            "exp00_setID0000", #plant 1
            "exp00_setID0002", #plant 1
            "exp01_setID0000", # plant 1
            "exp02_setID0000", # plant 2
            "exp02_setID0002", # plant 2
            "exp02_setID0010", # plant 2
            "exp03_setID0018", #  plant 3
            "exp03_setID0028", #  plant 3
            "exp01_setID0002", #plant 1
            "exp01_setID0004", #plant 1
            "exp01_setID0006", #plant 1
            "exp02_setID0012", # plant 2
            "exp02_setID0014", # plant 2
            "exp02_setID0016", # plant 2
            "exp03_setID0030", # plant 3
            "exp03_setID0032", #  plant 3
            "exp03_setID0034"  #  plant 3
        ]
            
    elif name == "bottom_gall_valv2":
        
        samples = [
            "exp00_setID0004", #plant 1
            "exp00_setID0006", #plant 1
            "exp00_setID0008", #plant 1
            "exp02_setID0004", # plant 2
            "exp02_setID0006", # plant 2
            "exp02_setID0008", # plant 2            
            "exp03_setID0020", #  plant 3
            "exp03_setID0022", #  plant 3
            "exp03_setID0024", #  plant 3
        ]        


    elif name == "top_gall_trainv2":
        
        samples = [
            "exp00_setID0001", #plant 1
            "exp00_setID0009", #plant 1
            "exp01_setID0001", #plant 1
            "exp02_setID0001", # plant 2
            "exp02_setID0003", # plant 2
            "exp02_setID0011", # plant 2
            "exp03_setID0019", #  plant 3
            "exp03_setID0029", #  plant 3
            "exp01_setID0003", #plant 1
            "exp01_setID0005", #plant 1
            "exp01_setID0007", #plant 1
            "exp02_setID0013", # plant 2
            "exp02_setID0015", # plant 2
            "exp02_setID0017", # plant 2
            "exp03_setID0031", #  plant 3
            "exp03_setID0033", #  plant 3
            "exp03_setID0035"  #  plant 3
            ]

    elif name == "top_gall_valv2":
        
        samples = [
            "exp00_setID0003", #plant 1
            "exp00_setID0005", #plant 1
            "exp00_setID0007", #plant 1            
            "exp02_setID0005", # plant 2
            "exp02_setID0007", # plant 2
            "exp02_setID0009", # plant 2            
            "exp03_setID0021", #  plant 3
            "exp03_setID0023", #  plant 3
            "exp03_setID0025", #  plant 3            
            ]

    elif name == "full_plant":
        
        samples = [
            "exp01_setID0014",
            "exp01_setID0015",
            "exp01_setID0016",
            "exp01_setID0017",
            "exp01_setID0018",
            "exp01_setID0019",
            "exp01_setID0020",
            "exp03_setID0000",
            "exp03_setID0001",
            "exp03_setID0002",
            "exp03_setID0003",
            "exp03_setID0042",
            "exp03_setID0043",
            "exp03_setID0044",
            "exp03_setID0045",
            "exp03_setID0046",
            "exp02_setID0024",
        ]


    elif name == "top_all":
        
        samples = [
            "exp00_setID0001", #plant 1
            "exp00_setID0003", #plant 1
            "exp00_setID0005", #plant 1
            "exp00_setID0007", #plant 1
            "exp00_setID0009", #plant 1
            "exp01_setID0001", #plant 1
            "exp02_setID0001", # plant 2
            "exp02_setID0003", # plant 2
            "exp02_setID0005", # plant 2
            "exp02_setID0007", # plant 2
            "exp02_setID0009", # plant 2
            "exp02_setID0011", # plant 2
            "exp03_setID0019", #  plant 3
            "exp03_setID0021", #  plant 3
            "exp03_setID0023", #  plant 3
            "exp03_setID0025", #  plant 3
            "exp03_setID0029", #  plant 3
            "exp01_setID0003", #plant 1
            "exp01_setID0005", #plant 1
            "exp01_setID0007", #plant 1
            "exp02_setID0013", # plant 2
            "exp02_setID0015", # plant 2
            "exp02_setID0017", # plant 2
            "exp03_setID0031", #  plant 3
            "exp03_setID0033", #  plant 3
            "exp03_setID0035",  #  plant 3
            "exp01_setID0009", #plant 1
            "exp01_setID0011", #plant 1
            "exp01_setID0013", #plant 1
            "exp02_setID0019", # plant 2
            "exp02_setID0021", # plant 2
            "exp02_setID0023", # plant 2
            "exp03_setID0037", #  plant 3
            "exp03_setID0039", #  plant 3
            "exp03_setID0041", #  plant 3
                    ]    

    
    elif name == "bot_all":
        
        samples = [
            "exp00_setID0000", #plant 1
            "exp00_setID0002", #plant 1
            "exp00_setID0004", #plant 1
            "exp00_setID0006", #plant 1
            "exp00_setID0008", #plant 1
            "exp01_setID0000", # plant 1
            "exp02_setID0000", # plant 2
            "exp02_setID0002", # plant 2
            "exp02_setID0004", # plant 2
            "exp02_setID0006", # plant 2
            "exp02_setID0008", # plant 2
            "exp02_setID0010", # plant 2
            "exp03_setID0018", #  plant 3
            "exp03_setID0020", #  plant 3
            "exp03_setID0022", #  plant 3
            "exp03_setID0024", #  plant 3
            "exp03_setID0028", #  plant 3
            "exp01_setID0002", #plant 1
            "exp01_setID0004", #plant 1
            "exp01_setID0006", #plant 1
            "exp02_setID0012", # plant 2
            "exp02_setID0014", # plant 2
            "exp02_setID0016", # plant 2
            "exp03_setID0030", # plant 3
            "exp03_setID0032", #  plant 3
            "exp03_setID0034",  #  plant 3
            "exp01_setID0008", #plant 1
            "exp01_setID0010", #plant 1
            "exp01_setID0012", #plant 1
            "exp02_setID0018", # plant 2
            "exp02_setID0020", # plant 2
            "exp02_setID0022", # plant 2
            "exp03_setID0036", #  plant 3
            "exp03_setID0038", #  plant 3
            "exp03_setID0040", #  plant 3
            ] 
                     
    return samples

def select_hook(method, opts, feature_dim, num_classes, models,device,dataset_size=1024):
    """
    method: one of
      'bsp_bnm', 'mmd', 'coral', 'afn', 'adanbn',
      'atdoc', 'vada', 'dann', 'dann_mcc_atdoc',
      'cdan', 'cdan_vat', 'mcd', 'mcd_afn_mmd'
    opts: your list of optimizers [opt_G, opt_C, opt_D]
    dataset_size, feature_dim, num_classes: ints for ATDOC, CDAN, etc.
    models: your Models({"G":G,"C":C,"D":D})
    """
    misc = {}

    if method == "bsp_bnm":
        hook = pa.hooks.ClassifierHook(
            opts=[opts[0], opts[1]],
            post=[pa.hooks.BSPHook(), pa.hooks.BNMHook()],
            weighter=pa.weighters.MeanWeighter()
        )
    
    elif method == "dann":
        hook = pa.hooks.DANNHook(opts, gradient_reversal_weight=0.1)

    
    elif method == "dann_mcc_atdoc":
        mcc   = pa.hooks.MCCHook(opts)
        atdoc = pa.hooks.ATDOCHook(
            dataset_size=dataset_size,
            feature_dim=feature_dim,
            num_classes=num_classes
        )
        hook = pa.hooks.DANNHook(opts, post_g=[mcc, atdoc])
    
        
    elif method == "cdan":
        misc = {
            "feature_combiner": pa.layers.RandomizedDotProduct(
                [feature_dim, num_classes], feature_dim
            )
        }
        hook = pa.hooks.CDANHook(d_opts=opts[2:], g_opts=opts[:2])        
     
        
    elif method == "cdan_vat":
        misc = {
            "feature_combiner": pa.layers.RandomizedDotProduct(
                [feature_dim, num_classes], feature_dim
            ),
            "combined_model": nn.Sequential(models["G"], models["C"])
        }
        hook = pa.hooks.CDANHook(d_opts=opts[2:], g_opts=opts[:2], post_g=[pa.hooks.VATHook()])    
    

    elif method == "mcd":
        # two‐classifier MCD
        C2 = pa.utils.common_functions.reinit(copy.deepcopy(models["C"]))
        C_multi = pa.layers.MultipleModels(models["C"], C2)
        models["C"] = C_multi
        C_multi.register_forward_hook(lambda *a: setattr(C_multi, "count", C_multi.count + 1))
        models["G"].count, C_multi.count = 0, 0
        hook = pa.hooks.MCDHook(g_opts=opts[:1], c_opts=opts[1:2])
    

    elif method == "mcd_afn_mmd":
        C2 = pa.utils.common_functions.reinit(copy.deepcopy(models["C"]))
        models["C"] = pa.layers.MultipleModels(models["C"], C2).to(device)
        opts_g, opts_c = opts[:1], opts[1:2]
        kernel_scales = pa.layers.utils.get_kernel_scales(low=-3, high=3, num_kernels=10)
        loss_fn = pa.layers.MMDLoss(kernel_scales=kernel_scales)
        hook = pa.hooks.MCDHook(
            g_opts=opts_g,
            c_opts=opts_c,
            post_x=[pa.hooks.AFNHook()],
            # avoid softmax on list-of-logits:
            post_z=[pa.hooks.AlignerHook(loss_fn=loss_fn)]
            # (or use AlignerPlusCHook(opts=opts_c, loss_fn=loss_fn, softmax=False))
        )        
    
    # Paper Reference (https://github.com/KevinMusgrave/pytorch-adapt/blob/main/examples/getting_started/PaperImplementationsAsHooks.ipynb)
    # https://github.com/KevinMusgrave/pytorch-adapt/blob/main/examples/getting_started/PaperImplementationsAsHooks.ipynb
    
    elif method == "adda":
        # 1) deep-copy the source G → target T
        T = copy.deepcopy(models["G"])
        # 2) create a fresh optimizer for T
        T_opt = torch.optim.Adam(T.parameters(), lr=opts[0].param_groups[0]['lr'])
        # 3) register T in the models dict
        models["T"] = T
        # 4) use opts[2] (the D optimizer) as the discriminator optim
        hook = pa.hooks.ADDAHook(g_opts=[T_opt], d_opts=[opts[2]])

    elif method == "afn":
        hook = pa.hooks.ClassifierHook(opts=[opts[0], opts[1]], post=[pa.hooks.AFNHook()])       
        
    elif method == "atdoc": # Laptop jault!
        atdoc = pa.hooks.ATDOCHook(
            dataset_size=dataset_size,
            feature_dim=feature_dim,
            num_classes=num_classes)
        hook = pa.hooks.ClassifierHook(opts=[opts[0], opts[1]], post=[atdoc])
        
    elif method == "bnm": # Laptop jault!
        hook = pa.hooks.ClassifierHook(opts=[opts[0], opts[1]], post=[pa.hooks.BNMHook()])     

    elif method == "bsp": # Laptop jault!
        weighter = pa.weighters.MeanWeighter(weights={"bsp_loss": 1e-3})
        hook = pa.hooks.ClassifierHook(opts=[opts[0], opts[1]], post=[pa.hooks.BSPHook()], weighter=weighter)
      
    elif method == "coral":
        hook = pa.hooks.AlignerPlusCHook(opts=[opts[0], opts[1]], loss_fn=pa.layers.CORALLoss(), softmax=False)
        
    elif method == "dch": #domain confusion
    
        D_ = nn.Sequential(nn.Linear(16,16), nn.ReLU(), nn.Linear(16,2)).to(device)
        models["D"] = D_
        D_opt_ = torch.optim.Adam(D_.parameters())
        hook = pa.hooks.DomainConfusionHook(g_opts=[opts[0], opts[1]], d_opts=[D_opt_])
     
    elif method == "gan":
        hook = pa.hooks.GANHook(g_opts=[opts[0], opts[1]], d_opts=[opts[2]])
        
    elif method == "gvb":
        C_ = pa.layers.ModelWithBridge(models["C"]).to(device)
        D_ = pa.layers.ModelWithBridge(nn.Sequential(nn.Linear(num_classes, 1))).to(device)
        models["C"], models["D"] = C_, D_
        C_opt_ = torch.optim.Adam(C_.parameters(), lr=opts[1].param_groups[0]['lr'])
        D_opt_ = torch.optim.Adam(D_.parameters(), lr=opts[2].param_groups[0]['lr'])
        hook = pa.hooks.GVBHook(opts=[opts[0], C_opt_, D_opt_])    
    
    elif method == "im":
    
        hook = pa.hooks.ClassifierHook(opts=[opts[0], opts[1]], post=[pa.hooks.TargetEntropyHook(), pa.hooks.TargetDiversityHook()])

    elif method == "itl":
        
        hook = pa.hooks.ClassifierHook(
            opts=[opts[0], opts[1]],
            post=[pa.hooks.ISTLossHook(), pa.hooks.TargetEntropyHook(), pa.hooks.TargetDiversityHook()],
        )
    
    elif method == "jmmd":
           
        kernel_scales = pa.layers.utils.get_kernel_scales(low=-3, high=3, num_kernels=10)
        loss_fn = pa.layers.MMDLoss(kernel_scales=kernel_scales)
        aligner_hook = pa.hooks.JointAlignerHook(loss_fn=loss_fn)
        hook = pa.hooks.AlignerPlusCHook(opts=[opts[0], opts[1]], aligner_hook=aligner_hook)
    
    elif method == "mcc":
        
        hook = pa.hooks.ClassifierHook(opts=[opts[0], opts[1]], post=[pa.hooks.MCCHook()])
        
    elif method == "mmd":
        
        kernel_scales = pa.layers.utils.get_kernel_scales(low=-3, high=3, num_kernels=10)
        loss_fn = pa.layers.MMDLoss(kernel_scales=kernel_scales)
        hook = pa.hooks.AlignerPlusCHook(opts=[opts[0], opts[1]], loss_fn=loss_fn)
        
    elif method == "rtn":
        residual_model = pa.layers.PlusResidual(nn.Linear(num_classes, num_classes)).to(device)
        feature_combiner = pa.layers.RandomizedDotProduct([feature_dim, num_classes], feature_dim).to(device)
        scales = pa.layers.utils.get_kernel_scales(low=-3, high=3, num_kernels=10)
        loss_fn = pa.layers.MMDLoss(kernel_scales=scales)
        hook = pa.hooks.RTNHook(opts=[opts[0], opts[1]], aligner_loss_fn=loss_fn)
        misc = {"feature_combiner": feature_combiner}
        models["residual_model"] = residual_model  # keep container and add key

    elif method == "star":
            C_ = pa.layers.StochasticLinear(16, 2)
            C_ = pa.layers.MultipleModels(C_, pa.utils.common_functions.reinit(copy.deepcopy(C_))).to(device)
            models["C"] = C_
            hook = pa.hooks.MCDHook(g_opts=[opts[0]], c_opts=[opts[1]])

    elif method == "swd":
            C_ = pa.layers.MultipleModels(models["C"], pa.utils.common_functions.reinit(copy.deepcopy(models["C"]))).to(device)
            models["C"] = C_
            loss_fn = pa.layers.SlicedWasserstein(m=128)
            hook = pa.hooks.MCDHook(g_opts=[opts[0]], c_opts=[opts[1]], discrepancy_loss_fn=loss_fn)

    elif method == "symnets":
        C_ = pa.layers.MultipleModels(models["C"], pa.utils.common_functions.reinit(copy.deepcopy(models["C"]))).to(device)
        models["C"] = C_
        hook = pa.hooks.SymNetsHook(g_opts=[opts[0]], c_opts=[opts[1]])        

    elif method == "vada":
        combined_model = torch.nn.Sequential(models["G"], models["C"]).to(device)
        misc = {"combined_model": combined_model}
        hook = pa.hooks.VADAHook(g_opts=[opts[0], opts[1]], d_opts=[opts[2]])


    return hook, misc, models


# 3. CombinedLoader for domain adapt hooks
class CombinedLoader:
    def __init__(self, src_loader, tgt_loader):
        self.src_loader = src_loader
        self.tgt_loader = tgt_loader
        self.bs = src_loader.batch_size
    def __iter__(self):
        self.src_iter = iter(self.src_loader)
        self.tgt_iter = iter(self.tgt_loader)
        return self
    def __next__(self):
        xs, ys = next(self.src_iter)
        xt, yt = next(self.tgt_iter)
        idx = torch.arange(self.bs, dtype=torch.long)
        return {
            "src_imgs":       xs,
            "src_labels":     ys,
            "target_imgs":    xt,
            "src_domain":     torch.zeros(self.bs, dtype=torch.float32),
            "target_domain":  torch.ones(self.bs, dtype=torch.float32),
            "src_sample_idx":    idx,
            "target_sample_idx": idx,
        }

def flatten_scalars(obj):
    vals = []
    if isinstance(obj, torch.Tensor):
        vals.append(obj.item())
    elif isinstance(obj, (float, int)):
        vals.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            vals.extend(flatten_scalars(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            vals.extend(flatten_scalars(v))
    return vals


class HSLineShuffleDataset(Dataset):

    def __init__(self, data_dir, file_names, mask_folder, roi,sub_rate,up_rate, class_balance, rep_balance, snv=False, folder = "hs_np", nbhd = 0):
        self.file_names = file_names
        self.img_dir    = os.path.join(data_dir, folder)
        self.mask_dir   = os.path.join(data_dir, mask_folder)
        self.img_files  = [os.path.join(self.img_dir, f"{name}.npy") for name in file_names]
        self.mask_files = [os.path.join(self.mask_dir, f"{name}.png") for name in file_names]
        self.roi        = roi
        self.snv        = snv
        self.nbhd       = nbhd
        self.rep_balance = rep_balance
        self.neg_idx     = 0
     
        self.sub_rate       = sub_rate #subsampling rate
        self.up_rate        = up_rate #subsampling rate        
        self.class_balance  = class_balance #subsampling rate
        
        self.pos_data, self.pos_labels, self.neg_data, self.neg_labels, self.num_pos, self.num_neg = self.load_full_data(self.img_files, self.mask_files, 
                                self.roi, self.sub_rate, self.up_rate, self.class_balance, self.snv, self.nbhd, self.rep_balance) 
        
        self.neg_index_arr = self.init_neg_idx_arr() #self.num_neg, self.num_pos used internally
        
        self.select_shuffled_data()
        
    
    def init_neg_idx_arr(self):
             
        pos_bal = int(self.num_pos*self.class_balance) #class balance
        
        neg_batch_num = int(np.floor(self.num_neg/pos_bal))
        neg_indices   = np.arange(self.num_neg)
        np.random.shuffle(neg_indices)
        neg_indices   = neg_indices[:int(pos_bal*neg_batch_num)]
        neg_index_arr = neg_indices.reshape(neg_batch_num,pos_bal)
        
        return neg_index_arr

    def select_shuffled_data(self):
        
        if self.neg_idx == self.neg_index_arr.shape[0]:
            
            self.neg_index_arr = self.init_neg_idx_arr()
            self.neg_idx = 0
        
        neg_spectra = self.neg_data[self.neg_index_arr[self.neg_idx,:]]
        neg_labels  = self.neg_labels[self.neg_index_arr[self.neg_idx,:]]
        
        self.spectra = np.concatenate([self.pos_data,neg_spectra])
        self.labels  = np.concatenate([self.pos_labels,neg_labels])
        
        self.neg_idx+=1
        
    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectral = torch.from_numpy(self.spectra[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return {"spectra": spectral, "label": label}

    def get_filename(self, idx):
        return self.file_names[idx]

    def load_full_data(self, img_files, mask_files, roi, sub_rate, up_rate, class_balance, snv, nbhd, rep_balance):

        spectra_list, labels_list = [], []
        
        num_pos = 0
        num_neg = 0
        
        pos_spectra_list = []
        neg_spectra_list = []
        
        pos_labels_list  = []
        neg_labels_list  = []
        
    
        for img_path, mask_path in zip(img_files, mask_files):
            
            # img  = np.load(img_path)[roi[0]:roi[1], roi[2]:roi[3], :]
            # mask = imageio.imread(mask_path)[roi[0]:roi[1], roi[2]:roi[3]] // 255
            img = np.load(img_path)
            mask = imageio.imread(mask_path)// 255
    
            if roi != None:
                
                img  = img[roi[0]:roi[1], roi[2]:roi[3], :]
                mask = mask[roi[0]:roi[1], roi[2]:roi[3]] 
    
            pos_indices = np.argwhere(mask==1) # galls
            neg_indices = np.argwhere(mask==0) # leaf area
                            
            print(f"pos samples: {pos_indices.shape[0]} \n neg samples: {neg_indices.shape[0]}")
        
            if snv:
                img = self.apply_snv(img)
            
            pos_spectra = img[pos_indices[:,0],pos_indices[:,1]]            
            neg_spectra = img[neg_indices[:,0],neg_indices[:,1]]    
             
            pos_spectra_list.append(pos_spectra)
            neg_spectra_list.append(neg_spectra)
            
            pos_labels_list.append(np.ones(pos_spectra.shape[0]))
            neg_labels_list.append(np.zeros(neg_spectra.shape[0]))
                
            num_pos += pos_indices.shape[0]
            num_neg += neg_indices.shape[0]
                     
        pos_spectra = np.concatenate(pos_spectra_list, axis = 0)
        neg_spectra = np.concatenate(neg_spectra_list, axis = 0)
        
        pos_labels  = np.concatenate(pos_labels_list, axis = 0)
        neg_labels  = np.concatenate(neg_labels_list, axis = 0)
        
        print(f"dataset used nbhd: {nbhd}, the new shape of images is: {img.shape}")
            
        return  pos_spectra, pos_labels, neg_spectra, neg_labels, num_pos, num_neg  

 
    def apply_snv(self, img):
        img_mean = np.mean(img, axis=-1)
        img_std  = np.std(img, axis=-1)
        img_snv  = (img - np.expand_dims(img_mean, -1)) / np.expand_dims(img_std + 1e-8, -1)
        return img_snv
    
    
    def create_nbhd_arr(self,img,nbhd):
    
        H,W,_ = img.shape
    
        if nbhd % 2 == 0 or nbhd < 1:
            raise ValueError("nbhd must be an odd integer >= 1 (or 0 to disable).")
        
        r = nbhd // 2
    
        # Pad spatially to handle borders cleanly (reflect keeps local statistics)
        padded = np.pad(img, ((r, r), (r, r), (0, 0)), mode="constant") # applies zero padding
        
        # Build the full field of neighborhoods by stacking shifted views.
        # neighbors will be (H, W, nbhd*nbhd, B), ordered row-major from top-left to bottom-right
        neighbors = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                block = padded[r + dy : r + dy + H, r + dx : r + dx + W, :]
                neighbors.append(block)
        neighbors = np.stack(neighbors, axis=2)  # (H, W, K, B), where K = nbhd*nbhd
        
        neighbors = neighbors.reshape(H, W, -1)
        
        return neighbors

class HSImageDataset(Dataset):

    def __init__(self, data_dir, file_names, mask_folder, roi, train=True, snv=False, folder = "hs_np", preload = True, nbhd = 0):
        self.file_names  = file_names
        self.folder      = folder
        self.mask_folder = mask_folder
        self.img_dir     = os.path.join(data_dir,folder)
        self.img_files   = [os.path.join(self.img_dir, f"{name}.npy") for name in file_names]
        self.nbhd        = nbhd
        
        if self.mask_folder:
            self.mask_dir    = os.path.join(data_dir, self.mask_folder)        
            self.mask_files  = [os.path.join(self.mask_dir, f"{name}.png") for name in file_names]
        
        self.train       = train
        self.roi         = roi
        self.snv         = snv
        self.preload     = preload
    
        print("loading image dataset")

        if self.preload:
            self.data       = self.preload_data(self.img_files)
        
        self.set_augmentation(self.train) #--> set set.augment on/off

    def set_augmentation(self,train):
        
        if train:
            self.augment = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(rotate=(-180, 180), scale=(0.9, 1.1), shear=(-30, 30), p=0.9),                
                ToTensorV2(transpose_mask=True),
            ])
        else:
            self.augment = A.Compose([
                ToTensorV2(transpose_mask=True),
            ])            
            
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        
        if self.preload:            
            image = self.data[0][idx]  # [H, W, C] — still a NumPy array
            
            if self.mask_folder:
                mask  = self.data[1][idx]  # [H, W]
        
        else:
            image = self.load_img(self.img_files[idx])
            
            if self.mask_folder:
                mask  = self.load_mask(self.mask_files[idx])  # [H, W]
                
        
        if self.mask_folder:
            augmented = self.augment(image=image, mask=mask)
            image_aug = augmented['image']  # [C, H, W] (tensor)
            mask_aug  = augmented['mask']   # [H, W] (tensor)
        
            return {"image": image_aug, "mask": mask_aug}
        
        else:
            augmented = self.augment(image=image)
            image_aug = augmented['image']  # [C, H, W] (tensor)
        
            return {"image": image_aug}

    def get_filename(self, idx):
        return self.file_names[idx]


    def load_img(self,img_path):
        
        t0 = time.time()

        img  = np.load(img_path)
        
        t1 = time.time()

        print(f"loading image took: {np.round(t1-t0,3)} s")

        if self.roi != None:            
            img  = img[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3], :]

        if self.snv:
            img = self.apply_snv(img)
            
        t2 = time.time()

        print(f"remaining operations took: {np.round(t2-t1,3)} s")
        
        return img
    
    
    def load_mask(self,mask_path):
        
         mask = imageio.imread(mask_path)// 255
        
         if self.roi != None:
             mask = mask[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] 
             
         return mask
        

    def preload_data(self, img_files):

        img_list, mask_list = [], []

        for i,img_path in enumerate(img_files):

            img = self.load_img(img_path)
            img_list.append(img)
            
            if self.mask_folder: # not set to None
                mask_path = self.mask_files[i]
                mask = self.load_mask(mask_path)
                mask_list.append(mask)

            print(f"finished loading: {img_path}")
        
        return img_list,mask_list
        
    
    def apply_snv(self, img):
        img_mean = np.mean(img, axis=-1)
        img_std  = np.std(img, axis=-1)
        img_snv  = (img - np.expand_dims(img_mean, -1)) / np.expand_dims(img_std + 1e-8, -1)
        return img_snv

    def create_nbhd_arr(self,img,nbhd):
    
        H,W,_ = img.shape
    
        if nbhd % 2 == 0 or nbhd < 1:
            raise ValueError("nbhd must be an odd integer >= 1 (or 0 to disable).")
        
        r = nbhd // 2
    
        # Pad spatially to handle borders cleanly (reflect keeps local statistics)
        padded = np.pad(img, ((r, r), (r, r), (0, 0)), mode="constant") # applies zero padding
        
        # Build the full field of neighborhoods by stacking shifted views.
        # neighbors will be (H, W, nbhd*nbhd, B), ordered row-major from top-left to bottom-right
        neighbors = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                block = padded[r + dy : r + dy + H, r + dx : r + dx + W, :]
                neighbors.append(block)
        neighbors = np.stack(neighbors, axis=2)  # (H, W, K, B), where K = nbhd*nbhd
        
        neighbors = neighbors.reshape(H, W, -1)
        
        return neighbors



class HSLineDataset(Dataset):

    def __init__(self, data_dir, file_names, mask_folder, roi,sub_rate,up_rate, class_balance, rep_balance = 0, snv=False, folder = "hs_np", nbhd = 0):
        self.file_names = file_names
        self.img_dir    = os.path.join(data_dir, folder)
        self.mask_dir   = os.path.join(data_dir, mask_folder)
        self.img_files  = [os.path.join(self.img_dir, f"{name}.npy") for name in file_names]
        self.mask_files = [os.path.join(self.mask_dir, f"{name}.png") for name in file_names]
        self.roi        = roi
        self.snv        = snv
        
        self.sub_rate       = sub_rate #subsampling rate
        self.up_rate        = up_rate #subsampling rate        
        self.class_balance  = class_balance #subsampling rate
        
        self.spectra, self.labels       = self.load_data(self.img_files, self.mask_files, 
                                self.roi, self.sub_rate, self.up_rate, self.class_balance, self.snv)
        
        self.num_pos = 10
        self.num_neg = 10
          

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectral = torch.from_numpy(self.spectra[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return {"spectra": spectral, "label": label}


    def get_filename(self, idx):
        return self.file_names[idx]

    def load_data(self, img_files, mask_files, roi, sub_rate, up_rate, class_balance, snv):

        spectra_list, labels_list = [], []

        for img_path, mask_path in zip(img_files, mask_files):
            
            img = np.load(img_path)
            mask = imageio.imread(mask_path)// 255

            if roi != None:
                
                img  = img[roi[0]:roi[1], roi[2]:roi[3], :]
                mask = mask[roi[0]:roi[1], roi[2]:roi[3]] 

            pos_indices = np.argwhere(mask==1) # galls
            neg_indices = np.argwhere(mask==0) # leaf area
                       
            if sub_rate != 0:    
                max_samples         = int(neg_indices.shape[0]/sub_rate)
                sample_indices      = np.arange(neg_indices.shape[0])
                np.random.shuffle(sample_indices)
                sampled_indices     = sample_indices[:max_samples]
                
                neg_indices         = neg_indices[sampled_indices]
            
            if up_rate != 0: 
                pos_indices         = np.repeat(pos_indices,up_rate,axis = 0) 
                
                
            if class_balance!=0:
                max_samples         = int(pos_indices.shape[0]*class_balance)
                sample_indices      = np.arange(neg_indices.shape[0])
                np.random.shuffle(sample_indices)
                sampled_indices     = sample_indices[:max_samples]
                
                neg_indices         = neg_indices[sampled_indices]
             
            print(f"pos samples: {pos_indices.shape[0]} \n neg samples: {neg_indices.shape[0]}")
        
            pos_spectra         = img[pos_indices[:,0],pos_indices[:,1]]            
            neg_spectra         = img[neg_indices[:,0],neg_indices[:,1]]            
            
            if snv:
                pos_spectra = self.apply_snv(pos_spectra)
                neg_spectra = self.apply_snv(neg_spectra) 
             
                
            spectra = np.concatenate([pos_spectra,neg_spectra])
            labels  = np.concatenate([np.ones(pos_spectra.shape[0]),np.zeros(neg_spectra.shape[0])])
            
            spectra_list.append(spectra)
            labels_list.append(labels)
            
        all_spectra = np.concatenate(spectra_list, axis = 0)
        all_labels  = np.concatenate(labels_list, axis = 0)
            
        return  all_spectra, all_labels  
            

    def apply_snv(self, img):
        img_mean = np.mean(img, axis=-1)
        img_std  = np.std(img, axis=-1)
        img_snv  = (img - np.expand_dims(img_mean, -1)) / np.expand_dims(img_std + 1e-8, -1)
        return img_snv


def return_data_loader(samples,folder,shuffle,aug,DATASET_DIR,config, roi, force_img = False,img_folder = "hs_np"):
    
    if config["model_type"] == "img" or force_img:
        
        data = HSImageDataset(DATASET_DIR, 
                                samples, 
                                folder, 
                                roi, 
                                train = aug, 
                                snv = config["snv"],
                                folder = img_folder,
                                nbhd   = config["nbhd"])
        
    elif config["model_type"] == "line":  
        
        if config["model_spec"] in ["3Dconv","2Dconv","3Dspectral","PatchCNN"]:
            
            
            data = HS3DPatchDataset(DATASET_DIR, 
                                    samples, 
                                    folder, 
                                    roi,
                                    config["sub_rate"],
                                    config["up_rate"],
                                    config["class_balance"],
                                    config["rep_balance"],
                                    config["snv"],
                                    folder = img_folder,
                                    nbhd   = config["nbhd"],
                                    out_spec = config["model_spec"],
                                    train = aug)


        else:
            
            data = HSLineShuffleDataset(DATASET_DIR, 
                                    samples, 
                                    folder, 
                                    roi,
                                    config["sub_rate"],
                                    config["up_rate"],
                                    config["class_balance"],
                                    config["rep_balance"],
                                    config["snv"],
                                    folder = img_folder,
                                    nbhd   = config["nbhd"])
 
            print(f"{folder} dataset contains samples: {data.num_pos} positive, {data.num_neg} negative")
    
    loader = DataLoader(data, batch_size=config["batch_size"], shuffle=shuffle)      
    
    return data,loader


class UNet(nn.Module):
    """
    U-Net with optional BatchNorm and Dropout.

    Args:
        in_channels (int): # input channels
        out_channels (int): # output channels
        sub (int, optional): channel downscaling factor. Defaults to 1.
        batch_norm (bool, optional): whether to use BatchNorm2d. Defaults to False.
        dropout (float|None, optional): Dropout probability; None or 0.0 disables. Defaults to None.
    """
    def __init__(self, in_channels, out_channels, sub=1, dropout=None, batch_norm=False):
        super().__init__()
        self.batch_norm = bool(batch_norm)
        self.p = 0.0 if (dropout is None or float(dropout) == 0.0) else float(dropout)

        def conv_block(in_c, out_c):
            layers = []
            # Common practice: no bias when BatchNorm follows
            bias = not self.batch_norm

            # Conv -> (BN) -> ReLU -> (Dropout)
            layers += [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=bias)]
            if self.batch_norm:
                layers += [nn.BatchNorm2d(out_c)]
            layers += [nn.ReLU(inplace=True)]
            if self.p > 0.0:
                layers += [nn.Dropout2d(self.p)]

            # Conv -> (BN) -> ReLU -> (Dropout)
            layers += [nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=bias)]
            if self.batch_norm:
                layers += [nn.BatchNorm2d(out_c)]
            layers += [nn.ReLU(inplace=True)]
            if self.p > 0.0:
                layers += [nn.Dropout2d(self.p)]

            return nn.Sequential(*layers)

        c64   = int(64 / sub)
        c128  = int(128 / sub)
        c256  = int(256 / sub)
        c512  = int(512 / sub)
        c1024 = int(1024 / sub)

        # Encoder
        self.enc1 = conv_block(in_channels, c64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(c64, c128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(c128, c256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(c256, c512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(c512, c1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(c1024, c512, kernel_size=2, stride=2)
        self.dec4 = conv_block(c1024, c512)

        self.upconv3 = nn.ConvTranspose2d(c512, c256, kernel_size=2, stride=2)
        self.dec3 = conv_block(c512, c256)

        self.upconv2 = nn.ConvTranspose2d(c256, c128, kernel_size=2, stride=2)
        self.dec2 = conv_block(c256, c128)

        self.upconv1 = nn.ConvTranspose2d(c128, c64, kernel_size=2, stride=2)
        self.dec1 = conv_block(c128, c64)

        # Final layer (bias=True since no BN after)
        self.final_conv = nn.Conv2d(c64, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.final_conv(dec1)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        feature_dim=64,      
        num_classes=1,     
        dropout=None,
        batch_norm=False,
        classification_layer=True
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, feature_dim) 
        self.fc3 = nn.Linear(feature_dim, num_classes) if classification_layer else None  

        self.dropout = dropout
        self.batch_norm = batch_norm
        self.classification_layer = classification_layer

        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout is not None else None
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(feature_dim)    
        else:
            self.bn1 = None
            self.bn2 = None

        self.feature_dim = feature_dim          

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        if self.batch_norm and self.bn1 is not None:
            x = self.bn1(x)
        if self.dropout_layer:
            x = self.dropout_layer(x)

        x = torch.relu(self.fc2(x))
        if self.batch_norm and self.bn2 is not None:
            x = self.bn2(x)
        if self.dropout_layer:
            x = self.dropout_layer(x)

        if self.classification_layer and self.fc3 is not None:
            x = self.fc3(x)                          # logits
        return x                                      # features if no classification layer
 
    
def model_selection(config,device):
          
    if config["model_spec"]  == "unet":
        model = UNet(in_channels=config["channel_num"], out_channels=1,sub=1,batch_norm=config["batch_norm"],dropout=config["dropout"])
    
    elif config["model_spec"]  == "mlp":  
        model = MLP(input_dim=config["channel_num"], batch_norm=config["batch_norm"],dropout=config["dropout"],feature_dim = config["feature_dim"])    
             
    elif config["model_spec"] == "PatchCNN":
        model = PatchCNN(input_dim=config["channel_num"],feature_dim=16,bottleneck=config["bottleneck"])
    
    print(config["model_spec"])
    
    model = model.to(device)
        
    return model


def compute_iou(outputs, labels, threshold=0.5, apply_sigmoid = True):
    
    if apply_sigmoid:
        outputs = torch.sigmoid(outputs)
    
    outputs = (outputs > threshold).bool()
    labels = labels.bool()
    intersection = (outputs & labels).float().sum((1, 2, 3))
    union = (outputs | labels).float().sum((1, 2, 3))
    iou = intersection / union
    return iou.mean().item()

def compute_dice(outputs, labels, threshold=0.5, apply_sigmoid = True):
    # Ensure outputs and labels are the same dimensions and 4D tensors
    
    if apply_sigmoid:
        outputs = torch.sigmoid(outputs)
    
    if outputs.dim() == 3:  # Single image, no batch dimension
        outputs = outputs.unsqueeze(0)
    if labels.dim() == 3:  # Single image, no batch dimension
        labels = labels.unsqueeze(0)

    # Convert to binary masks using the threshold
    outputs = (outputs > threshold).bool()
    labels = labels.bool()

    # Compute intersection and union
    intersection = (outputs & labels).float().sum((1, 2, 3))
    dice = (2 * intersection) / (outputs.float().sum((1, 2, 3)) + labels.float().sum((1, 2, 3)) + 1e-6)

    return dice.mean().item() 


def binary_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Computes accuracy for binary classification using raw logits.
    
    Args:
        logits (torch.Tensor): Raw model outputs of shape (B,).
        labels (torch.Tensor): Ground truth labels of shape (B,), values should be 0 or 1.
        threshold (float): Probability threshold to convert to binary prediction (default: 0.5).
        
    Returns:
        float: Accuracy as a percentage (e.g., 0.93 means 93% correct).
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total

@torch.no_grad()
def evaluate_img_based(model, val_loader, criterion,device):
    model.eval()
    
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    
    for batch in val_loader:
        inputs = batch["image"].to(device)
        labels = batch["mask"].unsqueeze(1).float().to(device)  # Add channel dim and convert to float
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * inputs.size(0)
        
        iou = compute_iou(outputs, labels)
        dice = compute_dice(outputs, labels)
        
        running_iou += iou * inputs.size(0)
        running_dice += dice * inputs.size(0)
    
    total_samples = len(val_loader.dataset)
    avg_loss = running_loss / total_samples
    avg_iou = running_iou / total_samples
    avg_dice = running_dice / total_samples
    
    return avg_loss, avg_iou, avg_dice


@torch.no_grad()
def evaluate_line_based(model, val_loader, criterion,device,config):
    model.eval()
    
    running_loss = 0.0
    running_acc = 0.0
    
    for batch in val_loader:
                
        inputs = batch["spectra"].to(device) 

        labels = batch["label"].float().to(device)
        
        
        outputs = model(inputs).squeeze()
        
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * inputs.size(0)
        
        acc = binary_accuracy_from_logits(outputs, labels)
        running_acc  += acc * inputs.size(0)
    
    total_samples = len(val_loader.dataset)
    avg_loss = running_loss / total_samples
    avg_acc = running_acc / total_samples
    
    return avg_loss, avg_acc


def create_comp_array(comps):
    
    arr = np.zeros((comps[1].shape[0],comps[1].shape[1],np.max(comps[1])))

    for i in range(np.max(comps[1])):
        coords = np.argwhere(comps[1]==i+1)    
    
        arr[coords[:,0],coords[:,1],i] = 1
    
    arr = arr.astype(bool)
        
    return arr


def create_iou_arr(gt_arr,pred_arr):
    
    intersection_arr    = np.zeros((gt_arr.shape[-1],pred_arr.shape[-1]))
    union_arr           = np.zeros((gt_arr.shape[-1],pred_arr.shape[-1])) 
    iou_arr             = np.zeros((gt_arr.shape[-1],pred_arr.shape[-1]))  
      
    for i in range(gt_arr.shape[-1]):
        
        gt_mask = gt_arr[:,:,i]
        
        intersection = np.zeros_like(pred_arr, dtype=np.uint8)
        intersection = (gt_mask[:, :, np.newaxis] & pred_arr)
        
        union        = np.zeros_like(pred_arr, dtype=np.uint8)
        union        = (gt_mask[:, :, np.newaxis] | pred_arr)
        
        intersection_sum    = np.sum(intersection,axis= (0,1))
        union_sum           = np.sum(union,axis= (0,1))
        
        iou                 =  intersection_sum/union_sum
        
        intersection_arr[i,:] = intersection_sum
        union_arr[i,:]        = union_sum
        iou_arr[i,:]          = iou 
        
    return iou_arr,intersection_arr,union_arr


    
def compute_obj_detection(mask,pred_thresh,thresh):
    
    # create connected components
    gt_comps    = cv2.connectedComponentsWithStats((mask*255).astype(np.uint8))
    pred_comps  = cv2.connectedComponentsWithStats((pred_thresh*255).astype(np.uint8))
    
    # create component array (each gall mask (bool) in a separate channel c: (h x w x c))
    gt_arr      = create_comp_array(gt_comps)
    pred_arr    = create_comp_array(pred_comps) 
    
    # create IOU array
    iou_arr,_,_ = create_iou_arr(gt_arr,pred_arr)
    
    # gt based lists
    tp_gt_list      = []
    fn_gt_list      = []
    
    # pred based lists
    tp_pred_list    = []
    fn_pred_list    = []
    fp_zero_pred_list = []
    fp_low_pred_list = []
    
    if iou_arr.shape[1] == 0:
           
        score_dict = {}

        score_dict["TP"] = 0
        score_dict["FP"] = 0
        score_dict["FN"] = 0
        
        score_dict["precision"] = 0.0
        score_dict["recall"]    = 0.0
        score_dict["F1"]        = 0.0
        
        score_dict["TP_GT"] = 0
        score_dict["FN_GT"] = 0
        score_dict["FP_GT"] = 0
        score_dict["FP_GT_low"] = 0
        
        score_dict["TP_PR"] = 0
        score_dict["FP_PR"] = 0
        score_dict["FP_PR_0"] = 0
        score_dict["FP_PR_low"] = 0
        
        return score_dict, [], [], None, None, False
    
    
    for i in range(iou_arr.shape[0]):
        
        gt_match_value = np.max(iou_arr[i,:])
        gt_match_index = np.argmax(iou_arr[i,:])
        
        pred_match_value = np.max(iou_arr[:,gt_match_index])
        pred_match_index = np.argmax(iou_arr[:,gt_match_index])
        
        if (pred_match_index == i and gt_match_value>=thresh and gt_match_value>0):
            
            tp_gt_list.append(i)
            tp_pred_list.append(gt_match_index)
                   
        elif(pred_match_index == i and gt_match_value<thresh and gt_match_value>0):
            
            fn_gt_list.append(i)
            fn_pred_list.append(gt_match_index)
            
        elif(gt_match_value==0):
            
            fn_gt_list.append(i)
            
        else:
            fn_gt_list.append(i)
            
            
    index_pred_list = tp_pred_list + fn_pred_list
    
    for i in range(iou_arr.shape[1]):
        
        if(i in index_pred_list):
            continue
        
        pred_match_value   = np.max(iou_arr[:,i])
        pred_match_index   = np.argmax(iou_arr[:,i])
        
        gt_match_value = np.max(iou_arr[pred_match_index,:])
        gt_match_index = np.argmax(iou_arr[pred_match_index,:])
        
        
        if(pred_match_value == 0):
            fp_zero_pred_list.append(i)
        else:
            fp_low_pred_list.append(i)
            
    gt_set      = set(tp_gt_list + fn_gt_list) 
    pred_set    = set(tp_pred_list + fn_pred_list + fp_zero_pred_list + fp_low_pred_list)
    
    assert(len(gt_set) == iou_arr.shape[0])
    assert(len(pred_set) == iou_arr.shape[1])   
    
    TP = len(tp_gt_list)
    FP = len(fp_zero_pred_list + fp_low_pred_list)
    FN = len(fn_gt_list)
    
    precision   = (TP)/(TP+FP+0.00001) 
    recall      = (TP)/(TP+FN+0.00001)
    
    if precision == 0 and recall == 0:
        F1 = 0  # Or another value that makes sense for your application
    else:
        F1 = (2 * precision * recall + 0.00001) / (precision + recall + 0.00001)    
    
    score_dict = {}
    score_dict["TP"] = TP
    score_dict["FP"] = FP
    score_dict["FN"] = FN
    
    score_dict["precision"] = precision
    score_dict["recall"]    = recall
    score_dict["F1"]        = F1
    
    score_dict["TP_GT"] = len(tp_gt_list)
    score_dict["FN_GT"] = len(fn_gt_list)
    
    score_dict["TP_PR"] = len(tp_pred_list)
    score_dict["FP_PR_0"] = len(fp_zero_pred_list)
    score_dict["FP_PR_low"] = len(fp_low_pred_list)
    
    gt_dict_list = [
        {"name":"TP",       "list": tp_gt_list,         "color": [0, 255, 0]},      # green
        {"name":"FN",       "list": fn_gt_list,         "color": [255, 0, 0]},      # red
        ]
    
    pred_dict_list = [
        {"name":"TP",       "list": tp_pred_list,       "color": [0, 255, 0]},      # green
        {"name":"FP",       "list": fp_zero_pred_list + fp_low_pred_list,       "color": [65, 105, 225]},   # blue
        {"name":"FN",       "list": fp_zero_pred_list + fp_low_pred_list,       "color": [65, 105, 225]},
        ]
    
    return score_dict, gt_dict_list,pred_dict_list,gt_arr,pred_arr,True


def create_colored_segmentation(mask_array,
                                pred_dict_list,
                                mode="fill",
                                thickness=1,
                                offset=1,
                                paint_on = None):
    """
    Args:
        mask_array: np.ndarray of shape (H, W, n), binary masks per channel
        pred_dict_list: list of dicts with
            'name': str,
            'list': list of channel-indices,
            'color': [R, G, B]
        mode: "fill" or "bbox"
        thickness: line-thickness for bbox
        offset: how many pixels outside the true bbox to draw the box
    Returns:
        rgb_image: np.ndarray of shape (H, W, 3), uint8
    """
    
    if isinstance(paint_on, np.ndarray):
        paint_on_mode = True
    else:
        paint_on_mode = False
           
    h, w, _ = mask_array.shape
    
    if paint_on_mode:
        rgb = paint_on.copy()
    else:
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for entry in pred_dict_list:
        
        if paint_on_mode: # for paint on mode only paint the bounding boxes for FP     
            if entry["name"] != "FP":
                continue
        
        color = entry["color"]
        for idx in entry["list"]:
            mask = (mask_array[:, :, idx] > 0)

            if mode == "fill":
                # exactly as before
                for c in range(3):
                    rgb[:, :, c] = np.where(mask, color[c], rgb[:, :, c])

            elif mode == "bbox":
                # 1) fill segment in white
                           
                if not paint_on_mode:
                    rgb[mask] = [255, 255, 255]

                # 2) label & extract bboxes
                labels = label(mask)
                for region in regionprops(labels):
                    
                    minr, minc, maxr, maxc = region.bbox

                    # compute outer box coords
                    top    = max(minr - offset, 0)
                    left   = max(minc - offset, 0)
                    bottom = min(maxr + offset, h)
                    right  = min(maxc + offset, w)

                    # draw top edge
                    rgb[top : top + thickness, left : right] = color
                    # draw bottom edge
                    rgb[bottom - thickness : bottom, left : right] = color
                    # draw left edge
                    rgb[top : bottom, left : left + thickness] = color
                    # draw right edge
                    rgb[top : bottom, right - thickness : right] = color

            else:
                raise ValueError(f"Unknown mode {mode!r}")
    return rgb


def run_object_detection_eval(mask,pred_thresh,thresh):
          
    score_dict, gt_dict_list,pred_dict_list,gt_arr,pred_arr,ret = compute_obj_detection(mask,pred_thresh,thresh)

    if ret:
        
        description = f"pr: {np.round(score_dict['precision'],2)}, re: {np.round(score_dict['recall'],2)}, F1: {np.round(score_dict['F1'],2)}"
    
        gt_vis      = create_colored_segmentation(gt_arr, gt_dict_list, mode="bbox", thickness = 1) 
        pred_vis    = create_colored_segmentation(pred_arr, pred_dict_list, mode="bbox", thickness = 1) 
        gt_vis      = create_colored_segmentation(pred_arr, pred_dict_list, mode="bbox", thickness = 1, paint_on = gt_vis)
    
    
        return score_dict,gt_vis,pred_vis,description

    else: 
        
        description = "Empty prediction or ground truth"
        
        return score_dict, None, None, description
        
    
def visualize_segmentation_overlap(gt_mask, pred_mask):
    """
    Args:
        gt_mask: np.ndarray of shape (H, W), values 0 or 1
        pred_mask: np.ndarray of shape (H, W), values 0 or 1

    Returns:
        rgb_image: np.ndarray of shape (H, W, 3), uint8 RGB visualization
    """
    h, w = gt_mask.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    # True Positives (green)
    tp = (gt_mask == 1) & (pred_mask == 1)
    rgb_image[tp] = [0, 255, 0]

    # False Negatives (red)
    fn = (gt_mask == 1) & (pred_mask == 0)
    rgb_image[fn] = [255, 0, 0]

    # False Positives (blue)
    fp = (gt_mask == 0) & (pred_mask == 1)
    rgb_image[fp] = [0, 0, 255]

    return rgb_image

def create_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def convert_for_json(obj):
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "item"):  # for numpy scalars (e.g., np.int32)
        try:
            return obj.item()
        except:
            return str(obj)
    else:
        return obj


def combine_classifier_outputs(out, how="logits"):
    """
    out: Tensor or list/tuple of [B, K] tensors from MultipleModels
    how: "logits" -> average logits (then softmax)
         "probs"  -> average probabilities (softmax each, then mean)
    returns: Tensor [B, K]
    """
    if isinstance(out, (list, tuple)):
        if how == "probs":
            outs = [torch.softmax(o, dim=1) for o in out]
            return torch.stack(outs, dim=0).mean(dim=0)
        # default: average logits
        return torch.stack(out, dim=0).mean(dim=0)
    return out


def evaluation_loop(data,model,device,OUT_DIR,config,split_set,set_folder,sample_names,split,model_type,ml_framework,extended_eval = True):
     
    results = []
    
    vis_dir  = create_make_dir(os.path.join(OUT_DIR,split,split_set))
    plot_dir = create_make_dir(os.path.join(vis_dir,"plots"))
    pred_dir = create_make_dir(os.path.join(vis_dir,"predictions"))
    gt_dir   = create_make_dir(os.path.join(vis_dir,"gt"))
    
    iou_dice_dir = create_make_dir(os.path.join(vis_dir,"iou_dice"))
    roc_auc_dir  = create_make_dir(os.path.join(vis_dir,"roc_auc"))
    obj_det_dir  = create_make_dir(os.path.join(vis_dir,"obj_det"))
     
    with torch.no_grad():     
        
        for i,sample in enumerate(data):
    
            image_tensor = sample["image"]
            mask_tensor  = sample["mask"]
            
            labels = mask_tensor.unsqueeze(0).unsqueeze(1).float().to(device)
            
            
            t0 = time.time()
            output = model_predict(image_tensor,model,device,model_type,ml_framework,config)
            t1 = time.time()
            
            print(f"inference took: {np.round(t1-t0,4)}")
            
            pred_mask_np = output.squeeze().cpu().numpy()
    
            # Convert input and ground truth to numpy
            img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            mask_np = mask_tensor.cpu().numpy()
            
            img_np_thresh = (pred_mask_np >= 0.5).astype(np.uint8)   
            
            iou  = compute_iou(output, labels,apply_sigmoid=False)
            dice = compute_dice(output, labels,apply_sigmoid=False)            
            
            score_dict,gt_vis,pred_vis,description = run_object_detection_eval(mask_np,img_np_thresh,0.2)
                    
            score_dict["iou"]  = iou
            score_dict["dice"] = dice
            score_dict["name"] = sample_names[i]
            score_dict["set"]  = set_folder
        
            if not config["minimal_run"]:
                
                overlap_vis = visualize_segmentation_overlap(mask_np, img_np_thresh)
        
                # Create figure with num_samples rows and 3 columns
                fig, axes = plt.subplots(2, 3, figsize=(16, 10))
                
                # Plot input image
                axes[0, 0].imshow(img_np[:,:,50])
                axes[0, 0].set_title("Input Image")
                axes[0, 0].axis('off')
                
                # Plot ground truth mask
                axes[0, 1].imshow(mask_np, cmap='gray')
                axes[0, 1].set_title(f"Ground Truth: {score_dict['set']}")
                axes[0, 1].axis('off')
    
                # Plot prediction mask
                axes[0, 2].imshow(pred_mask_np, cmap='gray',vmin=0, vmax=1)
                axes[0, 2].set_title("Prediction")
                axes[0, 2].axis('off')
                
                # Plot overlap
                axes[1, 0].imshow(overlap_vis)
                axes[1, 0].set_title(f"Overlap: iou: {np.round(iou,2)}, dice: {np.round(dice,2)}")
                axes[1, 0].axis('off')
    
                if description!="Empty prediction or ground truth":
                    # Plot gt object detection
                    axes[1, 1].imshow(gt_vis)
                    axes[1, 1].set_title(f"GT: {description}")
                    axes[1, 1].axis('off')
    
                    # Plot prediction object detection
                    axes[1, 2].imshow(pred_vis)
                    axes[1, 2].set_title(f"Pred: {description}")
                    axes[1, 2].axis('off') 
                
                else:
                    axes[1, 1].imshow(np.zeros(mask_np.shape),cmap="gray")
                    axes[1, 1].set_title(f"GT: {description}")
                    axes[1, 1].axis('off')
                    
                    # Plot prediction object detection
                    axes[1, 2].imshow(np.zeros(mask_np.shape),cmap="gray")
                    axes[1, 2].set_title(f"Pred: {description}")
                    axes[1, 2].axis('off') 
    
                #save figures, images
                plt.savefig(os.path.join(plot_dir,score_dict["name"]+".png"),dpi=120)                
                np.save(os.path.join(pred_dir,score_dict["name"]+".npy"),pred_mask_np)
                np.save(os.path.join(gt_dir,score_dict["name"]+".npy"),mask_np)
                 
                if extended_eval:
                    
                    print("performing extensive evaluation")
                
                
                    # compute threshold variation metrics
                    
                    # iou and dice threshhold variation
                    thresh_range = np.arange(0.01,0.99,0.01)
                    iou_list  = [compute_iou(output,labels,threshold=thr,apply_sigmoid=False) for thr in thresh_range]
                    dice_list = [compute_dice(output, labels,threshold=thr,apply_sigmoid=False) for thr in thresh_range]
                    
                    score_dict["thresh_range"] = thresh_range
                    score_dict["iou_list"]     = iou_list
                    score_dict["dice_list"]     = iou_list
                    
                    plt.figure()
                    plt.plot(thresh_range, iou_list, label='IOU')
                    plt.plot(thresh_range, dice_list, label='DICE')
                    plt.title('IOU/DICE over thresholds')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.legend(loc="lower right")
                    
                    plt.savefig(os.path.join(iou_dice_dir,score_dict["name"]+".png"),dpi=60)
                    
                    # compute roc/auc
                    
                    # Compute micro-average ROC curve and ROC area
                    fpr, tpr, thresholds = roc_curve(mask_np.ravel(), pred_mask_np.ravel())
                    roc_auc     = auc(fpr, tpr)
                       
                    score_dict["fpr"]               = fpr
                    score_dict["tpr"]               = tpr
                    score_dict["roc_thresholds"]    = thresholds
                    score_dict["auc"]               = roc_auc
                     
                    j_scores = tpr - fpr
                    optimal_idx = j_scores.argmax()
                    optimal_threshold = thresholds[optimal_idx]
                    
                    # Plot of a ROC curve for a specific class
                    plt.figure()
                    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver operating characteristic example')
                    plt.legend(loc="lower right")
                    plt.show()
        
                    plt.savefig(os.path.join(roc_auc_dir,score_dict["name"]+".png"),dpi=60)            
                    
                    
                    # iterate thresholds for object detection
        
                    thresh_range = np.arange(0.01,0.99,0.01)
                    score_list = []
                    for thr in thresh_range:
                        
                        img_np_thresh = (pred_mask_np >= thr).astype(np.uint8)
                        
                        thr_score,_,_,_,_,_ = compute_obj_detection(mask_np,img_np_thresh,0.5)
                        
                        score_list.append(thr_score)
                        
                    # Convert to dict of lists
                    aggregated_scores = defaultdict(list)
                    
                    for entry in score_list:
                        for key, value in entry.items():
                            aggregated_scores[key].append(value)
                    
                    # Convert defaultdict back to normal dict (optional)
                    aggregated_scores = dict(aggregated_scores)
                    
                    plt.figure()
                    plt.plot(thresh_range, aggregated_scores["F1"], label='F1')
                    plt.plot(thresh_range, aggregated_scores["precision"], label='precision')
                    plt.plot(thresh_range, aggregated_scores["recall"], label='recall')
                    
                    plt.title('Object detection over thresholds')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.legend(loc="lower right")
        
                    plt.savefig(os.path.join(obj_det_dir,score_dict["name"]+".png"),dpi=60)
                         
                    score_dict["object_detection_scores"] = aggregated_scores
            
            results.append(score_dict)
            
            print(f"finished eval of {i+1} of {len(data)} for {split_set}")
                       
            plt.close('all')
        
        return results 


def make_scheduler(optimizer, scheduler_type, model_type = "line"):
    """
    Returns a ReduceLROnPlateau scheduler with sensible defaults for your task.
    No other config keys required.
    """
    
    assert model_type in  ["line","img"]
    
    if scheduler_type == "reduce_on_plateau":
        # Monitor validation accuracy (higher is better)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=0.5, patience=8, threshold=0.002, threshold_mode="rel", # increase patience
            # factor=0.5, patience=8, threshold=0.002, threshold_mode="rel", #
            cooldown=2, min_lr=1e-6, verbose=True)
        

def training_loop_line(model, config, train_loader, test_loaders,num_epochs,results_dict,criterion,optimizer,device,model_path,scheduler=None,train_data = None):
    
    
    if "transfer" in config.keys():
        
        if config["transfer"] == "pretrained":
            
            print(f"loading pretrained model from: {config['weight_path']}")
            model.load_state_dict(torch.load(config["weight_path"]))
    
    train_epoch_metrics = {"set":config["train_set"],
                           "folder":config["train_set_folder"],
                           "acc":[],
                           "loss":[]}

    test_epoch_metrics  = [{"set":f,"folder":g,"acc":[],"loss":[]} for f,g in zip(config["test_sets"],config["test_set_folders"])]


    validation_loss = np.inf


    torch.save(model.state_dict(), model_path)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc  = 0.0
        
        val_metric_for_sched = None
         
        for batch in train_loader:
            
            inputs = batch["spectra"].to(device)
    
            labels = batch["label"].float().to(device)  # Add channel dim and convert to float        

            optimizer.zero_grad()
            
            outputs = model(inputs).squeeze()
                    
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or Inf detected in loss at epoch {epoch+1}")
                continue
            
            loss.backward()
            optimizer.step()
            
            acc = binary_accuracy_from_logits(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_acc  += acc * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)
        
        # save in metrics
        train_epoch_metrics["acc"].append(epoch_acc)
        train_epoch_metrics["loss"].append(epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        
        val_complete = False
        
        for i,loader in enumerate(test_loaders):
            #print("got here")
            val_loss, val_acc = evaluate_line_based(model, loader, criterion,device,config)
            test_epoch_metrics[i]["acc"].append(val_acc)
            test_epoch_metrics[i]["loss"].append(val_loss) 
        
            is_val = (config["test_sets"][i] == config["val"])
            
            if config["early_stopping"] and is_val and not val_complete:
                                
                if val_loss < validation_loss:
                    
                    print(f"Lowest validation loss: {np.round(val_loss,5)}, saving model")
                    torch.save(model.state_dict(), model_path)
                    
                    validation_loss = val_loss
                
            if scheduler is not None and is_val and not val_complete:
                val_metric_for_sched = val_loss  # monitor accuracy
        
            if is_val:
                val_complete = True
        
        if scheduler is not None and val_metric_for_sched is not None:
            scheduler.step(val_metric_for_sched) 
                    
        print(f"Epoch {epoch+1}/{num_epochs}, Last Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
        if config["shuffle_line_data"]:
            print("shuffle select")
            train_data.select_shuffled_data()
            train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
        
    if config["early_stopping"]:
     
        model.load_state_dict(torch.load(model_path))
         
    results_dict["epoch_results"] = {}
    results_dict["epoch_results"]["train"] = train_epoch_metrics
    results_dict["epoch_results"]["test"]  = test_epoch_metrics
    
    return model, results_dict
    
    
def training_loop_img(model, config, train_loader, test_loaders, num_epochs, results_dict,criterion,optimizer,device,model_path,scheduler = None):
    
    if "transfer" in config.keys():
        
        if config["transfer"] == "pretrained":
            
            print(f"loading pretrained model from: {config['weight_path']}")
            model.load_state_dict(torch.load(config["weight_path"]))
            
    # Setting up epoch metric tracking
    train_epoch_metrics = {"set":config["train_set"],
                           "folder":config["train_set_folder"],
                           "iou":[],
                           "dice":[],
                           "loss":[]}
    
    test_epoch_metrics  = [{"set":f,"folder":g,"iou":[],"dice":[],"loss":[]} for f,g in zip(config["test_sets"],config["test_set_folders"])]
    
    validation_loss = np.inf
    validation_iou  = 0
    
    torch.save(model.state_dict(), model_path)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        for batch in train_loader:

            inputs = batch["image"].to(device)
            labels = batch["mask"].unsqueeze(1).float().to(device)  # Add channel dim and convert to float        

            optimizer.zero_grad()
            
            outputs = model(inputs)
                    
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or Inf detected in loss at epoch {epoch+1}")
                continue
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate IoU and Dice for the batch
            iou = compute_iou(outputs, labels)
            dice = compute_dice(outputs, labels)
            running_iou += iou * inputs.size(0)
            running_dice += dice * inputs.size(0)
                
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_iou = running_iou / len(train_loader.dataset)
        epoch_dice = running_dice / len(train_loader.dataset)
        
        # save in metrics
        train_epoch_metrics["iou"].append(epoch_iou)
        train_epoch_metrics["dice"].append(epoch_dice)
        train_epoch_metrics["loss"].append(epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train IoU: {epoch_iou:.4f}, Train Dice: {epoch_dice:.4f}")
        
        val_metric_for_sched = None
        val_complete         = False
        
        for i,loader in enumerate(test_loaders):
            val_loss, val_iou, val_dice = evaluate_img_based(model, loader, criterion,device)
            test_epoch_metrics[i]["iou"].append(val_iou)
            test_epoch_metrics[i]["dice"].append(val_dice)
            test_epoch_metrics[i]["loss"].append(val_loss)
                        
            is_val = (config["test_sets"][i] == config["val"])

            if config["early_stopping"] and is_val and not val_complete:
                
                if val_iou > validation_iou:
                    
                    print(f"Highest IoU: {np.round(val_iou,5)}, saving model")
                    torch.save(model.state_dict(), model_path)
                    
                    validation_iou = val_iou
                    
            if scheduler is not None and is_val and not val_complete:
                val_metric_for_sched = val_loss
            
            if is_val:
                val_complete = True
            
        if scheduler is not None and val_metric_for_sched is not None and epoch>=100:
            scheduler.step(val_metric_for_sched)
                           
        print(f"Epoch {epoch+1}/{num_epochs}, Last Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
        
    if config["early_stopping"]:
     
        model.load_state_dict(torch.load(model_path))

    results_dict["epoch_results"] = {}
    results_dict["epoch_results"]["train"] = train_epoch_metrics
    results_dict["epoch_results"]["test"]  = test_epoch_metrics
    
    return model, results_dict  

@torch.no_grad()
def evaluate_da_model(loader, C, G, device, criterion):
    G.eval(); C.eval()
    correct = 0; total = 0; total_loss = 0.0
    for X, y in loader:
        X = X.to(device); y = y.to(device)
        z = G(X)
        logits = C(z)
    
        logits = combine_classifier_outputs(logits, how="logits")
        
        if isinstance(logits, (list, tuple)):
            logits = torch.stack(logits, dim=0).mean(dim=0)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct/total, total_loss/total


def train_torch_model(OUT_DIR,config,device,results_dict,train_loader,test_loaders,model,criterion,optimizer,scheduler = None, train_data = None):
  
    model_path = os.path.join(OUT_DIR,"model_weights.pth")
    
    num_epochs = config["num_epochs"] 
    
    if config["model_type"] == "line":
        model, results_dict = training_loop_line(model, config, train_loader, test_loaders,num_epochs,results_dict,criterion,optimizer,device,model_path,scheduler,train_data)
    elif config["model_type"] == "img":
        model, results_dict = training_loop_img(model, config, train_loader, test_loaders,num_epochs,results_dict,criterion,optimizer,device,model_path,scheduler)     
    
    torch.save(model.state_dict(), model_path)
    
    epoch_dir = create_make_dir(os.path.join(OUT_DIR,"epoch_results"))
    
    plt.close('all')
    
    epoch_results = results_dict["epoch_results"]
    epoch_result_list = [epoch_results["train"]] + epoch_results["test"]
    
    epoch_ticks = list(np.arange(1,num_epochs+1))
    
    if not config["minimal_run"]:
    
        for metric in ("iou","dice","loss","acc"):
            
            if metric in epoch_results["train"].keys():
            
                plt.figure(figsize=(16,12))
                
                for i,d in enumerate(epoch_result_list):
                    plt.plot(epoch_ticks,d[metric], label = d["set"])
                
                plt.title(metric)
                plt.legend()
                plt.show()
                plt.savefig(os.path.join(epoch_dir,f"{metric}.png"),dpi = 120)
            
    return model, results_dict, model_path 


def initialize_torch_model(ROOT_DIR,config,device,train_data,weight_path=None):
    
    model = model_selection(config,device) 
    
    if config["pretrained"] != None:
        
        print(f"loading pretrained model: {config['pretrained']}")
        
        if weight_path!=None:
            model.load_state_dict(torch.load(weight_path))
        else:
            model.load_state_dict(torch.load(os.path.join(ROOT_DIR,"results",config["pretrained"],'model_weights.pth')))
    
    if config["weighted_loss"]:
        
        # counts from your training set
        num_pos = train_data.num_pos   # e.g., sum(train_labels)
        num_neg = train_data.num_neg   # e.g., len(train_labels) - num_pos
        
        pos_weight = torch.tensor([num_neg / (num_pos + 1e-8)], device=device)
        
        print(f"using a weighted loss of: {num_neg / (num_pos + 1e-8)}")
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
    else:            
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    scheduler = None
    if config["scheduler"] != None:
        scheduler = make_scheduler(optimizer, config["scheduler"], model_type = config["model_type"])
    
    return model,criterion,optimizer,scheduler
    
    

class SqueezeLast(nn.Module):
    def forward(self, x):  # x: [B,1]
        return x.squeeze(-1)  # [B]


def initialize_da_model(ROOT_DIR,config,device,train_data,dataset_size = None,weight_path = None):
        
    
    if config["model_spec"] == "mlp":
        G = MLP(input_dim=config["channel_num"], feature_dim=config["feature_dim"], classification_layer=False, dropout = config["dropout"] ).to(device)
        
    elif config["model_spec"] == "PatchCNN":
        G = PatchCNN(input_dim=config["channel_num"],feature_dim=config["feature_dim"],bottleneck=config["bottleneck"],classification_layer=False).to(device) 
   
    C = nn.Linear(16, 2).to(device)
    
    D = nn.Sequential(
        nn.Linear(16,16), nn.ReLU(),
        nn.Linear(16,1),   # returns [B,1]
        SqueezeLast(),
        
    ).to(device)
     
    model = pa.containers.Models({"G": G, "C": C, "D": D}) 
    
    optimizers = pa.containers.Optimizers((torch.optim.Adam, {"lr":config["lr"]}))
    optimizers.create_with(model)
    optim_list = list(optimizers.values())
    
    feature_dim  = 16    # your G ends in Linear(...,16)
    num_classes  = 2     # your C outputs 2 classes
    
    hook, misc,model = select_hook(
        config["uda_spec"],          # e.g. "cdan"
        optim_list,
        feature_dim,
        num_classes,
        model,
        device
    )
     
    ### 2025-10-08 implement weighted loss ###
    if config["weighted_loss"]:
        
        # counts from your training set
        num_pos   = train_data.num_pos  
        num_neg   = train_data.num_neg  
        num_total = num_pos + num_neg
        w_neg = num_total / (2.0 * num_neg)
        w_pos = num_total / (2.0 * num_pos)
        
        class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        print(f"using a weighted loss of: pos: {w_pos}, neg: {w_neg}")
             
    else:
        # use the same criterion as CLossHook:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    
    if config["pretrained"] != None:
        
        print(f"loading pretrained model: {config['pretrained']}")
        if weight_path!=None:
           model.load_state_dict(torch.load(weight_path))       
        
        else:       
            model.load_state_dict(torch.load(os.path.join(ROOT_DIR,"results",config["pretrained"],'model_weights.pth')))
    
    return model,hook,misc,criterion
    

   
def train_da_model(OUT_DIR,config,device,results_dict,train_data,test_data_splits,model,hook,misc,criterion):
    
    
    def create_da_loader(data,config):
        loader_all = DataLoader(data, batch_size=len(data), shuffle=False)
        data_da = next(iter(loader_all))
        
        Xda = data_da["spectra"].cpu().numpy()
        yda = data_da["label"].cpu().numpy()
        
        da_ds       = TensorDataset(torch.from_numpy(Xda).float(),torch.from_numpy(yda).long())
        da_loader   = DataLoader(da_ds, batch_size=config["batch_size"], shuffle=True, drop_last=True)
        
        return da_loader
        
    
    model_path = os.path.join(OUT_DIR,"model_weights.pth")
    
    torch.save(model.state_dict(), model_path)
    
    
    target_index    = config["test_sets"].index(config["target_set"])
    
    target_data     = test_data_splits[target_index]
    
    
    train_epoch_metrics = {"set":config["train_set"],
                           "folder":config["train_set_folder"],
                           "acc":[],
                           "loss":[]}

    test_epoch_metrics  = [{"set":config["target_set"],
                            "folder":config["test_set_folders"][target_index],
                            "acc":[],
                            "loss":[]}]    

    # manipulate data structure to match domain adapt package
    src_loader = create_da_loader(train_data,config)
    tgt_loader = create_da_loader(target_data,config)

    train_loader = CombinedLoader(src_loader, tgt_loader)

    if config["early_stopping"]:
        
        val_index    = config["test_sets"].index(config["val"])
        val_loader   = create_da_loader(test_data_splits[val_index],config)
        
    num_epochs = config["num_epochs"] 
    
    validation_loss = np.inf
    
    for epoch in range(1, num_epochs+1):
                
        model["G"].train(); model["C"].train(); model["D"].train()
        
        model["G"].train()
        model["C"].train()
        if "D" in model:  # guard
            model["D"].train()
        
        total_loss = 0.0
        for batch in train_loader:
            batch = pa.utils.common_functions.batch_to_device(batch, device)
            outputs, losses = hook({**model,**misc, **batch})
            vals = flatten_scalars(losses)
            batch_loss = sum(vals)
            total_loss += batch_loss
            
        avg_loss = total_loss / len(src_loader)    
        
        model["G"].eval() 
        model["C"].eval()
        if "D" in model:
            model["D"].eval()
        
        src_acc, src_loss = evaluate_da_model(src_loader, model["C"], model["G"], device, criterion)
        tgt_acc, tgt_loss = evaluate_da_model(tgt_loader, model["C"], model["G"], device, criterion)
        
        print(f"[{config['model_spec']:10s}] Epoch {epoch:2d} | loss: {avg_loss:.4f} | src acc: {src_acc*100:.1f}% | tgt acc: {tgt_acc*100:.1f}%")
            
        train_epoch_metrics["acc"].append(src_acc)
        train_epoch_metrics["loss"].append(avg_loss)
    
        test_epoch_metrics[0]["acc"].append(tgt_acc)
        test_epoch_metrics[0]["loss"].append(tgt_loss)
        
        if config["early_stopping"]:
            
            val_acc, val_loss = evaluate_da_model(val_loader, model["C"], model["G"], device, criterion)
            
            if val_loss < validation_loss:
                
                print(f"Lowest validation loss: {np.round(val_loss,5)}, saving model")
                torch.save(model.state_dict(), model_path)
                
                validation_loss = val_loss
                
                
        if config["shuffle_line_data"]:
            
            #time_start = time.time()
            train_data.select_shuffled_data()
            target_data.select_shuffled_data()
            
            # manipulate data structure to match domain adapt package
            src_loader = create_da_loader(train_data,config)
            tgt_loader = create_da_loader(target_data,config)

            train_loader = CombinedLoader(src_loader, tgt_loader)
            
    if config["early_stopping"]:
        
        model.load_state_dict(torch.load(model_path))     
            
    results_dict["epoch_results"] = {}
    results_dict["epoch_results"]["train"] = train_epoch_metrics
    results_dict["epoch_results"]["test"]  = test_epoch_metrics
        
    epoch_dir = create_make_dir(os.path.join(OUT_DIR,"epoch_results"))

    plt.close('all')

    epoch_results = results_dict["epoch_results"]
    epoch_result_list = [epoch_results["train"]] + epoch_results["test"]

    epoch_ticks = list(np.arange(1,num_epochs+1))


    if not config["minimal_run"]:
        for metric in ("iou","dice","loss","acc"):
            
            if metric in epoch_results["train"].keys():
            
                plt.figure(figsize=(16,12))
                
                for i,d in enumerate(epoch_result_list):
                    plt.plot(epoch_ticks,d[metric], label = d["set"])
                
                plt.title(metric)
                plt.legend()
                plt.show()
                plt.savefig(os.path.join(epoch_dir,f"{metric}.png"),dpi = 120)
    
    #saving model        
    model_path = os.path.join(OUT_DIR,"model_weights.pth")
    torch.save(model.state_dict(), model_path)
    
    return model, results_dict, model_path    


@torch.no_grad()
def predict_tiled(
    model,
    image,                    # torch.Tensor, [C,H,W] or [H,W]
    patch_size: int = 128,
    overlap: int = 32,        # 0..(patch_size-1); try 16-64 to reduce seams
    batch_size: int = 16,
    amp: bool = False,        # set True if you use autocast on GPU
    device: torch.device | str | None = None,
    transform=None            # optional callable applied to each input patch
):
    """
    Sliding-window tiled inference for encoder-decoder models trained on fixed patch size.
    Assumes the model outputs a per-pixel prediction with the same HxW as the input patch.

    Returns:
        pred (torch.Tensor): prediction with same HxW as input image (and matching channel dims)
    """

    # ----- Normalize input shape to [N=1,C,H,W] -----
    inp = image
    if inp.dim() == 2:
        inp = inp.unsqueeze(0)       # [1,H,W]
    if inp.dim() == 3:
        inp = inp.unsqueeze(0)       # [1,C,H,W]
    if inp.dim() != 4:
        raise ValueError(f"Expected image of shape [C,H,W] or [H,W], got {tuple(image.shape)}")

    N, C, H, W = 1, inp.size(1), inp.size(2), inp.size(3)

    # ----- Device handling -----
    if device is None:
        device = next(model.parameters()).device
    inp = inp.to(device)

    # ----- Stride & padding to cover whole image -----
    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be < patch_size")

    # Amount of extra pixels needed so patches tile perfectly
    pad_h_needed = (-(H - patch_size)) % stride if H > patch_size else (patch_size - H) % stride
    pad_w_needed = (-(W - patch_size)) % stride if W > patch_size else (patch_size - W) % stride

    # If the image is smaller than a patch, we need at least that much padding
    pad_h_needed = max(pad_h_needed, max(0, patch_size - H))
    pad_w_needed = max(pad_w_needed, max(0, patch_size - W))

    # Pad on the right/bottom; use reflect to avoid edge artifacts
    pad = (0, pad_w_needed, 0, pad_h_needed)  # (left,right,top,bottom) for F.pad uses (L,R,T,B)
    inp_pad = F.pad(inp, pad, mode="reflect")
    Hp, Wp = inp_pad.size(2), inp_pad.size(3)

    # ----- Extract patches with unfold -----
    patches = F.unfold(inp_pad, kernel_size=patch_size, stride=stride)  # [1, C*ps*ps, L]
    L = patches.size(-1)  # number of patches

    # Prepare for batched model calls: [L, C, ps, ps]
    patches = patches.transpose(1, 2)                       # [1, L, C*ps*ps]
    patches = patches.reshape(L, C, patch_size, patch_size) # [L, C, ps, ps]

    # Optional per-patch transform (e.g., normalization)
    if transform is not None:
        patches = torch.stack([transform(p) for p in patches], dim=0)

    # ----- Run model in batches -----
    model_was_training = model.training
    model.eval()

    preds = []
    autocast_ctx = torch.cuda.amp.autocast if (amp and str(device).startswith("cuda")) else nullcontext
    with autocast_ctx():
        for i in range(0, L, batch_size):
            batch = patches[i:i+batch_size]  # [B, C, ps, ps]
            out = model(batch)               # expect [B, Cout, ps, ps] or [B, ps, ps]
            if out.dim() == 3:               # [B, ps, ps] -> add channel dim
                out = out.unsqueeze(1)
            preds.append(out)
    pred_patches = torch.cat(preds, dim=0)  # [L, Cout, ps, ps]

    # ----- Fold back with weights to blend overlaps -----
    Cout = pred_patches.size(1)

    pred_cols = pred_patches.reshape(L, Cout * patch_size * patch_size)  # [L, Cout*ps*ps]
    pred_cols = pred_cols.transpose(0, 1).unsqueeze(0)                   # [1, Cout*ps*ps, L]

    # A weight window to reduce seams (raised-cosine-ish)
    yy = torch.linspace(-1, 1, patch_size, device=device)
    xx = torch.linspace(-1, 1, patch_size, device=device)
    wy = (1 - yy.abs())
    wx = (1 - xx.abs())
    weight2d = (wy[:, None] * wx[None, :])  # [ps, ps], peak in center
    weight2d = weight2d.clamp_min(1e-6)     # avoid zeros
    weight2d = weight2d / weight2d.max()

    weight_patch = weight2d.expand(Cout, patch_size, patch_size)         # [Cout, ps, ps]
    weight_cols = weight_patch.reshape(1, Cout * patch_size * patch_size, 1).repeat(1, 1, L)

    # Fold predictions and weights
    out_sum = F.fold(pred_cols * weight_cols, output_size=(Hp, Wp),
                     kernel_size=patch_size, stride=stride)              # [1, Cout, Hp, Wp]
    weight_sum = F.fold(weight_cols, output_size=(Hp, Wp),
                        kernel_size=patch_size, stride=stride)           # [1, Cout, Hp, Wp]

    out_blended = out_sum / weight_sum

    # ----- Crop back to original HxW and restore model mode -----
    out_final = out_blended[:, :, :H, :W]  # [1, Cout, H, W]
    if model_was_training:
        model.train()

    #out_final = torch.sigmoid(out_final)
    # Return w/o batch dim if input had no batch
    return out_final.squeeze(0)

#######################################################
#######################################################
#######################################################
##################### MAIN CODE #######################
#######################################################
#######################################################
#######################################################



def get_standard_config():
    
    #if name == "2025_10_21_MLP_top_flex":
                         
    config = {}

    config["exp_name"]         = "name"
           
    # data loading settings
    config["bot_gt_folder"]    = "hs_mask_annot_bot"
    config["top_gt_folder"]    = "hs_mask_annot_top"
    config["roi"]              = [205,333,176,304]
    config["model_type"]       = "line"#"img" #options: "img", "line"
    config["snv"]              = True
    config["aug"]              = True
    config["channel_num"]      = 204
    config["feature_dim"]      = 16

    config["sub_rate"]         = 0
    config["up_rate"]          = 0
    config["class_balance"]    = 1

    config["nbhd"]             = 0
    config["weighted_loss"]    = False

    #torch specific
    config["dropout"]          = None
    config["batch_norm"]       = False

       
    # model settings
    config["ml_framework"]     = "torch" #"torch", "sklearn" "domain_adapt"
    config["model_spec"]       = "mlp" #options: "encoder_decoder", "MLP", "SVM", "FLD" "dann"
    config["num_epochs"]       = 150
    config["batch_size"]       = 1024 # 8
    config["pretrained"]       = None # exp_name of pretrained model or None
    config["lr"]               = 0.005

    config["train_set"]        = "top_gall_trainv2"
    config["test_sets"]        = ["top_gall_valv2","top_gall_test","top_gall_trainv2","top_gall_valv2","top_gall_test"]

    config["train_set_folder"] = "hs_mask_annot_top"
    config["test_set_folders"] = ["hs_mask_annot_top","hs_mask_annot_top","hs_mask_annot_top","hs_mask_annot_top","hs_mask_annot_top"]
    config["val"]              = "top_gall_valv2"

    config["target_set"]       = "top_gall_trainv2"

    config["extended_eval"]    = False
    config["eval_plants"]      = False

    config["early_stopping"]   = True
    
    
    config["minimal_run"]      = True
    
    config["preset"]           = "name"
    
    config["rep_balance"]      = False

    return config    



def model_predict(image_tensor,model,device,model_type,ml_framework,config):
    
    
    def predict_patch_based(image, nbhd, config,model):
        
        img_arr = image.cpu().numpy()
        img_arr = img_arr.transpose(0,2,3,1)
        # img_arr = np.expand_dims(img_arr, axis=0)
        
        #print(img_arr.shape)
        
        win_arr = make_window_view(img_arr,pad_mode='reflect', k = config["nbhd"])

        H, W = img_arr.shape[1],img_arr.shape[2]
        
        # Generate a grid of y and x indices
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        # Stack and reshape into (128*128, 2)
        coords = np.stack([ys, xs], axis=-1).reshape(-1, 2)
        coords = np.concatenate([np.zeros((coords.shape[0],1)),coords],axis = -1).astype(int)

        patches = extract_patches(win_arr,coords)

        patches_torch = torch.from_numpy(patches).float()
        
        
        print("patches_torch shape:", patches_torch.shape)
        print("patches_torch dtype:", patches_torch.dtype)
        print("approx bytes:", patches_torch.numel() * patches_torch.element_size())
        
        if config["model_spec"] in ["PatchCNN"]:
            patches_torch = patches_torch.permute(0, 3, 1, 2).to(device)
            

        dl = DataLoader(TensorDataset(patches_torch), batch_size=config["batch_size"], shuffle=False)

        model.eval()
        pred_chunks = []
        with torch.inference_mode():          # lower memory than no_grad
            for (xb,) in dl:
                xb = xb.to(device, non_blocking=True)
                
                
                if config["ml_framework"] == "domain_adapt":       
                    z = model["G"](xb)                       # [B, 16]
                    logits = combine_classifier_outputs(model["C"](z), how="logits")
                    probs  = torch.softmax(logits, dim=1)
                    out = probs[:, 1]
                    
                else:
                    out = model(xb)               # shape depends on model
                
                pred_chunks.append(out.detach().cpu())

        pred = torch.cat(pred_chunks, dim=0)  # (N, ...)
        return pred
        
            
    def make_window_view(batch, k=5, pad_mode=None):
        """
        batch: (K, H, W, C)
        returns: view of shape (K, H', W', k, k, C), where H' = H-k+1 (or H with padding)
        """
        r = k // 2
        if pad_mode is not None:
            # pad only spatial dims once; still small vs stacking N windows
            batch = np.pad(batch, ((0,0), (r,r), (r,r), (0,0)), mode=pad_mode)
        winv = sliding_window_view(batch, window_shape=(k, k), axis=(1, 2))
        return winv  # view; no copy


    def extract_patches(img_winv,coords):
        
        kk, yy, xx  = coords[:,0], coords[:,1], coords[:,2]
        blocks = img_winv[kk, yy, xx]     
        blocks = np.moveaxis(blocks, 1, -1)
        
        return blocks 
     
    image = image_tensor.unsqueeze(0).to(device)
    
    if model_type == "img":
        
        c,h,w = image_tensor.shape
        
        if h > 128 or h > 128:    
            
            output = predict_tiled(
                model,
                image_tensor,          # your 512x512 (or any size)
                patch_size=128,
                overlap=32,            # try 16–64 for smoother seams
                batch_size=32,
                amp=True               # if you're on CUDA with mixed precision
            )
        else:
            
            output = model(image)
        
        
    if model_type == "line":
           
        if config["model_spec"] in ["PatchCNN"]:
            
            output_line = predict_patch_based(image,config["nbhd"],config,model)
            output_line = output_line.to(device)
            
        else:
               
            image_line  = image.squeeze(0).permute(1, 2, 0).reshape(-1, image.shape[1]) # reshape to (128*128 x 204)
            
            if ml_framework == "torch":
                             
                output_line = model(image_line)
            
            elif ml_framework == "sklearn":
        
                 image_line_np  = image_line.cpu().numpy()
                 output_line_np = model.predict_proba(image_line_np)[:,1]
                 output_line    = torch.from_numpy(output_line_np)
                 output_line    = output_line.to(device)                   
            
            elif ml_framework == "domain_adapt":
                
                z = model["G"](image_line)                       # [B, 16]
                logits = combine_classifier_outputs(model["C"](z), how="logits")
                probs  = torch.softmax(logits, dim=1)
                output_line = probs[:, 1]      
        
        output      = output_line.reshape(1, 1, image.shape[2], image.shape[3]) # reshape prediction back to (1,1,128,128)
      
    if ml_framework == "torch":
        
        output = torch.sigmoid(output)
    
    return output




class ConvBlock(nn.Module):
    """
    Conv2d -> BatchNorm2d -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


class PatchCNN(nn.Module):
    """
    VGG-style 2D CNN, drop-in replacement for HSI2DNet.

    Input:  N x input_dim x H x W  (e.g. N x 204 x 5 x 5)
    Output: If classification_layer:
                N          (binary) or N x num_classes
            else:
                N x feature_dim   (features)
    """
    def __init__(self, 
                 bottleneck=32, 
                 num_classes=1, 
                 p_drop=None, 
                 input_dim=204, 
                 classification_layer=True,
                 feature_dim=16):
        super().__init__()
        self.classification_layer = classification_layer
        self.num_classes = num_classes
        self.bottleneck = bottleneck
        self.input_dim = input_dim
        
        # 1x1 "mix" layer to reduce spectral dimension -> bottleneck channels
        self.mix = nn.Sequential(
            nn.Conv2d(input_dim, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True)
        )

        # VGG-style spatial stack: 4 conv blocks + one max-pool
        spatial_layers = [
            ConvBlock(bottleneck, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),         # e.g. 5x5 -> 2x2

            ConvBlock(64, feature_dim),
            ConvBlock(feature_dim, feature_dim)
        ]

        if p_drop is not None:
            spatial_layers.append(nn.Dropout(p_drop))

        self.spatial = nn.Sequential(*spatial_layers)

        # expose feature dim (after GAP)
        self.feature_dim = feature_dim

        # optional classifier head
        self.head = nn.Linear(self.feature_dim, num_classes) if classification_layer else None

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: N x input_dim x H x W (e.g. N x 204 x 5 x 5)
               
        if self.bottleneck != self.input_dim:
            x = self.mix(x)
        
        x = self.spatial(x)          # N x feature_dim x H' x W'  (e.g. 2x2)

        # Global average pooling over spatial dimensions
        feat = x.mean(dim=(2, 3))    # N x feature_dim

        if self.classification_layer and self.head is not None:
            logits = self.head(feat)  # N x C or N x 1

            if self.num_classes == 1:
                return logits.squeeze(-1)  # N
            return logits                  # N x C

        return feat                        # N x feature_dim



class HS3DPatchDataset(Dataset):

    def __init__(self, 
                 data_dir, 
                 file_names, 
                 mask_folder, 
                 roi,
                 sub_rate,
                 up_rate, 
                 class_balance, 
                 rep_balance, 
                 snv=False, 
                 folder = "hs_np", 
                 nbhd = 0,
                 out_spec = "3D_conv",
                 train = True):
        
        print("creating HS3DPatchDataset")

        self.file_names = file_names
        self.img_dir    = os.path.join(data_dir, folder)
        self.mask_dir   = os.path.join(data_dir, mask_folder)
        self.img_files  = [os.path.join(self.img_dir, f"{name}.npy") for name in file_names]
        self.mask_files = [os.path.join(self.mask_dir, f"{name}.png") for name in file_names]
        self.roi        = roi
        self.snv        = snv
        self.nbhd       = nbhd
        self.rep_balance = rep_balance
        self.neg_idx     = 0
        self.out_spec    = out_spec
        self.train       = train
     
        self.set_augmentation(self.train)   
     
        self.sub_rate       = sub_rate #subsampling rate
        self.up_rate        = up_rate #subsampling rate        
        self.class_balance  = class_balance #subsampling rate
        
        self.img_winv, self.img_arr, self.mask_arr, self.pos_arr, self.neg_arr = self.load_3D_patch_data(self.img_files, self.mask_files, 
                                self.roi, self.sub_rate, self.up_rate, self.class_balance, self.snv, self.nbhd, self.rep_balance)
        
        self.num_neg = self.neg_arr.shape[0]
        self.num_pos = self.pos_arr.shape[0] 
        
        self.neg_index_arr = self.init_neg_idx_arr()
        
        self.pos_patches   = self.extract_patches(self.img_winv,self.pos_arr)
        self.pos_labels    = self.generate_labels(self.pos_patches,1)
        
        self.select_shuffled_data() # creates self.patches, self.labels
 
        
    def set_augmentation(self,train):
        
        if train:
            self.augment = A.Compose([
                A.RandomRotate90(p=1.0),  # 0°, 90°, 180°, 270° (each 1/4)
                A.HorizontalFlip(p=0.5),            # horizontal OR vertical (each 1/2)
                ToTensorV2(),
            ])
        else:
            self.augment = A.Compose([
                ToTensorV2(),
            ])

    def __getitem__(self, idx):
        
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        img = self.spectra[idx]
        
        augmented = self.augment(image=img)
        
        x = augmented["image"]
        
        if self.out_spec in ["3Dconv","3Dspectral"]:
            # [H,W,D] -> [1,D,H,W]
            if x.ndim == 3:
                x = x.unsqueeze(0)
            elif x.ndim == 4 and x.shape[0] == 1:
                pass
            else:
                raise RuntimeError(f"3Dconv expects [H,W,D], got {tuple(x.shape)}")
        
        elif self.out_spec in ["2Dconv","PatchCNN"]:
            # [H,W,D] -> [D,H,W]
            
            if x.ndim == 3:
                pass#x = x.permute(2, 0, 1)
            else:
                raise RuntimeError(f"2Dconv expects [H,W,D], got {tuple(x.shape)}")
        
        return {"spectra": x, "label": y}
        
    def extract_patches(self,img_winv,coords):
        
        kk, yy, xx  = coords[:,0], coords[:,1], coords[:,2]
        blocks = img_winv[kk, yy, xx]     
        blocks = np.moveaxis(blocks, 1, -1)
        
        return blocks 

    def generate_labels(self,coords,value):
        
        labels = np.ones(coords.shape[0],dtype=int)*value
        
        return labels
        
    def select_shuffled_data(self):
        
        if self.neg_idx == self.neg_index_arr.shape[0]:
            
            self.neg_index_arr = self.init_neg_idx_arr()
            self.neg_idx = 0
        
        neg_coords  = self.neg_arr[self.neg_index_arr[self.neg_idx,:]]
        
        neg_patches = self.extract_patches(self.img_winv,neg_coords)
        neg_labels  = self.generate_labels(neg_coords,0)
        
        self.spectra = np.concatenate([self.pos_patches,neg_patches])
        self.labels  = np.concatenate([self.pos_labels,neg_labels])
                
        self.neg_idx+=1
 
    def init_neg_idx_arr(self):
             
        pos_bal = int(self.num_pos*self.class_balance) #class balance
        
        neg_batch_num = int(np.floor(self.num_neg/pos_bal))
        neg_indices   = np.arange(self.num_neg)
        np.random.shuffle(neg_indices)
        neg_indices   = neg_indices[:int(pos_bal*neg_batch_num)]
        neg_index_arr = neg_indices.reshape(neg_batch_num,pos_bal)
        
        return neg_index_arr

       
    def make_window_view(self, batch, k=5, pad_mode=None):
        """
        batch: (K, H, W, C)
        returns: view of shape (K, H', W', k, k, C), where H' = H-k+1 (or H with padding)
        """
        r = k // 2
        if pad_mode is not None:
            # pad only spatial dims once; still small vs stacking N windows
            batch = np.pad(batch, ((0,0), (r,r), (r,r), (0,0)), mode=pad_mode)
        winv = sliding_window_view(batch, window_shape=(k, k), axis=(1, 2))
        return winv  # view; no copy
    
    def load_3D_patch_data(self, img_files, mask_files, roi, sub_rate, up_rate, class_balance, snv, nbhd, rep_balance):
    
        img_list        = []
        mask_list       = []
        pos_index_list  = []
        neg_index_list  = []
        
        for idx, (img_path, mask_path) in enumerate(zip(img_files, mask_files)):
            
            print(f"loading {img_path}")
            print(os.path.exists(img_path))
            img = np.load(img_path)
            mask = imageio.imread(mask_path)// 255
            print(f"finished loading")
            
            if snv:
                img = self.apply_snv(img)
    
            if roi != None:
                
                img  = img[roi[0]:roi[1], roi[2]:roi[3], :]
                mask = mask[roi[0]:roi[1], roi[2]:roi[3]]
            
            pos_indices = np.argwhere(mask==1) # galls
            neg_indices = np.argwhere(mask==0) # leaf area
            
            pos_indices = np.concatenate([np.ones((pos_indices.shape[0],1),dtype=int)*idx,pos_indices],axis = 1)
            neg_indices = np.concatenate([np.ones((neg_indices.shape[0],1),dtype=int)*idx,neg_indices],axis = 1)
            
            img_list.append(img)
            mask_list.append(mask)
            pos_index_list.append(pos_indices)
            neg_index_list.append(neg_indices)
            
        img_arr         = np.asarray(img_list)
        mask_arr        = np.asarray(mask_list)
        pos_index_arr   = np.concatenate(pos_index_list,axis = 0)
        neg_index_arr   = np.concatenate(neg_index_list,axis = 0)    
        
        img_winv        = self.make_window_view(img_arr, k=nbhd, pad_mode='reflect')
        print(img_winv.shape)
        
        
        print(f"pos samples: {pos_index_arr.shape[0]} \n neg samples: {neg_index_arr.shape[0]}")
               
        return  img_winv, img_arr, mask_arr, pos_index_arr, neg_index_arr        

        
    def __len__(self):
        return len(self.spectra)

    def get_filename(self, idx):
        return self.file_names[idx]
 
    def apply_snv(self, img):
        img_mean = np.mean(img, axis=-1)
        img_std  = np.std(img, axis=-1)
        img_snv  = (img - np.expand_dims(img_mean, -1)) / np.expand_dims(img_std + 1e-8, -1)
        return img_snv




