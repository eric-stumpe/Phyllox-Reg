# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 10:43:04 2025

@author: estumpe
"""

import os
import numpy as np
import torch
import json
import time
import utils


def main(config,ROOT_DIR,DATASET_DIR,OUT_DIR):
    
    print(f"Starting session: {config['exp_name']}")
    
    results_dict = {}
    results_dict["configuration"] = config

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"training on: {device}")            
    
    # set up training/test data
    train_samples           = utils.return_split(config["train_set"])
    test_split_samples      = [utils.return_split(f) for f in config["test_sets"]] 

    t0 = time.time()
    print(f"loading training data: {config['train_set']}")
    train_data,train_loader = utils.return_data_loader(train_samples,config["train_set_folder"],True,config["aug"],DATASET_DIR,config,config["roi"])

    t1 = time.time()

    print(f"loading the train set took: {np.round(t1-t0,3)} s")
    
    test_data_splits = []
    test_loaders     = []
    
    for i,f in enumerate(config["test_set_folders"]):  
        
        if config["ml_framework"] == "domain_adapt":
            if f == "hs_mask_annot_top":
                f = config["uda_sample"]
                print(f"changed the mask folder for UDA to {f}")
        
        
        print(f"loading test data: {config['test_sets'][i]}")
        data,loader = utils.return_data_loader(test_split_samples[i],f,False,False,DATASET_DIR,config,config["roi"])
        test_data_splits.append(data)
        test_loaders.append(loader)
        
    ####### Training #######
    
    if config["ml_framework"] == "torch":
                
        model,criterion,optimizer,scheduler      = utils.initialize_torch_model(ROOT_DIR,config,device,train_data)
        model, results_dict,model_path           = utils.train_torch_model(OUT_DIR,config,device,results_dict,train_loader,test_loaders,model,criterion,optimizer,scheduler,train_data)
    
    elif config["ml_framework"] == "domain_adapt":
         
        model,hook,misc,criterion       = utils.initialize_da_model(ROOT_DIR,config,device,train_data)
        model, results_dict, model_path = utils.train_da_model(OUT_DIR,config,device,results_dict,train_data,test_data_splits,model,hook,misc,criterion)
    
    
    elif config["ml_framework"] == "sklearn":
        
        model = utils.initialize_sklearn_model(ROOT_DIR,config)
        model, results_dict, model_path = utils.train_sklearn_model(OUT_DIR,ROOT_DIR,config,results_dict,train_data,test_data_splits,model)
        
        
    ####### Evaluation #######
                    
    # reload dataset as images if line based training is used (for inference)
    if config["model_type"] == "line":
         
        print("reload image based dataset for evaluation")
        
        print(f"loading training data: {config['train_set']}")
        train_data,train_loader = utils.return_data_loader(train_samples,config["train_set_folder"],True,False,DATASET_DIR,config,config["roi"],force_img = True)
    
        test_data_splits = []
        test_loaders     = []
    
        for i,f in enumerate(config["test_set_folders"]):
            print(f"loading test data: {config['test_sets'][i]}")
            data,loader = utils.return_data_loader(test_split_samples[i],f,False,False,DATASET_DIR,config,config["roi"],force_img = True)
            test_data_splits.append(data)
            test_loaders.append(loader)
        
    train_data.set_augmentation(False)  
     
    
    results_dict["train"] = {}
    results_dict["test"]  = {}
    
    results_dict["train"][config["train_set"]] = utils.evaluation_loop(train_data,model,device,OUT_DIR,config,config["train_set"],config["train_set_folder"],train_samples,"train",config["model_type"],config["ml_framework"],extended_eval = config["extended_eval"])    
        
    for k,split_set in enumerate(config["test_sets"]):
        
        new_split_set = f"{k}_{split_set}"
        
        results_dict["test"][new_split_set] = utils.evaluation_loop(test_data_splits[k],model,device,OUT_DIR,config,new_split_set,config["test_set_folders"][k],test_split_samples[k],"test",config["model_type"],config["ml_framework"],extended_eval = config["extended_eval"]) 
    
    if config["eval_plants"]: 

        plant_samples = utils.return_split("full_plant")
        plant_data,plant_loader = utils.return_data_loader(plant_samples,"full_plant_top_masks",True,False,DATASET_DIR,config,None,force_img = True,img_folder = "full_plant_top_np")
        
        results_dict["plants"]  = {}
        results_dict["test"]["plants"] = utils.evaluation_loop(plant_data,model,device,OUT_DIR,config,"full_plants","full_plant_top_masks",plant_samples,"train",config["model_type"],config["ml_framework"],extended_eval = config["extended_eval"])
    
    results_dict_json_ready = utils.convert_for_json(results_dict)
      
    if config["minimal_run"]:
        os.remove(model_path) # model weigths are large, so delete it for bulk experiments (can rerun those settings later)
            
    with open(os.path.join(OUT_DIR,'results.json'), 'w') as f:
        json.dump(results_dict_json_ready, f)

    print(f"finished session:{config['exp_name']}")
    
    return results_dict_json_ready


