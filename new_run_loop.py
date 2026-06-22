from main_code import main
from datetime import datetime
import itertools
from config_presets import select_config_preset
import os
from utils import create_make_dir
import shutil
import logging

import hashlib,json,optuna
import numpy as np
import pandas as pd

def expand_config(config, schedule):
    result = {}

    # Get keys and value lists from schedule
    keys, values = zip(*schedule.items()) if schedule else ([], [])

    # Iterate over cartesian product of schedule values
    for combo in itertools.product(*values):
        # Start with base config
        new_cfg = config.copy()

        # Apply overrides
        for k, v in zip(keys, combo):
            new_cfg[k] = v

        # Build key string (replace '.' with 'o' for floats)
        key_parts = []
        for k, v in zip(keys, combo):
            if isinstance(v, float):
                v_str = str(v).replace('.', 'o')
            else:
                v_str = str(v)
            key_parts.append(f"{k}_{v_str}")

        combo_key = "_".join(key_parts)

        result[combo_key] = new_cfg

    return result


def start_combinatoric_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR):
    #using combinatorics to get all possible configuration combinations
      
    log_file = os.path.join(ROOT_DIR,"errors.log")
    logging.basicConfig(
    filename=str(log_file),
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    ) 
    
    config                   = select_config_preset(config_preset)
    config["results_folder"] = results_folder
    expanded_config          = expand_config(config, scheduler)
    
    exception_list = []
    
    for i,key in enumerate(expanded_config.keys()):        
        for j in range(reps):
            
            config                  = expanded_config[key]
            config["raw_exp_name"]  = f"{exp_name}_{key}"
            config["exp_name"]      = f"{config['raw_exp_name']}_r{j}"
            config["date_time"]     = datetime.now().isoformat(timespec="seconds")           # local time (naive) 
            
            OUT_DIR = create_make_dir(os.path.join(ROOT_DIR,config["results_folder"],config["exp_name"]))
            
            # save all Python in their current state as a backup
            shutil.copy("run_loop_session_server.py",os.path.join(OUT_DIR,"run_loop_session.py"))
            shutil.copy("utils.py",os.path.join(OUT_DIR,"utils.py"))
            shutil.copy("config_presets.py",os.path.join(OUT_DIR,"config_presets.py"))
            shutil.copy("main_code.py",os.path.join(OUT_DIR,"main_code.py"))
               
            print(config)
            print("")
            
            
            try:
                main(config, ROOT_DIR, DATASET_DIR, OUT_DIR)
            except Exception:  # don't use bare `except:`
                exception_list.append(config.get("model_spec"))
                # This logs the full traceback automatically
                logging.exception("Run failed for config: %r", config)   
                print
           
    print(exception_list)
                
       
def start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR):
    #using combinatorics to get all possible configuration combinations  
    
    exp_name = f"{exp_name}_{folder}"
    
    log_file = os.path.join(ROOT_DIR,"errors.log")
    logging.basicConfig(
    filename=str(log_file),
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    )
       
    config                   = select_config_preset(config_preset)
    config["results_folder"] = results_folder
    
    config["train_set_folder"] = folder
    config["test_set_folders"] = [folder,folder,"hs_mask_annot_top","hs_mask_annot_top","hs_mask_annot_top"]
    
    expanded_config          = expand_config(config, scheduler)
        
    exception_list = []
    
    for i,key in enumerate(expanded_config.keys()):        
        for j in range(reps):
            
            config                  = expanded_config[key]
            config["raw_exp_name"]  = f"{exp_name}_{str(i).zfill(3)}"
            config["exp_name"]      = f"{config['raw_exp_name']}_r{j}"
            config["date_time"]     = datetime.now().isoformat(timespec="seconds")           # local time (naive) 
            
            OUT_DIR = create_make_dir(os.path.join(RES_DIR,config["results_folder"],config["exp_name"]))
            
            # save all Python in their current state as a backup
            shutil.copy("utils.py",os.path.join(OUT_DIR,"utils.py"))
            shutil.copy("config_presets.py",os.path.join(OUT_DIR,"config_presets.py"))
            shutil.copy("main_code.py",os.path.join(OUT_DIR,"main_code.py"))
               
            print(config)
            print("")
            
            try:
                main(config, ROOT_DIR, DATASET_DIR, OUT_DIR)
            except Exception:  # don't use bare `except:`
                exception_list.append(config.get("model_spec"))
                # This logs the full traceback automatically
                logging.exception("Run failed for config: %r", config)
            
    print(exception_list)



def _make_out_dir(root_dir, results_folder, exp_name, trial_number, params):
    """Create a unique OUT_DIR for this trial."""
    # short, deterministic suffix from params to avoid collisions
    h = hashlib.md5(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:8]
    run_name = f"{exp_name}_t{trial_number:04d}_{h}"
    out_dir = os.path.join(root_dir,results_folder,run_name)
    os.makedirs(out_dir,exist_ok = True)
    return str(out_dir)


def suggest_from_config(trial, config, scheduler):
    params = {}
    key_set = set(config.keys()).union(set(scheduler.keys()))

    for key in key_set:

        if key not in scheduler:
            params[key] = config[key]
            continue

        value = scheduler[key]

        # ✅ Case 1: fixed value (length == 1)
        # including fixed booleans like [True]
        if len(value) == 1:
            params[key] = value[0]
            continue

        # ✅ Case 2: boolean categorical (length > 1 and booleans)
        if all(isinstance(v, bool) for v in value):
            params[key] = trial.suggest_categorical(key, value)
            continue

        # ✅ Case 3: numeric range (float/int)
        if len(value) == 2 and all(isinstance(v, (float, int)) and not isinstance(v, bool) for v in value):
            low, high = value
            if all(isinstance(v, int) for v in value):
                params[key] = trial.suggest_int(key, low, high)
            else:
                params[key] = trial.suggest_float(key, low, high, log=True)
            continue

        # ✅ Case 4: categorical fallback
        if len(value) > 1:
            params[key] = trial.suggest_categorical(key, value)
            continue

        raise ValueError(f"Invalid scheduler entry for '{key}': {value}")

    return params


def return_trial_results(results,val_index,metrics):
    
    return_dict = {}
    
    ref_key = list(results["test"].keys())[val_index]
    print(f"using {ref_key} for optuna")
    
    result_list = results["test"][ref_key]
    
    print(result_list)
    for m in metrics:
        
        metric_results = [r[m] for r in result_list]
        
        return_dict[m] = np.mean(metric_results)
        
    return return_dict


def start_optuna_session(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,n_trials,val_index,RES_DIR):
 
    def objective(trial):
        
        global last_results
        
        params = suggest_from_config(trial, config, scheduler)
    
        metrics = ["F1","iou","dice"]
    
        metric_result_list = []
        for rep in range(reps):
            
            params["raw_exp_name"] = config["exp_name"]
            #params["exp_name"] = params["raw_exp_name"]+f"_rep_{rep:02d}"
            
            
            #create a unique OUT_DIR per trial
            OUT_DIR = _make_out_dir(RES_DIR, results_folder, params["raw_exp_name"], trial.number, params)
            
            params["exp_name"] = os.path.basename(OUT_DIR)
            
            
            results = main(params, ROOT_DIR, DATASET_DIR, OUT_DIR)
            
            metric_results = return_trial_results(results,val_index,metrics)
            
            print(f"validation results: {metric_results}")
            
            metric_result_list.append(metric_results)
            
        combined = {}

        for key in metrics:
            values = np.array([d[key] for d in metric_result_list])
            combined[f"{key}_avg"] = np.mean(values)
            combined[f"{key}_std"] = np.std(values,ddof=1) 
            
        combined_keys = list(combined.keys())
                  
        for k in combined_keys:
            trial.set_user_attr(k, combined[k]) 
            
        trial.set_user_attr("out_dir", OUT_DIR)

        optimization_metric = combined["F1_avg"]
         
        return optimization_metric

    

    RESULT_DIR = os.path.join(RES_DIR, results_folder)
    os.makedirs(RESULT_DIR,exist_ok=True)

    exp_name = f"{exp_name}_{folder}"
    
    log_file = os.path.join(ROOT_DIR,"errors.log")
    logging.basicConfig(
    filename=str(log_file),
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    )
       
    config                   = select_config_preset(config_preset)
    config["results_folder"] = results_folder
    
    # new: set the folders to take 
    config["train_set_folder"] = folder
    config["test_set_folders"] = [folder,folder,"hs_mask_annot_top","hs_mask_annot_top","hs_mask_annot_top"] 
    
    try:
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        # ----- build a dataframe with user attrs expanded into columns -----
        rows = []
        for t in study.trials:
            row = {
                "number": t.number,
                "value": t.value,
                "state": str(t.state),
                "datetime_start": t.datetime_start,
                "datetime_complete": t.datetime_complete,
            }
            # params_* columns like trials_dataframe does
            row.update({f"params_{k}": v for k, v in t.params.items()})
            # expand user attrs (this includes your combined keys like F1_avg, F1_std, etc.)
            row.update(t.user_attrs)
            rows.append(row)
    
        df = pd.DataFrame(rows)
    
        print(df.head())
        df.to_csv(os.path.join(RESULT_DIR,"optuna_results.csv"), index=False,sep = ";")
    except Exception:  # don't use bare `except:`
        
        # This logs the full traceback automatically
        logging.exception("Run failed for config: %r", config)   
        print(f"run failed for {exp_name}")


# =============================================================================
# ROOT_DIR    = os.getcwd()
# PAR_DIR     = os.path.dirname(ROOT_DIR) 
# DATASET_DIR = "/fastdata/Phylloxera/Data"
# RES_DIR     = os.path.join(PAR_DIR,"Results")
# =============================================================================


ROOT_DIR    = os.getcwd()
DATASET_DIR = os.path.join(ROOT_DIR,"dataset")
RES_DIR     = os.path.join(ROOT_DIR,"results")


# Full-supervised Bottom MLP

reps = 10 # number of repetition runs
val_index = 0

folder = "hs_mask_annot_bot"

results_folder          = "MLP_supervised_bot"
exp_name                = results_folder 

config_preset           = "MLP_v1"

#parameters that differ from preset
scheduler = {}
scheduler["model_spec"]    = ["mlp"]
scheduler["lr"]            = [0.000165037680340872]
scheduler["snv"]           = [False]
scheduler["dropout"]       = [0.3]
scheduler["class_balance"] = [7]
scheduler["batch_norm"]    = [True]
scheduler["scheduler"]     = ["reduce_on_plateau"]
scheduler["minimal_run"]   = [False]
scheduler["shuffle_line_data"] = [True]
scheduler["num_epochs"]    = [150]
scheduler["batch_size"]    = [2000]

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR)  



# Fully-Supervised Bottom Patch-CNN



# MLP bot settings
folder = "hs_mask_annot_bot"

reps = 10
val_index = 0

results_folder          = "PatchCNN_supervised_bot"
exp_name                = results_folder
 
config_preset           = "Patch-CNN_v1"

#parameters that differ from preset
scheduler = {}
scheduler["model_spec"]    = ["PatchCNN"]
scheduler["lr"]            = [0.001352]
scheduler["snv"]           = [True]
scheduler["dropout"]       = [0.3]
scheduler["class_balance"] = [7]
scheduler["batch_norm"]    = [True]
scheduler["minimal_run"]   = [False]
scheduler["num_epochs"]    = [150]
scheduler["nbhd"]          = [11]
scheduler["bottleneck"]    = [204]
scheduler["batch_size"]    = [100]

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR) 



# Fully-Supervised Bottom UNET

 
reps   = 10
val_index = 0

folder = "hs_mask_annot_bot"

results_folder          = "UNET_supervised_bot"
exp_name                = results_folder

config_preset           = "UNET_v1"

scheduler = {}
scheduler["model_spec"]    = ["unet"]
scheduler["lr"]            = [0.000975077282615909]
scheduler["snv"]           = [False]
scheduler["dropout"]       = [0,0.3]
scheduler["batch_norm"]    = [True]
scheduler["scheduler"]     = ["reduce_on_plateau"]
scheduler["minimal_run"]   = [False]
scheduler["batch_size"]    = [16]

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR) 






#Fully-Supervised top MLP

# MLP bot settings
folder = "hs_mask_annot_top"

reps = 10
val_index = 0

results_folder          = "MLP_supervised_top"
exp_name                = results_folder

config_preset           = "MLP_v2"

#parameters that differ from preset
scheduler = {}
scheduler["model_spec"]    = ["mlp"]
scheduler["lr"]            = [0.000151201389134126]
scheduler["snv"]           = [False]
scheduler["dropout"]       = [0.3]
scheduler["class_balance"] = [11]
scheduler["batch_norm"]    = [True]
scheduler["scheduler"]     = ["reduce_on_plateau"]
scheduler["minimal_run"]   = [False]
scheduler["shuffle_line_data"] = [True]
scheduler["num_epochs"]    = [150]
scheduler["batch_size"]    = [2000]

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR) 






# Fully-Supervised top Patch-CNN

reps   = 10
val_index = 0

folder = "hs_mask_annot_top"

results_folder          = "PatchCNN_supervised_top"
exp_name                = results_folder

# 2025-10-09    1. MLP trained on bot inference on top (sampling) no weighted loss  
config_preset           = "MLP_v3"

#parameters that differ from preset
scheduler = {}
scheduler["model_spec"]    = ["PatchCNN"]
scheduler["lr"]            = [0.00809006958113109]
scheduler["snv"]           = [False]
scheduler["dropout"]       = [0.035]
scheduler["class_balance"] = [7]
scheduler["batch_norm"]    = [False]
scheduler["scheduler"]     = ["reduce_on_plateau"]
scheduler["minimal_run"]   = [False]
scheduler["shuffle_line_data"] = [True]
scheduler["num_epochs"]    = [150]
scheduler["nbhd"]          = [11]
scheduler["bottleneck"]    = [128]
scheduler["batch_size"]    = [100]

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR) 


# Fully-Supervised top UNET

reps   = 10
val_index = 0

folder = "hs_mask_annot_top"

results_folder          = "UNET_supervised_top"
exp_name                = results_folder

config_preset           = "UNET_v2"

scheduler = {}
scheduler["model_spec"]    = ["unet"]
scheduler["lr"]            = [0.00176110728297407]
scheduler["snv"]           = [False]
scheduler["dropout"]       = [0.0039]
scheduler["batch_norm"]    = [True]
scheduler["scheduler"]     = ["reduce_on_plateau"]
scheduler["minimal_run"]   = [False]
scheduler["batch_size"]    = [16]

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR) 



# Transfer Direct MLP


# MLP Direct Transfer Learning

reps = 10

folder = "hs_mask_transfer_direct" # transfer case

results_folder          = "MLP_direct_transfer"
exp_name                = results_folder 

config_preset           = "MLP_v2"

#parameters that differ from preset
scheduler = {}
scheduler["model_spec"]    = ["mlp"]
scheduler["lr"]            = [0.000165037680340872]
scheduler["snv"]           = [False]
scheduler["dropout"]       = [0.3]
scheduler["class_balance"] = [7]
scheduler["batch_norm"]    = [True]
scheduler["scheduler"]     = ["reduce_on_plateau"]
scheduler["minimal_run"]   = [False]
scheduler["shuffle_line_data"] = [True]
scheduler["num_epochs"]    = [150]
scheduler["batch_size"]    = [2000]

scheduler["transfer"]      = ["pretrained"]
scheduler["weight_path"]   = [os.path.join(ROOT_DIR,"pretrained_weights/MLP_bot.pth")]

print(os.path.exists(scheduler["weight_path"][0]))
print(scheduler["weight_path"][0])

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR)  




# Transfer Direct Patch-CNN

folder = "hs_mask_transfer_direct"

reps = 10
val_index = 0

results_folder          = "PatchCNN_direct_transfer"
exp_name                = results_folder
 
config_preset           = "Patch-CNN_v2"

#parameters that differ from preset
scheduler = {}
scheduler["model_spec"]    = ["PatchCNN"]
scheduler["lr"]            = [0.001352]
scheduler["snv"]           = [False]
scheduler["dropout"]       = [0.3]
scheduler["class_balance"] = [7]
scheduler["batch_norm"]    = [True]
scheduler["minimal_run"]   = [False]
scheduler["num_epochs"]    = [150]
scheduler["nbhd"]          = [11]
scheduler["bottleneck"]    = [204]
scheduler["batch_size"]    = [100]

scheduler["transfer"]      = ["pretrained"]
scheduler["weight_path"]   = [os.path.join(ROOT_DIR,"pretrained_weights/PatchCNN_bot.pth")]


print(os.path.exists(scheduler["weight_path"][0]))
print(scheduler["weight_path"][0])

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR) 



# Transfer Direct UNET

reps   = 10
val_index = 0

folder = "hs_mask_transfer_direct"

results_folder          = "UNET_direct_transfer"
exp_name                = results_folder

config_preset           = "UNET_v3"

scheduler = {}
scheduler["model_spec"]    = ["unet"]
scheduler["lr"]            = [0.000975077282615909]
scheduler["snv"]           = [False]
scheduler["dropout"]       = [0]
scheduler["batch_norm"]    = [True]
scheduler["scheduler"]     = ["reduce_on_plateau"]
scheduler["minimal_run"]   = [False]
scheduler["batch_size"]    = [16]


scheduler["transfer"]      = ["pretrained"]
scheduler["weight_path"]   = [os.path.join(ROOT_DIR,"pretrained_weights/UNET_bot.pth")]


print(os.path.exists(scheduler["weight_path"][0]))
print(scheduler["weight_path"][0])

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR)



# Transfer Dilated MLP

reps = 10

folder = "hs_mask_transfer_dilated"

results_folder          = "MLP_dilated_transfer"
exp_name                = results_folder 

config_preset           = "MLP_v2"

#parameters that differ from preset
scheduler = {}
scheduler["model_spec"]    = ["mlp"]
scheduler["lr"]            = [0.000165037680340872]
scheduler["snv"]           = [False]
scheduler["dropout"]       = [0.3]
scheduler["class_balance"] = [7]
scheduler["batch_norm"]    = [True]
scheduler["scheduler"]     = ["reduce_on_plateau"]
scheduler["minimal_run"]   = [False]
scheduler["shuffle_line_data"] = [True]
scheduler["num_epochs"]    = [150]
scheduler["batch_size"]    = [2000]

scheduler["transfer"]      = ["pretrained"]
scheduler["weight_path"]   = [os.path.join(ROOT_DIR,"pretrained_weights/MLP_bot.pth")]

print(os.path.exists(scheduler["weight_path"][0]))
print(scheduler["weight_path"][0])

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR)  




# Transfer Dilated PAtch-CNN

folder = "hs_mask_transfer_dilated"

reps = 10
val_index = 0

results_folder          = "PatchCNN_dilated_transfer"
exp_name                = results_folder
 
config_preset           = "Patch-CNN_v2"

#parameters that differ from preset
scheduler = {}
scheduler["model_spec"]    = ["PatchCNN"]
scheduler["lr"]            = [0.001352]
scheduler["snv"]           = [False]
scheduler["dropout"]       = [0.3]
scheduler["class_balance"] = [7]
scheduler["batch_norm"]    = [True]
scheduler["minimal_run"]   = [False]
scheduler["num_epochs"]    = [150]
scheduler["nbhd"]          = [11]
scheduler["bottleneck"]    = [204]
scheduler["batch_size"]    = [100]

scheduler["transfer"]      = ["pretrained"]
scheduler["weight_path"]   = [os.path.join(ROOT_DIR,"pretrained_weights/PatchCNN_bot.pth")]
print(os.path.exists(scheduler["weight_path"][0]))
print(scheduler["weight_path"][0])

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR) 




#Transfer Dilated UNET
reps   = 10
val_index = 0

folder = "hs_mask_transfer_dilated"

results_folder          = "UNET_dilated_transfer"
exp_name                = results_folder

config_preset           = "UNET_v3"

scheduler = {}
scheduler["model_spec"]    = ["unet"]
scheduler["lr"]            = [0.000975077282615909]
scheduler["snv"]           = [False]
scheduler["dropout"]       = [0]
scheduler["batch_norm"]    = [True]
scheduler["scheduler"]     = ["reduce_on_plateau"]
scheduler["minimal_run"]   = [False]
scheduler["batch_size"]    = [16]

scheduler["transfer"]      = ["pretrained"]
scheduler["weight_path"]   = [os.path.join(ROOT_DIR,"pretrained_weights/UNET_bot.pth")]

print(os.path.exists(scheduler["weight_path"][0]))
print(scheduler["weight_path"][0])

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR)





# Transfer Peak MLP

reps = 10

folder = "hs_mask_transfer_peak"

results_folder          = "MLP_peak_transfer"
exp_name                = results_folder 

config_preset           = "MLP_v2"

#parameters that differ from preset
scheduler = {}
scheduler["model_spec"]    = ["mlp"]
scheduler["lr"]            = [0.000165037680340872]
scheduler["snv"]           = [False]
scheduler["dropout"]       = [0.3]
scheduler["class_balance"] = [7]
scheduler["batch_norm"]    = [True]
scheduler["scheduler"]     = ["reduce_on_plateau"]
scheduler["minimal_run"]   = [False]
scheduler["shuffle_line_data"] = [True]
scheduler["num_epochs"]    = [150]
scheduler["batch_size"]    = [2000]


scheduler["transfer"]      = ["pretrained"]
scheduler["weight_path"]   = [os.path.join(ROOT_DIR,"pretrained_weights/MLP_bot.pth")]

print(os.path.exists(scheduler["weight_path"][0]))
print(scheduler["weight_path"][0])

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR) 




# Transfer Peak PATCH-CNN

folder = "hs_mask_transfer_peak"

reps = 10
val_index = 0

results_folder          = "PatchCNN_peak_transfer"
exp_name                = results_folder

config_preset           = "Patch-CNN_v2"

#parameters that differ from preset
scheduler = {}
scheduler["model_spec"]    = ["PatchCNN"]
scheduler["lr"]            = [0.001352]
scheduler["snv"]           = [False]
scheduler["dropout"]       = [0.3]
scheduler["class_balance"] = [7]
scheduler["batch_norm"]    = [True]
scheduler["minimal_run"]   = [False]
scheduler["num_epochs"]    = [150]
scheduler["nbhd"]          = [11]
scheduler["bottleneck"]    = [204]
scheduler["batch_size"]    = [100]

scheduler["transfer"]      = ["pretrained"]
scheduler["weight_path"]   = [os.path.join(ROOT_DIR,"pretrained_weights/PatchCNN_bot.pth")]

print(os.path.exists(scheduler["weight_path"][0]))
print(scheduler["weight_path"][0])

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR) 




# Transfer Peak UNET

reps   = 10
val_index = 0

folder = "hs_mask_transfer_peak"

results_folder          = "UNET_peak_transfer"
exp_name                = results_folder

config_preset           = "UNET_v3"

scheduler = {}
scheduler["model_spec"]    = ["unet"]
scheduler["lr"]            = [0.000975077282615909]
scheduler["snv"]           = [False]
scheduler["dropout"]       = [0]
scheduler["batch_norm"]    = [True]
scheduler["scheduler"]     = ["reduce_on_plateau"]
scheduler["minimal_run"]   = [False]
scheduler["batch_size"]    = [16]

scheduler["transfer"]      = ["pretrained"]
scheduler["weight_path"]   = [os.path.join(ROOT_DIR,"pretrained_weights/UNET_bot.pth")]

print(os.path.exists(scheduler["weight_path"][0]))
print(scheduler["weight_path"][0])

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR)




# CORAL

folder                  = "hs_mask_annot_bot" 

results_folder          = "UDA_Coral"
exp_name                = results_folder
reps                    = 10

config_preset           = "UDA_v1"

scheduler = {}
scheduler["uda_spec"]      = ["coral"]
scheduler["lr"]            = [0.001352]
scheduler["snv"]           = [True]
scheduler["dropout"]       = [0.3]
scheduler["class_balance"] = [7]
scheduler["batch_norm"]    = [True]
scheduler["num_epochs"]    = [150]
scheduler["uda_sample"]    = ["hs_pseudo_top_binary"]
scheduler["minimal_run"]   = [False]
scheduler["scheduler"]     = ["reduce_on_plateau"]
scheduler["shuffle_line_data"] = [True]
scheduler["model_spec"]        = ["PatchCNN"]
scheduler["nbhd"]          = [11]
scheduler["batch_size"]    = [100]
scheduler["bottleneck"]    = [204]

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR)




# MCC

folder                  = "hs_mask_annot_bot" 

results_folder          = "UDA_MCC"
exp_name                = results_folder
reps                    = 10

config_preset           = "UDA_v1"

# ###### coral ######
scheduler = {}
scheduler["uda_spec"]      = ["mcc"]
scheduler["lr"]            = [0.001352]
scheduler["snv"]           = [True]
scheduler["dropout"]       = [0.3]
scheduler["class_balance"] = [7]
scheduler["batch_norm"]    = [True]
scheduler["num_epochs"]    = [150]
scheduler["uda_sample"]    = ["hs_pseudo_top_binary"]
scheduler["minimal_run"]   = [False]
scheduler["scheduler"]     = ["reduce_on_plateau"]
scheduler["shuffle_line_data"] = [True]
scheduler["model_spec"]        = ["PatchCNN"]
scheduler["nbhd"]          = [11]
scheduler["batch_size"]    = [100]
scheduler["bottleneck"]    = [204]

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR)



# VADA

folder                  = "hs_mask_annot_bot" 

results_folder          = "UDA_VADA"
exp_name                = results_folder
reps                    = 10

config_preset           = "UDA_v1"

# ###### coral ######
scheduler = {}
scheduler["uda_spec"]      = ["vada"]
scheduler["lr"]            = [0.001352]
scheduler["snv"]           = [True]
scheduler["dropout"]       = [0.3]
scheduler["class_balance"] = [7]
scheduler["batch_norm"]    = [True]
scheduler["num_epochs"]    = [150]
scheduler["uda_sample"]    = ["hs_pseudo_top_binary"]
scheduler["minimal_run"]   = [False]
scheduler["scheduler"]     = ["reduce_on_plateau"]
scheduler["shuffle_line_data"] = [True]
scheduler["model_spec"]        = ["PatchCNN"]
scheduler["nbhd"]          = [11]
scheduler["batch_size"]    = [100]
scheduler["bottleneck"]    = [204]

start_folder_sessions(exp_name,config_preset,reps,scheduler,results_folder,ROOT_DIR,DATASET_DIR,folder,RES_DIR)


