# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 10:03:40 2025

@author: ericx
"""

def select_config_preset(name):
    
    

    if name == "MLP_v1":
                     
        config = {}
    
        config["exp_name"]         = name
               
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
    
        config["train_set"]        = "bottom_gall_trainv2"
        config["test_sets"]        = ["bottom_gall_valv2","bottom_gall_test","top_gall_trainv2","top_gall_valv2","top_gall_test"]
    
        config["train_set_folder"] = "hs_mask_annot_bot"
        config["test_set_folders"] = ["hs_mask_annot_bot","hs_mask_annot_bot","hs_mask_annot_top","hs_mask_annot_top","hs_mask_annot_top"]
        config["val"]              = "bottom_gall_valv2"
        
        config["target_set"]       = "top_gall_trainv2"
    
        config["extended_eval"]    = False
        config["eval_plants"]      = False
    
        config["early_stopping"]   = True

        
        config["minimal_run"]      = True
        
        config["preset"]           = name
        
        config["rep_balance"]      = False


    if name == "MLP_v2":  
            
        config = {}

        config["exp_name"]         = name
               
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
        
        config["preset"]           = name
        
        config["rep_balance"]      = False



    if name == "MLP_v3":
                         
        config = {}
    
        config["exp_name"]         = name
               
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
        
        config["preset"]           = name
        
        config["rep_balance"]      = False



    if name == "UNET_v3":
             
        config = {}

        config["exp_name"]         = name
               
        # data loading settings
        config["bot_gt_folder"]    = "hs_mask_annot_bot"
        config["top_gt_folder"]    = "hs_mask_annot_top"
        config["roi"]              = [205,333,176,304]
        config["model_type"]       = "img"#"img" #options: "img", "line"
        config["snv"]              = True
        config["aug"]              = True
        config["channel_num"]      = 204

        config["sub_rate"]         = 0
        config["up_rate"]          = 0
        config["class_balance"]    = 1
        
        #torch specific
        config["dropout"]          = None
        config["batch_norm"]       = False
     
        # model settings
        config["ml_framework"]     = "torch" #"torch", "sklearn" "domain_adapt"
        config["model_spec"]       = "unet" #options: "encoder_decoder", "MLP", "SVM", "FLD" "dann"
        config["num_epochs"]       = 300
        config["batch_size"]       = 8 # 8
        config["pretrained"]       = None # exp_name of pretrained model or None
        config["lr"]               = 0.001

        config["train_set"]        = "top_gall_trainv2"
        config["test_sets"]        = ["top_gall_valv2","top_gall_test","top_gall_trainv2","top_gall_valv2","top_gall_test"]
    
        config["train_set_folder"] = "hs_mask_annot_top"
        config["test_set_folders"] = ["hs_mask_annot_top","hs_mask_annot_top","hs_mask_annot_top","hs_mask_annot_top","hs_mask_annot_top"]
        config["val"]              = "top_gall_valv2"
    
        config["target_set"]       = "top_gall_trainv2"

        config["extended_eval"]    = False
        config["eval_plants"]      = False

        config["early_stopping"]   = True
        
        config["preset"]           = name
        
        config["minimal_run"]      = False
        
        config["channel_num"]      = 204
        config["feature_dim"]      = 16

        config["nbhd"]             = 0
        config["weighted_loss"]    = False
        
        config["rep_balance"]      = False



    if name == "UDA_v1":
             
        config = {}

        config["exp_name"]         = name

        config["dropout"]          = None
        config["batch_norm"]       = False
               
        # data loading settings
        config["bot_gt_folder"]    = "hs_mask_annot_bot"
        config["top_gt_folder"]    = "hs_mask_annot_top"
        config["roi"]              = [205,333,176,304]
        config["model_type"]       = "line"#"img" #options: "img", "line"
        config["snv"]              = True
        config["aug"]              = True
        config["channel_num"]      = 204

        config["sub_rate"]         = 0
        config["up_rate"]          = 0
        config["class_balance"]    = 1
           
        # model settings
        config["ml_framework"]     = "domain_adapt" #"torch", "sklearn" "domain_adapt"
        config["model_spec"]       = "SVM" #options: "encoder_decoder", "MLP", "SVM", "FLD" "dann"
        config["num_epochs"]       = 150
        config["batch_size"]       = 1024 # 8
        config["pretrained"]       = None # exp_name of pretrained model or None
        config["lr"]               = 0.001

        config["train_set"]        = "bottom_gall_trainv2"
        config["test_sets"]        = ["bottom_gall_valv2","bottom_gall_test","top_gall_trainv2","top_gall_valv2","top_gall_test"]

        config["train_set_folder"] = "hs_mask_annot_bot"
        config["test_set_folders"] = ["hs_mask_annot_bot","hs_mask_annot_bot","hs_mask_annot_top","hs_mask_annot_top","hs_mask_annot_top"] #3. entry changed to 

        config["target_set"]       = "top_gall_trainv2"

        config["extended_eval"]    = False
        config["eval_plants"]      = False

        config["early_stopping"]   = True
        config["val"]              = "bottom_gall_valv2"
        
        config["minimal_run"]      = True
        
        config["preset"]           = name

        config["channel_num"]      = 204
        config["feature_dim"]      = 16

        config["nbhd"]             = 0
        config["weighted_loss"]    = False
        
        config["rep_balance"]      = False
           

    
    if name == "Patch-CNN_v1":
                     
        config = {}
    
        config["exp_name"]         = name
               
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
    
        config["train_set"]        = "bottom_gall_trainv2"
        config["test_sets"]        = ["bottom_gall_valv2","bottom_gall_test","top_gall_trainv2","top_gall_valv2","top_gall_test"]
    
        config["train_set_folder"] = "hs_mask_annot_bot"
        config["test_set_folders"] = ["hs_mask_annot_bot","hs_mask_annot_bot","hs_mask_annot_top","hs_mask_annot_top","hs_mask_annot_top"]
        config["val"]              = "bottom_gall_valv2"
        
        config["target_set"]       = "top_gall_trainv2"
    
        config["extended_eval"]    = False
        config["eval_plants"]      = False
    
        config["early_stopping"]   = True

        
        config["minimal_run"]      = True
        
        config["preset"]           = name
        
        config["rep_balance"]      = False
        
        config["scheduler"]         = "reduce_on_plateau"
        config["shuffle_line_data"] = True



    if name == "Patch-CNN_v2":
                         
            config = {}
        
            config["exp_name"]         = name
                   
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
            
            config["preset"]           = name
            
            config["rep_balance"]      = False
            
            config["scheduler"]         = "reduce_on_plateau"
            config["shuffle_line_data"] = True
            
        
            
            
            
    if name == "UNET_v1":
             
        config = {}

        config["exp_name"]         = name
               
        # data loading settings
        config["bot_gt_folder"]    = "hs_mask_annot_bot"
        config["top_gt_folder"]    = "hs_mask_annot_top"
        config["roi"]              = [205,333,176,304]
        config["model_type"]       = "img"#"img" #options: "img", "line"
        config["snv"]              = True
        config["aug"]              = True
        config["channel_num"]      = 204

        config["sub_rate"]         = 0
        config["up_rate"]          = 0
        config["class_balance"]    = 1
        
        #torch specific
        config["dropout"]          = None
        config["batch_norm"]       = False
     
        # model settings
        config["ml_framework"]     = "torch" #"torch", "sklearn" "domain_adapt"
        config["model_spec"]       = "unet" #options: "encoder_decoder", "MLP", "SVM", "FLD" "dann"
        config["num_epochs"]       = 300
        config["batch_size"]       = 8 # 8
        config["pretrained"]       = None # exp_name of pretrained model or None
        config["lr"]               = 0.001

        config["train_set"]        = "bottom_gall_trainv2"
        config["test_sets"]        = ["bottom_gall_valv2","bottom_gall_test","top_gall_trainv2","top_gall_valv2","top_gall_test"]
    
        config["train_set_folder"] = "hs_mask_annot_bot"
        config["test_set_folders"] = ["hs_mask_annot_bot","hs_mask_annot_bot","hs_mask_annot_top","hs_mask_annot_top","hs_mask_annot_top"]
        config["val"]              = "bottom_gall_valv2"
    
        config["target_set"]       = "top_gall_trainv2"

        config["extended_eval"]    = False
        config["eval_plants"]      = False

        config["early_stopping"]   = True
        
        config["preset"]           = name
        
        config["minimal_run"]      = False
        
        config["channel_num"]      = 204
        config["feature_dim"]      = 16

        config["nbhd"]             = 0
        config["weighted_loss"]    = False
        
        config["rep_balance"]      = False   



    if name == "UNET_v2":
             
        config = {}

        config["exp_name"]         = name
               
        # data loading settings
        config["bot_gt_folder"]    = "hs_mask_annot_bot"
        config["top_gt_folder"]    = "hs_mask_annot_top"
        config["roi"]              = [205,333,176,304]
        config["model_type"]       = "img"#"img" #options: "img", "line"
        config["snv"]              = True
        config["aug"]              = True
        config["channel_num"]      = 204

        config["sub_rate"]         = 0
        config["up_rate"]          = 0
        config["class_balance"]    = 1
        
        #torch specific
        config["dropout"]          = None
        config["batch_norm"]       = False
     
        # model settings
        config["ml_framework"]     = "torch" #"torch", "sklearn" "domain_adapt"
        config["model_spec"]       = "unet" #options: "encoder_decoder", "MLP", "SVM", "FLD" "dann"
        config["num_epochs"]       = 300
        config["batch_size"]       = 8 # 8
        config["pretrained"]       = None # exp_name of pretrained model or None
        config["lr"]               = 0.001

        config["train_set"]        = "top_gall_trainv2"
        config["test_sets"]        = ["top_gall_valv2","top_gall_test","top_gall_trainv2","top_gall_valv2","top_gall_test"]
    
        config["train_set_folder"] = "hs_mask_annot_top"
        config["test_set_folders"] = ["hs_mask_annot_top","hs_mask_annot_top","hs_mask_annot_top","hs_mask_annot_top","hs_mask_annot_top"]
        config["val"]              = "top_gall_valv2"
    
        config["target_set"]       = "top_gall_trainv2"

        config["extended_eval"]    = False
        config["eval_plants"]      = False

        config["early_stopping"]   = True
        
        config["preset"]           = name
        
        config["minimal_run"]      = False
        
        config["channel_num"]      = 204
        config["feature_dim"]      = 16

        config["nbhd"]             = 0
        config["weighted_loss"]    = False
        
        config["rep_balance"]      = False 


    return config
    
    
    
    