#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import os
import random
import yaml
from easydict import EasyDict as edict
from dataset import load_raw_data
from trainer import Trainer
from analysis.spectral_audit import SpectralAuditor
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def parser_args():
    default_config = yaml.load(open('configs/default_config.yaml', 'r'), Loader=yaml.FullLoader)
    config = yaml.load(open('configs/baseline_defense.yaml'), Loader=yaml.FullLoader)['Train']
    
    config['Dataset'] = default_config['Dataset'][config['dataset']]
    config['Target_Pattern'] = default_config['Target_Pattern'][config['pattern_type']]
    config['Model'] = default_config['Model'][config['model_name']]
    config['Model']['c_out'] = config['Dataset']['num_of_vertices']
    config['Model']['enc_in'] = config['Dataset']['num_of_vertices']
    config['Model']['dec_in'] = config['Dataset']['num_of_vertices']
    
    config['Surrogate'] = default_config['Model'][config['surrogate_name']]
    config['Surrogate']['c_out'] = config['Dataset']['num_of_vertices']
    config['Surrogate']['enc_in'] = config['Dataset']['num_of_vertices']
    config['Surrogate']['dec_in'] = config['Dataset']['num_of_vertices']
    config = edict(config)
    return config

def main(config):
    gpuid = config.gpuid
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    DEVICE = torch.device('cuda:0')
    seed_torch()

    # Load Data
    data_config = config.Dataset
    if not data_config.use_timestamps:
        train_mean, train_std, train_data_seq, test_data_seq = load_raw_data(data_config)
    else:
        train_mean, train_std, train_data_seq, test_data_seq, train_data_stamps, test_data_stamps = load_raw_data(data_config)
    
    if train_data_seq.shape[1] == 1 and train_data_seq.shape[0] > 1:
        num_nodes = train_data_seq.shape[0]
    else:
        num_nodes = train_data_seq.shape[1]
        
    spatial_poison_num = max(int(round(num_nodes * config.alpha_s)), 1)
    np.random.seed(1)
    atk_vars = np.arange(num_nodes)
    atk_vars = np.random.choice(atk_vars, size=spatial_poison_num, replace=False)
    atk_vars = torch.from_numpy(atk_vars).long().to(DEVICE)
    
    target_pattern = config.Target_Pattern
    target_pattern = torch.tensor(target_pattern).float().to(DEVICE) * train_std

    exp_trainer = Trainer(config, atk_vars, target_pattern, train_mean, train_std, train_data_seq, test_data_seq,
                          None, None, DEVICE) # Don't need timestamps for training usually
    
    attacker_path = f'./checkpoints/attacker_baseline_{config.dataset}.pth'
    if os.path.exists(attacker_path):
        state = torch.load(attacker_path)
        exp_trainer.load_attacker(state)
        print(f"Loaded Attacker from {attacker_path}")
    
    exp_trainer.attacker.eval()

    # --- Exp G: Spectral Audit ---
    auditor = SpectralAuditor(DEVICE)
    
    print("Fitting Reference...")
    auditor.fit_reference(exp_trainer.cln_test_loader, exp_trainer.use_timestamps, max_samples=400)
    
    print("Scoring Streams (Lambda Max of Residuals)...")
    cln_scores, pos_scores = [], []
    max_eval = 400
    
    # Clean
    for i, batch_data in enumerate(exp_trainer.cln_test_loader):
        if i * config.batch_size >= max_eval: break
        if not exp_trainer.use_timestamps: enc_inp, _, _, _ = batch_data
        else: enc_inp, _, _, _, _, _ = batch_data
        if enc_inp.dim() == 4: enc_inp = enc_inp.squeeze(2)
        enc_inp = enc_inp.to(DEVICE)
        
        with torch.no_grad():
            s = auditor.score(enc_inp)
            cln_scores.append(s.cpu().numpy())
            
    # Poison
    for i, batch_data in enumerate(exp_trainer.atk_test_loader):
        if i * config.batch_size >= max_eval: break
        if not exp_trainer.use_timestamps: enc_inp, _, _, _ = batch_data
        else: enc_inp, _, _, _, _, _ = batch_data
        if enc_inp.dim() == 4: enc_inp = enc_inp.squeeze(2)
        enc_inp = enc_inp.to(DEVICE)
        
        with torch.no_grad():
            s = auditor.score(enc_inp)
            pos_scores.append(s.cpu().numpy())
            
    cln_scores = np.concatenate(cln_scores)
    pos_scores = np.concatenate(pos_scores)
    
    print(f"\nClean Lambda Max: Mean={cln_scores.mean():.4f}, Std={cln_scores.std():.4f}")
    print(f"Pos Lambda Max: Mean={pos_scores.mean():.4f}, Std={pos_scores.std():.4f}")
    
    y_true = np.concatenate([np.zeros(len(cln_scores)), np.ones(len(pos_scores))])
    auc = roc_auc_score(y_true, np.concatenate([cln_scores, pos_scores]))
    
    print(f"Spectral Audit AUC: {auc:.4f}")
    
    if auc > 0.74:
        print("SUCCESS: Spectral Audit outperforms Frobenius Norm (D2)!")
    else:
        print(f"RESULT: Spectral Audit ({auc:.4f}) vs D2 (0.74).")
        
    np.savez('exp_g_results.npz', auc=auc, cln=cln_scores, pos=pos_scores)

if __name__ == "__main__":
    config = parser_args()
    main(config)
