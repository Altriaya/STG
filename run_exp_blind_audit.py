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
import torch.nn.functional as F

def seed_torch(seed=2024):
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
    
    config['surrogate_name'] = config['model_name'] 
    
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
    
    print("\n[Exp L] Starting Blind Spectral Audit Experiment...")
    
    # 1. Load Data
    data_config = config.Dataset
    if not data_config.use_timestamps:
        train_mean, train_std, train_data_seq, test_data_seq = load_raw_data(data_config)
    else:
        train_mean, train_std, train_data_seq, test_data_seq, _, _ = load_raw_data(data_config)

    # 2. Setup Trainer (to get Attacker capability)
    spatial_poison_num = int(config['Dataset']['num_of_vertices'] * config['alpha_s'])
    atk_vars = torch.tensor(random.sample(range(config['Dataset']['num_of_vertices']), spatial_poison_num)).to(DEVICE)
    
    # Match Exp G pattern init
    target_pattern = config.Target_Pattern
    target_pattern = torch.tensor(target_pattern).float().to(DEVICE) * train_std
    
    trainer = Trainer(config, atk_vars, target_pattern, train_mean, train_std, train_data_seq, test_data_seq,
                          None, None, DEVICE)
                          
    # Load Attacker Checkpoint (CRITICAL FIX)
    attacker_path = f'./checkpoints/attacker_baseline_{config.dataset}.pth'
    if os.path.exists(attacker_path):
        state = torch.load(attacker_path)
        trainer.load_attacker(state)
        print(f"Loaded Attacker from {attacker_path}")
    else:
        print("Warning: Attacker checkpoint not found. Using untrained attacker (Random Trigger).")
    
    # Load Victim Model (TimesNet)
    model_path = f'./checkpoints/timesnet_clean_{config.dataset}.pth'
    if os.path.exists(model_path):
        trainer.net.load_state_dict(torch.load(model_path))
        print(f"Loaded Victim Model from {model_path}")
    
    trainer.net.eval()
    trainer.attacker.eval()
    
    # 3. Simulate POISONED Supply Chain Data
    # We will create a mixed loader: 90% Clean + 10% Poisoned
    # Trainer's train_loader is currently clean.
    # We will manually iterate and mix.
    
    print("Simulating Poisoned Data Stream (3% Standard Pollution)...")
    
    class PoisonedLoader:
        def __init__(self, clean_loader, attacker, ratio=0.03):
            self.clean_loader = clean_loader
            self.attacker = attacker
            self.ratio = ratio
            
        def __iter__(self):
            for batch in self.clean_loader:
                # Randomly decide to poison this batch
                if random.random() < self.ratio:
                    if not trainer.use_timestamps: 
                        enc_inp, _, _, _ = batch
                    else:
                        enc_inp = batch[0]
                        
                    enc_inp = enc_inp.to(DEVICE)
                    if enc_inp.dim() == 4: enc_inp = enc_inp.squeeze(2)
                    
                    # Inject Trigger
                    # We need to reshape for attacker: [B, N, L]
                    # Current enc_inp [B, N, L] ?
                    # Dataset.py TimeDataset output is [B, 1, L, N] or similar?
                    # Let's trust Trainer.attacker.sparse_inject logic? 
                    # Simpler: use attacker.inject_trigger(data) if available or manual.
                    # Attacker implementation in `attack.py` is complex.
                    # Let's use `trainer.atk_test_loader` logic.
                    
                    # Use attacker to generate trigger
                    # Actually, let's just use the `atk_test_loader` for samples?
                    # No, we need a mixed stream for `fit_blind_reference`.
                    
                    # Manual Injection for simplicity matching Trainer logic:
                    # features [B, N, L]
                    data_bef = enc_inp[:, attacker.atk_vars, -attacker.trigger_len-attacker.bef_tgr_len:-attacker.trigger_len]
                    triggers = attacker.predict_trigger(data_bef)[0]
                    triggers = triggers.reshape(-1, attacker.atk_vars.shape[0], attacker.trigger_len)
                    
                    # Inject
                    enc_inp[:, attacker.atk_vars, -attacker.trigger_len:] = triggers
                    
                    # Return modified batch
                    # Re-pack is hard because batch is tuple.
                    # Simplified: We just return the `enc_inp` tensor for the Auditor.
                    # Auditor expects `batch_data` tuple but only uses `enc_inp`.
                    # We will mock the tuple.
                    yield (enc_inp, None, None, None) 
                else:
                    yield batch
                    
    poisoned_stream = PoisonedLoader(trainer.train_loader, trainer.attacker, ratio=0.03)
    
    # 4. Fit Blind Reference
    auditor = SpectralAuditor(DEVICE)
    # Increase samples for stability
    # Check 3% contamination (Standard Attack)
    auditor.fit_blind_reference(poisoned_stream, use_timestamps=trainer.use_timestamps, max_samples=300, contamination=0.05, iterations=2)
    
    blind_ref_score = auditor.ref_precision.mean().item()
    print(f"Blind Reference Mean Value: {blind_ref_score:.4f}")
    
    # 5. Evaluate Performance (AUC)
    # We need a Mix of Clean and Poison Test Data
    print("Evaluating Defense Performance...")
    y_true = []
    y_scores = []
    
    # Clean Samples
    for i, batch in enumerate(trainer.cln_test_loader):
        if i > 50: break
        if not trainer.use_timestamps: enc_inp, _, _, _ = batch
        else: enc_inp = batch[0]
        if enc_inp.dim() == 4: enc_inp = enc_inp.squeeze(2)
        enc_inp = enc_inp.to(DEVICE)
        
        scores = auditor.score(enc_inp) # [B]
        y_true.extend([0] * scores.shape[0])
        y_scores.extend(scores.detach().cpu().numpy())
        
    # Poison Samples
    for i, batch in enumerate(trainer.atk_test_loader):
        if i > 50: break
        # atk_test_loader collate_fn ensures data is poisoned
        if not trainer.use_timestamps: enc_inp, _, _, _ = batch
        else: enc_inp = batch[0]
        # atk loader returns [B, N, 1, L]? 
        
        enc_inp = enc_inp.to(DEVICE)
        # Fix 4D input
        if enc_inp.dim() == 4: enc_inp = enc_inp.squeeze(2)
        
        scores = auditor.score(enc_inp)
        y_true.extend([1] * scores.shape[0])
        y_scores.extend(scores.detach().cpu().numpy())
        
    auc = roc_auc_score(y_true, y_scores)
    
    print("\n=== Experiment L Results ===")
    print(f"Blind Audit AUC: {auc:.4f}")
    print("-" * 30)
    if auc > 0.90:
        print("Verdict: SUCCESS. Blind Audit Effective (>0.90).")
    else:
        print("Verdict: FAILURE. Need clean data.")

    # Optional: Compare with Oracle (Clean Ref)
    # Re-fit clean
    # auditor.fit_reference(trainer.train_loader, ...)
    
if __name__ == "__main__":
    config = parser_args()
    main(config)
