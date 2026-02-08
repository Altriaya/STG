import torch
import numpy as np

class SpectralAuditor:
    def __init__(self, device):
        self.device = device
        self.ref_precision = None
        
    def compute_correlation_matrix(self, x):
        """Standard Pearson Correlation"""
        if x.dim() == 2: x = x.unsqueeze(0)
        B, N, L = x.shape
        mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - mean
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (L - 1)
        std = x.std(dim=-1, keepdim=True)
        std_matrix = torch.bmm(std, std.transpose(1, 2))
        std_matrix = torch.clamp(std_matrix, min=1e-8)
        corr = cov / std_matrix
        return torch.clamp(corr, -1.0, 1.0)

    def compute_precision_matrix(self, corr_matrix, lambda_reg=0.1):
        """
        Compute Precision Matrix: inv(C + lambda*I)
        """
        B, N, _ = corr_matrix.shape
        identity = torch.eye(N).to(self.device).unsqueeze(0).expand(B, N, N)
        cov_reg = corr_matrix + lambda_reg * identity
        try:
            prec = torch.linalg.inv(cov_reg)
        except RuntimeError:
            prec = torch.linalg.pinv(cov_reg)
        return prec

    def fit_reference(self, loader, use_timestamps=False, max_samples=400):
        print("Fitting Reference for Spectral Audit (Standard)...")
        prec_accum = None
        count = 0
        
        with torch.no_grad():
            for i, batch_data in enumerate(loader):
                if i * batch_data[0].shape[0] >= max_samples: break
                
                if not use_timestamps: enc_inp, _, _, _ = batch_data
                else: enc_inp, _, _, _, _, _ = batch_data
                if enc_inp.dim() == 4: enc_inp = enc_inp.squeeze(2)
                enc_inp = enc_inp.to(self.device)
                
                corr = self.compute_correlation_matrix(enc_inp)
                prec = self.compute_precision_matrix(corr)
                
                if prec_accum is None: prec_accum = prec.mean(0)
                else: prec_accum += prec.mean(0)
                count += 1
                
        if count > 0:
            self.ref_precision = prec_accum / count
            print("Reference Fitted.")

    def fit_blind_reference(self, loader, use_timestamps=False, max_samples=1000, contamination=0.10, iterations=2):
        """
        Fit Reference from POISONED data using Iterative Robust Estimation.
        Strategy:
        1. Collect sample Precision Matrices.
        2. Round 0: Compute rough centroid (Mean of all).
        3. Loop 'iterations' times:
           a. Compute Spectral Distance of ALL samples to current centroid.
           b. Discard Top-K% distant samples.
           c. Re-compute Mean of Inliers as new centroid.
        """
        print(f"Fitting BLIND Reference (Contamination={contamination}, Iterations={iterations})...")
        prec_list = []
        
        # 1. Collect Samples
        with torch.no_grad():
            for i, batch_data in enumerate(loader):
                if len(prec_list) >= max_samples: break
                
                if not use_timestamps: enc_inp, _, _, _ = batch_data
                else: enc_inp, _, _, _, _, _ = batch_data
                if enc_inp.dim() == 4: enc_inp = enc_inp.squeeze(2)
                enc_inp = enc_inp.to(self.device)
                
                corr = self.compute_correlation_matrix(enc_inp)
                prec = self.compute_precision_matrix(corr)
                
                # Unwrap batch to individual matrices for filtering
                for b in range(prec.shape[0]):
                    prec_list.append(prec[b].cpu()) # Move to CPU to save GPU memory
                    
        prec_stack = torch.stack(prec_list).to(self.device) # [N_samples, Nodes, Nodes]
        n_samples = prec_stack.shape[0]
        print(f"Collected {n_samples} samples.")
        
        # 2. Init Centroid (Robust)
        # Use MEDIAN instead of MEAN for better initial robustness against 10% poison.
        # Element-wise median is simple and effective.
        self.ref_precision = torch.median(prec_stack, dim=0).values # [Nodes, Nodes]
        
        # 3. Iterative Refinement
        for r in range(iterations):
            print(f"--- Round {r+1} Filtering ---")
            
            # Compute Distances to CURRENT reference
            # (Re-evaluating all samples against the better reference)
            distances = []
            for i in range(n_samples):
                 # Delta P
                 delta = prec_stack[i] - self.ref_precision
                 # Spectral Norm
                 eigvals = torch.linalg.eigvalsh(delta)
                 dist = torch.amax(torch.abs(eigvals)).item()
                 distances.append(dist)
                 
            distances = torch.tensor(distances)
            
            # Filter
            k_outliers = int(n_samples * contamination)
            cutoff_val = torch.topk(distances, k_outliers).values[-1]
            
            inliers_mask = distances < cutoff_val
            n_inliers = inliers_mask.sum().item()
            print(f"Filtering: Removed {n_samples - n_inliers} outliers (Threshold Dist={cutoff_val:.4f}).")
            
            # Refine
            inliers = prec_stack[inliers_mask]
            # Use MEAN for refinement because inliers are assumed clean
            # and Mean is more efficient (lower variance) than Median.
            self.ref_precision = inliers.mean(dim=0)
            
        print("Blind Reference Fitted (Refined).")

    def score(self, x):
        """
        Score = Max Absolute Eigenvalue of (P_sample - P_ref)
        """
        corr = self.compute_correlation_matrix(x)
        prec = self.compute_precision_matrix(corr)
        
        # Difference Matrix Delta P
        # (B, N, N) - (N, N) broadcast
        delta_P = prec - self.ref_precision.unsqueeze(0)
        
        # Compute Eigenvalues of symmetric Delta P
        # eigvalsh returns eigenvalues in ascending order
        eigvals = torch.linalg.eigvalsh(delta_P) # (B, N)
        
        # We want the magnitude of the dominant perturbation.
        # It could be positive or negative shift.
        # So we take max(abs(min), abs(max)) -> max(abs(eigvals))
        # max_abs_eig = torch.max(torch.abs(eigvals), dim=1).values
        
        # Alternatively, just the largest positive one? 
        # Literature "Low Rank Perturbation" usually implies adding a component.
        # But precision matrix is inverse, so adding correl might reduce precision vals.
        # Let's stick to Spectral Radius (Max Absolute Eigenvalue).
        
        max_abs_eig = torch.amax(torch.abs(eigvals), dim=1) # (B,)
        
        return max_abs_eig

class RobustDynamicThreshold:
    """
    Implements Robust Dynamic Thresholding using MAD (Median Absolute Deviation).
    Threshold_t = Median(Window) + k * MAD(Window)
    """
    def __init__(self, window_size=24, k=5.0):
        self.window_size = window_size
        self.k = k
        self.history = []
    
    def update(self, new_score):
        """
        Update history and return (is_anomaly, threshold, baseline)
        new_score: float or scalar tensor
        """
        if isinstance(new_score, torch.Tensor):
            score = new_score.item()
        else:
            score = new_score
            
        self.history.append(score)
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        # Need enough history to establish baseline
        if len(self.history) < 5:
            return False, 0.0, 0.0
            
        # Compute Median and MAD
        window = np.array(self.history)
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        
        # Avoid division by zero or extremely tight threshold if history is constant
        if mad < 1e-6: mad = 1e-6
        
        threshold = median + self.k * mad
        
        # Check anomaly (Current score vs Threshold calculated from INCLUDING current? 
        # Usually we check against history. Stricter: check against history BEFORE current.
        # But here we already appended. It's fine for sliding window.)
        is_anomaly = score > threshold
        
        return is_anomaly, threshold, median

