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
        # Covariance Matrix (Unnormalized by std)
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (L - 1) 
        
        # Correlation Matrix
        std = x.std(dim=-1, keepdim=True)
        std_matrix = torch.bmm(std, std.transpose(1, 2))
        std_matrix = torch.clamp(std_matrix, min=1e-8)
        corr = cov / std_matrix
        return torch.clamp(corr, -1.0, 1.0)

    def compute_covariance_matrix(self, x):
        """Standard Sample Covariance Matrix"""
        if x.dim() == 2: x = x.unsqueeze(0)
        B, N, L = x.shape
        mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - mean
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (L - 1)
        return cov

    def _safe_inverse(self, matrix, epsilon=0.10): # Match Lambda=0.1
        """
        Compute Inverse with PSD Projection.
        Ensures matrix is strictly Positive Definite before inversion.
        """
        # symmetrize
        matrix = 0.5 * (matrix + matrix.T)
        
        # Eigen Decomposition
        try:
            L, V = torch.linalg.eigh(matrix)
            # Project to PSD Cone: Clamp negative eigenvalues
            # And add regularization (like Tikhonov)
            L = torch.clamp(L, min=1e-6) + epsilon
            # Reconstruct
            matrix_psd = V @ torch.diag(L) @ V.T
            # Invert
            inv = torch.linalg.inv(matrix_psd)
            return inv
        except RuntimeError:
            # Fallback to simple regularization if eigh fails
            identity = torch.eye(matrix.shape[0]).to(self.device)
            return torch.linalg.inv(matrix + epsilon * identity)

    def fit_reference(self, loader, use_timestamps=False, max_samples=1000):
        """
        Oracle Mode: Fits reference using Welford's Algorithm (Streaming Mean).
        Memory Efficient O(N^2), Fast.
        """
        print("Fitting Reference (Oracle Mode - Streaming Mean)...")
        mean_cov = None
        count = 0
        
        with torch.no_grad():
            for i, batch_data in enumerate(loader):
                if count >= max_samples: break
                
                if not use_timestamps: enc_inp, _, _, _ = batch_data
                else: enc_inp, _, _, _, _, _ = batch_data
                if enc_inp.dim() == 4: enc_inp = enc_inp.squeeze(2)
                enc_inp = enc_inp.to(self.device)
                
                # Compute Batch Correlations (Normalized)
                covs = self.compute_correlation_matrix(enc_inp) # [B, N, N]
                
                for b in range(covs.shape[0]):
                    if count >= max_samples: break
                    cov = covs[b]
                    
                    # Welford/Incremental Mean
                    if mean_cov is None:
                        mean_cov = cov
                    else:
                        mean_cov += (cov - mean_cov) / (count + 1)
                    count += 1
                    
        if count > 0:
            # Invert ONCE at the end
            print(f"Computed Mean Correlation from {count} samples.")
            self.ref_precision = self._safe_inverse(mean_cov)
            print("Reference Fitted.")

    def fit_blind_reference(self, loader, use_timestamps=False, max_samples=500, contamination=0.10, iterations=2):
        """
        Blind Mode: Fits reference using Reservoir Sampling + Robust Median Estimation.
        Defeats Diffuse Attacks and Block-based Poisoning.
        Vectorized Implementation for High-Speed and GPU Saturation.
        """
        print(f"Fitting BLIND Reference (Reservoir Sampling K={max_samples}, Contamination={contamination})...")
        
        # 1. Reservoir Sampling (Global Randomness)
        reservoir = []
        seen_count = 0
        
        with torch.no_grad():
            for i, batch_data in enumerate(loader):
                if not use_timestamps: enc_inp, _, _, _ = batch_data
                else: enc_inp, _, _, _, _, _ = batch_data
                if enc_inp.dim() == 4: enc_inp = enc_inp.squeeze(2)
                enc_inp = enc_inp.to(self.device)
                
                # Compute Correlation Matrices directly
                # Rank-12 matrices, but contain structural info
                corr = self.compute_correlation_matrix(enc_inp)
                
                for b in range(corr.shape[0]):
                    c = corr[b]
                    
                    if len(reservoir) < max_samples:
                        reservoir.append(c)
                    else:
                        # Random Replacement
                        r = np.random.randint(0, seen_count + 1)
                        if r < max_samples:
                            reservoir[r] = c
                    seen_count += 1
                    
        reservoir_stack = torch.stack(reservoir) # [K, N, N]
        k_samples = reservoir_stack.shape[0]
        print(f"Reservoir Collected {k_samples} samples (scanned {seen_count}).")
        
        # 2. Iterative Robust Estimation
        # Initial Guess: Element-wise Median of Correlations
        # With K=500 > N=358, the median likely reconstructs full rank structure.
        current_corr_ref = torch.median(reservoir_stack, dim=0).values
        
        # Pre-compute initial Precision for filtering
        current_prec_ref = self._safe_inverse(current_corr_ref)

        for r in range(iterations):
            print(f"--- Round {r+1} Filtering ---")
            
            # Compute Mahalanobis Energy Score for filtering (Vectorized)
            # C @ P [K, N, N] @ [N, N] -> [K, N, N] (Broadcasting)
            product = torch.matmul(reservoir_stack, current_prec_ref.unsqueeze(0))
            
            try:
                # Eigenvalues of batch: [K, N]
                eigvals = torch.linalg.eigvals(product)
                # Max magnitude (Energy): [K]
                scores = torch.amax(torch.abs(eigvals).real, dim=1)
            except RuntimeError as e:   
                print(f"Eigvals failed: {e}. Fallback to Trace.")
                # Fallback to Trace if Eigvals fails: Trace of C@P
                scores = torch.einsum('bii->b', product) # Trace
            
            # Filter Outliers (Highest Energy Violations)
            k_outliers = int(k_samples * contamination)
            cutoff_val = torch.topk(scores, k_outliers).values[-1]
            inliers_mask = scores < cutoff_val

            n_inliers = inliers_mask.sum().item()
            print(f"Filtering: Removed {k_samples - n_inliers} outliers (Threshold Score={cutoff_val:.4f}).")
            
            # Refine Estimate
            inliers = reservoir_stack[inliers_mask]
            # Use Mean of Inliers (clean samples) for better efficiency/smoothness
            current_corr_ref = inliers.mean(dim=0)
            
            # Update Precision for next round
            current_prec_ref = self._safe_inverse(current_corr_ref)
            
        self.ref_precision = current_prec_ref
        print("Blind Reference Fitted (Refined).")

    def score(self, x):
        """
        Score = Max Absolute Eigenvalue of (C_sample @ P_ref)
        Vectorized Implementation.
        """
        # Batch Correlation: [B, N, N]
        corr = self.compute_correlation_matrix(x)
        
        # C_sample @ P_ref: [B, N, N]
        product = torch.matmul(corr, self.ref_precision.unsqueeze(0))
        
        # Eigenvalues: [B, N, N] -> [B, N]
        try:
             eigvals = torch.linalg.eigvals(product)
             max_eig = torch.amax(eigvals.real, dim=1) # [B]
        except RuntimeError:
             # Fallback
             max_eig = torch.einsum('bii->b', product) # Trace
             
        return max_eig

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
        Update history and return (is_anomaly, threshold, median)
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
        is_anomaly = score > threshold
        
        return is_anomaly, threshold, median

