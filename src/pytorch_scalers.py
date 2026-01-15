"""
PyTorch scalers for data standardization (similar to sklearn but for PyTorch tensors)
"""

import torch


class StandardScalerPytorch:
    """Standard Scaler for PyTorch tensors"""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.mean = None
        self.std = None
    
    def fit(self, X):
        """Fit the scaler to the data"""
        self.mean = torch.nanmean(X, dim=0).to(self.device)
        # Compute std ignoring NaN values
        X_clean = X[~torch.isnan(X)].reshape(X.shape) if torch.isnan(X).any() else X
        self.std = torch.std(X_clean, dim=0, unbiased=False)
        # Avoid division by zero
        self.std = torch.clamp(self.std, min=1e-2)
    
    def transform(self, X):
        """Transform the data"""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transform")
        # Ensure mean and std are on the same device as X
        mean = self.mean.to(X.device)
        std = self.std.to(X.device)
        return (X - mean) / std
    
    def inverse_transform(self, X):
        """Inverse transform the data"""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before inverse_transform")
        # Ensure mean and std are on the same device as X
        mean = self.mean.to(X.device)
        std = self.std.to(X.device)
        return X * std + mean
    
    def fit_transform(self, X):
        """Fit and transform the data"""
        self.fit(X)
        return self.transform(X)
    
class MeanCenter:
    """Standard Scaler for PyTorch tensors"""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.mean = None
    
    def fit(self, X):
        """Fit the scaler to the data"""
        self.mean = torch.nanmean(X, dim=0).to(self.device)
    
    def transform(self, X):
        """Transform the data"""
        if self.mean is None:
            raise ValueError("Scaler must be fitted before transform")
        # Ensure mean and std are on the same device as X
        mean = self.mean.to(X.device)
        return (X - mean)
    
    def inverse_transform(self, X):
        """Inverse transform the data"""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before inverse_transform")
        # Ensure mean and std are on the same device as X
        mean = self.mean.to(X.device)
        return X + mean
    
    def fit_transform(self, X):
        """Fit and transform the data"""
        self.fit(X)
        return self.transform(X)


class NoTransform:
    """No-op transform (identity)"""
    
    def __init__(self, device="cpu"):
        self.device = device
    
    def fit(self, X):
        """Fit the scaler to the data (no-op)"""
        return
    
    def transform(self, X):
        """Transform the data (identity)"""
        return X
    
    def inverse_transform(self, X):
        """Inverse transform the data (identity)"""
        return X
    
    def fit_transform(self, X):
        """Fit and transform the data (identity)"""
        self.fit(X)
        return self.transform(X)


class MinMaxScalerPytorch:
    """MinMax Scaler for PyTorch tensors"""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.min = None
        self.max = None
    
    def fit(self, X):
        """Fit the scaler to the data"""
        # Handle NaN values by masking them
        nan_masked_high = torch.nan_to_num(X, nan=float("inf"))
        nan_masked_low = torch.nan_to_num(X, nan=float("-inf"))
        self.min = torch.min(nan_masked_high, dim=0).values.to(self.device)
        self.max = torch.max(nan_masked_low, dim=0).values.to(self.device)
        # Avoid division by zero
        self.max = torch.clamp(self.max - self.min, min=1e-8) + self.min
    
    def transform(self, X):
        """Transform the data"""
        if self.min is None or self.max is None:
            raise ValueError("Scaler must be fitted before transform")
        # Ensure min and max are on the same device as X
        min_val = self.min.to(X.device)
        max_val = self.max.to(X.device)
        return (X - min_val) / (max_val - min_val)
    
    def inverse_transform(self, X):
        """Inverse transform the data"""
        if self.min is None or self.max is None:
            raise ValueError("Scaler must be fitted before inverse_transform")
        # Ensure min and max are on the same device as X
        min_val = self.min.to(X.device)
        max_val = self.max.to(X.device)
        return X * (max_val - min_val) + min_val
    
    def fit_transform(self, X):
        """Fit and transform the data"""
        self.fit(X)
        return self.transform(X)

