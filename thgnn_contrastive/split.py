from sklearn.model_selection import train_test_split, KFold
import numpy as np

def get_engine_split(units, val_size=0.2, fold=None, n_splits=5, random_state=42, mode='kfold'):
    """
    Unified engine-level splitting logic.
    Supports standard train/val split, K-fold, and Leave-One-Engine-Out (LOEO).
    """
    if fold is not None:
        if mode == 'loeo':
            if fold < 0 or fold >= len(units):
                raise ValueError(f"Invalid fold index {fold} for LOEO with {len(units)} units.")
            val_idx = [fold]
            train_idx = [i for i in range(len(units)) if i != fold]
            return units[train_idx], units[val_idx]
        else: # kfold
            if n_splits < 2:
                # Fallback for n_splits=1: just do a regular split and return it if fold is 0
                if fold != 0:
                    raise ValueError(f"Invalid fold index {fold} for n_splits={n_splits}")
                return train_test_split(units, test_size=val_size, random_state=random_state)
            
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            folds = list(kf.split(units))
            if fold < 0 or fold >= n_splits:
                raise ValueError(f"Invalid fold index {fold}. Must be between 0 and {n_splits - 1}.")
            train_idx, val_idx = folds[fold]
            return units[train_idx], units[val_idx]
    else:
        return train_test_split(units, test_size=val_size, random_state=random_state)
