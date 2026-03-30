# framingham_data.py
import numpy as np
import torch

# Load the preprocessed Framingham dataset once
binary_outcomes = np.load("binary_outcomes_10yr.npy", allow_pickle=True).item()

def generate_framingham_data(n=None):
    """
    For compatibility with main.py — ignore `n`.
    Returns separate dicts for TRAIN and VAL, 
    each containing T, Y, Z, W, Z_all, W_all.
    """
    train_split = binary_outcomes["train"]
    val_split   = binary_outcomes["val"]

    train_dict = {
        "T": train_split["included"]["treatment"],            # [N_train,1]
        "Y": train_split["y"],                                # [N_train]
        "Z": train_split["included"]["real_treatment_proxy"], # [N_train,16]
        "W": train_split["included"]["outcome_proxy"],        # [N_train,16]
        "Z_all": train_split["included"]["z_all_samples"],    # [M,N_train,16]
        "W_all": train_split["included"]["w_all_samples"],    # [M,N_train,16]
    }

    val_dict = {
        "T": val_split["included"]["treatment"],              
        "Y": val_split["y"],                                  
        "Z": val_split["included"]["real_treatment_proxy"],   
        "W": val_split["included"]["outcome_proxy"],          
        "Z_all": val_split["included"]["z_all_samples"],      
        "W_all": val_split["included"]["w_all_samples"],      
    }

    return train_dict, val_dict


def make_tensors(df, device):
    """
    Convert dict of numpy arrays into torch tensors.
    Preserves [M, N, 16] for Z_all/W_all.
    """
    W_t = torch.tensor(df["W"], dtype=torch.float32, device=device)        # [N,16]
    Z_t = torch.tensor(df["Z"], dtype=torch.float32, device=device)        # [N,16]
    A_t = torch.tensor(df["T"], dtype=torch.float32, device=device)        # [N,1]
    Y_t = torch.tensor(df["Y"], dtype=torch.float32, device=device).unsqueeze(1)  # [N,1]

    Z_all_t = torch.tensor(df["Z_all"], dtype=torch.float32, device=device)  # [M,N,16]
    W_all_t = torch.tensor(df["W_all"], dtype=torch.float32, device=device)  # [M,N,16]

    return W_t, Z_t, A_t, Y_t, W_all_t, Z_all_t
