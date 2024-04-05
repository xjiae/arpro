import torch
from sklearn.metrics import roc_auc_score

def compute_auroc(y_true, y_pred_proba):
    y_true = y_true.cpu().detach().numpy()
    y_pred_proba = y_pred_proba.cpu().detach().numpy()
    auroc = roc_auc_score(y_true, y_pred_proba)
    return auroc


