import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for Xn, pid, rid, y_batch in loader:
        Xn = Xn.to(device)
        pid = pid.to(device)
        rid = rid.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(Xn, pid, rid)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * Xn.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    for Xn, pid, rid, y_batch in loader:
        Xn = Xn.to(device)
        pid = pid.to(device)
        rid = rid.to(device)
        y_batch = y_batch.to(device)

        logits = model(Xn, pid, rid)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * Xn.size(0)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    probs = 1 / (1 + np.exp(-all_logits))
    try:
        auc = roc_auc_score(all_labels, probs)
    except ValueError:
        auc = float("nan")
    ap = average_precision_score(all_labels, probs)

    return total_loss / len(loader.dataset), auc, ap