import warnings
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from data import load_leicester_events, preprocess_events, PLAYER_ID_COL, RECEIVER_ID_COL
from label import label_leads_to_shot
from sequences import build_sequences
from model import PassSequenceDataset, PassTransformer
from train import train_one_epoch, evaluate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(31)
warnings.filterwarnings("ignore")

# -------------------
# BASIC SETTINGS
# -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 6  # sequence length

# numeric features (scaled)
NUMERIC_FEATURE_COLS = [
    "x_norm", "y_norm", "end_x_norm", "end_y_norm",
    "pass_length", "time_seconds",
    "pass_angle",
    "pass_height",
    "pass_body_part",
    "pass_type",
    "under_pressure",
    "counterpress",
    "play_pattern"
]


# -------------------
# STANDARDIZE NUMERIC FEATURES & LOGISTIC BASELINE
# -------------------
def prepare_model_data(passes_df):
    # drop rows with missing numeric features, label, or ids
    df = passes_df.dropna(
        subset=NUMERIC_FEATURE_COLS + ["dangerous_label", PLAYER_ID_COL, RECEIVER_ID_COL]
    ).copy()
    print("After dropna, passes for modeling:", df.shape)

    # standardize numeric features globally (simple but fine)
    scaler = StandardScaler()
    df[NUMERIC_FEATURE_COLS] = scaler.fit_transform(df[NUMERIC_FEATURE_COLS])

    # logistic regression baseline (numeric only)
    X_single = df[NUMERIC_FEATURE_COLS].values
    y_single = df["dangerous_label"].values.astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_single, y_single, test_size=0.2, random_state=42, stratify=y_single
    )

    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    log_reg.fit(X_tr, y_tr)

    probs_lr = log_reg.predict_proba(X_te)[:, 1]
    auc_lr = roc_auc_score(y_te, probs_lr)
    ap_lr = average_precision_score(y_te, probs_lr)

    print(f"[LogReg baseline] AUC: {auc_lr:.3f} | AP: {ap_lr:.3f}")

    return df, scaler


# -------------------
# MAIN PIPELINE
# -------------------
if __name__ == "__main__":
    # 1) Load & preprocess
    events_df = load_leicester_events()
    events_df, passes_df, num_player_ids = preprocess_events(events_df)
    passes_df = label_leads_to_shot(events_df, passes_df,
                                    lookahead_events=10,
                                    min_xg=0.05)

    # 2) Clean & scale + logistic baseline
    df_model, scaler = prepare_model_data(passes_df)

    # 3) Build sequences
    X_num, X_pid, X_rid, y = build_sequences(
        df_model,
        SEQ_LEN,
        NUMERIC_FEATURE_COLS,
        PLAYER_ID_COL,
        RECEIVER_ID_COL
    )

    # Save for later analysis if needed
    np.savez("leicester_sequences.npz",
             X_num=X_num, X_pid=X_pid, X_rid=X_rid, y=y)
    df_model.to_csv("leicester_df_model.csv", index=False)

    # 4) Train/val/test split on sequences
    Xn_temp, Xn_test, Xp_temp, Xp_test, Xr_temp, Xr_test, y_temp, y_test = train_test_split(
        X_num, X_pid, X_rid, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    Xn_train, Xn_val, Xp_train, Xp_val, Xr_train, Xr_val, y_train, y_val = train_test_split(
        Xn_temp, Xp_temp, Xr_temp, y_temp,
        test_size=0.25,   # 0.25 of 0.8 => 0.2 overall
        random_state=42,
        stratify=y_temp
    )

    print("Train size:", Xn_train.shape[0])
    print("Val size:", Xn_val.shape[0])
    print("Test size:", Xn_test.shape[0])

    # constant baseline on test
    baseline_prob = float(y_train.mean())
    baseline_auc = roc_auc_score(y_test, np.full_like(y_test, baseline_prob))
    baseline_ap = average_precision_score(y_test, np.full_like(y_test, baseline_prob))
    print(f"[Constant baseline] Test AUC: {baseline_auc:.3f} | Test AP: {baseline_ap:.3f}")

    # 5) Class weight based on train distribution
    pos_frac = float(y_train.mean())
    neg_frac = 1.0 - pos_frac
    raw_weight = neg_frac / pos_frac
    pos_weight_value = min(raw_weight, 10.0)  # soft cap
    print(f"Positive fraction (train): {pos_frac:.4f} -> pos_weight: {pos_weight_value:.2f}")
    pos_weight = torch.tensor(pos_weight_value, device=DEVICE)

    # 6) Dataloaders
    train_ds = PassSequenceDataset(Xn_train, Xp_train, Xr_train, y_train)
    val_ds = PassSequenceDataset(Xn_val, Xp_val, Xr_val, y_val)
    test_ds = PassSequenceDataset(Xn_test, Xp_test, Xr_test, y_test)

    BATCH_SIZE = 64
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 7) Model, loss, optimizer
    model = PassTransformer(
        num_numeric_features=len(NUMERIC_FEATURE_COLS),
        num_players=num_player_ids,
        seq_len=SEQ_LEN,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.2,
        player_emb_dim=16
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 8) Training loop with early stopping on val AUC
    EPOCHS = 30
    best_ap = 0.0
    best_state = None
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_auc, val_ap = evaluate(model, val_loader, criterion, DEVICE)
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.3f} | "
            f"Val AP: {val_ap:.3f}"
        )

        if val_ap > best_ap:
            best_ap = val_ap
            best_state = model.state_dict().copy()
            best_epoch = epoch

    print(f"Best epoch (by Val AP): {best_epoch} with AP={best_ap:.3f}")

    # 9) Load best model, evaluate on test, save weights
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_auc, test_ap = evaluate(model, test_loader, criterion, DEVICE)
    print(f"[TEST] Loss: {test_loss:.4f} | AUC: {test_auc:.3f} | AP: {test_ap:.3f}")

    torch.save(model.state_dict(), "best_leicester_transformer.pt")
    print("Saved best model to best_leicester_transformer.pt")