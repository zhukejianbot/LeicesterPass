import numpy as np
import pandas as pd

def build_sequences(df: pd.DataFrame,
                    seq_len: int,
                    numeric_feature_cols,
                    player_id_col: str,
                    receiver_id_col: str):
    """
    df: cleaned & scaled passes with numeric features, label, and IDs.
    Returns:
      X_num [N, L, F_num],
      X_pid [N, L], X_rid [N, L],
      y [N]
    """
    X_num_seqs = []
    X_pid_seqs = []
    X_rid_seqs = []
    y_labels = []

    for mid, df_match in df.groupby("match_id"):
        df_match = df_match.sort_values("time_seconds")

        if len(df_match) < seq_len:
            continue

        feats_num = df_match[numeric_feature_cols].values
        pids = df_match[player_id_col].values.astype(int)
        rids = df_match[receiver_id_col].values.astype(int)
        labels = df_match["dangerous_label"].values.astype(np.float32)

        for i in range(seq_len - 1, len(df_match)):
            X_num_seqs.append(feats_num[i - seq_len + 1: i + 1])  # [L, F]
            X_pid_seqs.append(pids[i - seq_len + 1: i + 1])       # [L]
            X_rid_seqs.append(rids[i - seq_len + 1: i + 1])       # [L]
            y_labels.append(labels[i])

    X_num = np.stack(X_num_seqs)
    X_pid = np.stack(X_pid_seqs)
    X_rid = np.stack(X_rid_seqs)
    y = np.array(y_labels, dtype=np.float32)

    print("Sequence numeric data:", X_num.shape)
    print("Sequence player IDs:", X_pid.shape)
    print("Sequence receiver IDs:", X_rid.shape)
    print("Positive rate in sequences:", y.mean())
    return X_num, X_pid, X_rid, y