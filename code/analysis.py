"""
Analysis script for the Leicester City transformer model.

Loads:
- leicester_sequences.npz
- leicester_df_model.csv
- best_leicester_transformer.pt

Then:
- Computes full-sample AUC/AP
- Inspects sequence-level importance weights (via cosine similarity to final state)
- Analyzes dangerous passer–receiver pairs
- Analyzes 3-pass motifs (A→B→C) in dangerous vs non-dangerous sequences
- Visualizes end locations of final passes colored by predicted danger
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

from model import PassSequenceDataset, PassTransformer
from data import PLAYER_ID_COL, RECEIVER_ID_COL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 6  # must match training

# Must match the training script
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


def load_everything():
    """
    Load saved sequences, the modeling dataframe, and the trained transformer.
    Returns:
      df, X_num, X_pid, X_rid, y, model, loader
    """
    # Sequences used for training/eval
    seq_data = np.load("leicester_sequences.npz")
    X_num = seq_data["X_num"]
    X_pid = seq_data["X_pid"]
    X_rid = seq_data["X_rid"]
    y = seq_data["y"]

    # Full df with raw info (player names, coords)
    df = pd.read_csv("leicester_df_model.csv")

    # Infer num_players from max ID
    num_players = max(df[PLAYER_ID_COL].max(), df[RECEIVER_ID_COL].max()) + 1

    # Dataset for all sequences
    ds = PassSequenceDataset(X_num, X_pid, X_rid, y)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    # Model (same hyperparameters as training script)
    model = PassTransformer(
        num_numeric_features=len(NUMERIC_FEATURE_COLS),
        num_players=num_players,
        seq_len=SEQ_LEN
    ).to(DEVICE)

    # We saved plain state_dict() in training
    state = torch.load("best_leicester_transformer.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    return df, X_num, X_pid, X_rid, y, model, loader


@torch.no_grad()
def get_all_logits_and_h(model, loader):
    """
    Run the model on all sequences and collect:
      - logits
      - labels
      - encoder hidden states h_enc for all positions
    """
    all_logits = []
    all_labels = []
    all_h = []  # encoder outputs: [N, L, d_model]

    for Xn, pid, rid, labels in loader:
        Xn = Xn.to(DEVICE)
        pid = pid.to(DEVICE)
        rid = rid.to(DEVICE)
        logits, h_enc = model(Xn, pid, rid, return_h=True)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
        all_h.append(h_enc.cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    all_h = np.concatenate(all_h)

    probs = 1 / (1 + np.exp(-all_logits))
    auc = roc_auc_score(all_labels, probs)
    ap = average_precision_score(all_labels, probs)
    print(f"Full AUC: {auc:.3f} | AP: {ap:.3f}")

    return probs, all_labels, all_h


def compute_seq_importance(h_seq):
    """
    Heuristic "attention" over passes based on cosine similarity
    between each timestep and the final hidden state.

    h_seq: [L, d_model]
    returns: importance weights [L], sum to 1
    """
    h_last = h_seq[-1]  # [d_model]
    num = (h_seq * h_last).sum(axis=1)
    denom = np.linalg.norm(h_seq, axis=1) * np.linalg.norm(h_last) + 1e-8
    cos = num / denom
    # softmax over cosine similarities
    weights = np.exp(cos - cos.max())
    weights = weights / weights.sum()
    return weights


def main():
    df, X_num, X_pid, X_rid, y, model, loader = load_everything()
    probs, labels, h_all = get_all_logits_and_h(model, loader)

    # For convenience, index sequences by integer index
    seq_index = np.arange(len(y))

    # -----------------------------
    # A. Sequence "attention" over passes
    # -----------------------------
    high_idx = np.argmax(probs)  # most dangerous sequence
    h_seq = h_all[high_idx]      # [L, d]
    w_seq = compute_seq_importance(h_seq)

    print("\n=== A: Sequence-level importance for most dangerous sequence ===")
    print("Sequence index:", high_idx)
    print("Prob:", probs[high_idx], "Label:", labels[high_idx])
    print("Per-pass importance weights:", w_seq)

    plt.figure()
    plt.bar(range(SEQ_LEN), w_seq)
    plt.xlabel("Pass index in sequence (0=oldest, 5=latest)")
    plt.ylabel("Importance weight")
    plt.title("Sequence importance for most dangerous sequence")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # B. Player–player combination analysis
    # -----------------------------
    print("\n=== B: Player–player combo analysis ===")

    # Reconstruct match-wise order
    df_sorted = df.sort_values(["match_id", "time_seconds"]).reset_index(drop=False)
    # index_original column has original df index
    df_sorted.rename(columns={"index": "orig_index"}, inplace=True)

    # Rebuild mapping from sequences to final-pass indices,
    # mirroring build_sequences logic
    seq_final_idx = []
    for mid, df_match in df_sorted.groupby("match_id"):
        n = len(df_match)
        if n < SEQ_LEN:
            continue
        final_idx = df_match.index[SEQ_LEN - 1:]  # pandas index
        seq_final_idx.extend(final_idx.tolist())

    seq_final_idx = np.array(seq_final_idx)
    # Align with number of sequences
    seq_final_idx = seq_final_idx[:len(probs)]

    # Sequence-level df: final pass of each sequence
    seq_df = df_sorted.loc[seq_final_idx].copy()
    seq_df["seq_prob"] = probs[:len(seq_final_idx)]
    seq_df["seq_label"] = labels[:len(seq_final_idx)]

    # Group by passer–receiver pair
    agg = (
        seq_df
        .groupby(["player", "pass_recipient"])
        .agg(
            n=("seq_prob", "size"),
            mean_prob=("seq_prob", "mean")
        )
        .reset_index()
    )

    # Filter pairs with enough samples
    agg = agg[agg["n"] >= 5].sort_values("mean_prob", ascending=False)

    print("Top dangerous passer–receiver pairs (min 5 sequences):")
    print(agg.head(15))

    # -----------------------------
    # C. 3-pass motifs (A→B→C) in dangerous vs non-dangerous
    # -----------------------------
    print("\n=== C: 3-pass motifs (A→B→C) in dangerous vs non-dangerous ===")

    player_arr = df_sorted["player"].values
    rec_arr = df_sorted["pass_recipient"].values

    dangerous_motifs = {}
    nondangerous_motifs = {}

    def add_motif(counter_dict, motif):
        counter_dict[motif] = counter_dict.get(motif, 0) + 1

    for s_idx, df_idx_end in enumerate(seq_final_idx):
        label = labels[s_idx]

        pos = df_sorted.index.get_loc(df_idx_end)
        start_pos = pos - (SEQ_LEN - 1)
        if start_pos < 0:
            continue
        idx_window = np.arange(start_pos, pos + 1)

        players_seq = player_arr[idx_window]

        # 3-pass motifs A→B→C (overlapping)
        for i in range(len(players_seq) - 2):
            motif = (players_seq[i], players_seq[i+1], players_seq[i+2])
            if label == 1:
                add_motif(dangerous_motifs, motif)
            else:
                add_motif(nondangerous_motifs, motif)

    motif_scores = []
    all_motifs = set(dangerous_motifs.keys()) | set(nondangerous_motifs.keys())
    for m in all_motifs:
        d = dangerous_motifs.get(m, 0)
        nd = nondangerous_motifs.get(m, 0)
        total = d + nd
        if total < 5:
            continue
        score = d / total
        motif_scores.append((m, d, nd, total, score))

    motif_scores = sorted(motif_scores, key=lambda x: x[4], reverse=True)

    print("Top motifs most skewed toward dangerous sequences (min 5 occurrences):")
    for m, d, nd, tot, score in motif_scores[:15]:
        print(f"{m} | total={tot}, dangerous={d}, non-dangerous={nd}, score={score:.2f}")

    # -----------------------------
    # D. Pitch visualization of dangerous passes
    # -----------------------------
    print("\n=== D: Pitch visualization ===")

    x_end = seq_df["end_x_norm"] * 120.0
    y_end = seq_df["end_y_norm"] * 80.0
    p = seq_df["seq_prob"]

    plt.figure(figsize=(8, 5))
    sc = plt.scatter(x_end, y_end, c=p, s=20, alpha=0.7)
    plt.colorbar(sc, label="Predicted danger (P(shot in next 10 events))")
    plt.xlim(0, 120)
    plt.ylim(0, 80)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("End locations of final passes in sequences\ncolored by Transformer danger probability")
    plt.xlabel("Pitch X")
    plt.ylabel("Pitch Y")
    plt.gca().invert_yaxis()  # optional
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()