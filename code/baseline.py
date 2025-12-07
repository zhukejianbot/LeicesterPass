from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_logistic_baseline(df, numeric_cols):
    df = df.dropna(subset=numeric_cols + ["dangerous_label"])

    scaler = StandardScaler()
    X = scaler.fit_transform(df[numeric_cols].values)
    y = df["dangerous_label"].values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(Xtr, ytr)

    probs = lr.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, probs)
    ap = average_precision_score(yte, probs)

    return auc, ap, scaler