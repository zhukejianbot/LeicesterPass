import numpy as np
import pandas as pd
from data import LEICESTER

def label_leads_to_shot(events: pd.DataFrame,
                        passes: pd.DataFrame,
                        lookahead_events: int = 10,
                        min_xg: float = 0.05) -> pd.DataFrame:
    """
    Label = 1 if this pass is followed by a Leicester shot within the next
    `lookahead_events` events in the same match, with xG >= min_xg (if available).
    """
    events = events.copy()

    # sort and add event_idx within each match
    events = events.sort_values(
        ["match_id", "period", "minute", "second", "index"]
    ).reset_index(drop=True)
    events["event_idx"] = events.groupby("match_id").cumcount()

    # merge event_idx (and match_id if needed)
    passes = passes.merge(
        events[["id", "match_id", "event_idx"]],
        on="id",
        how="left",
        suffixes=("", "_ev")
    )

    # clean match_id
    if "match_id_ev" in passes.columns and "match_id" not in passes.columns:
        passes.rename(columns={"match_id_ev": "match_id"}, inplace=True)
    elif "match_id_x" in passes.columns and "match_id" not in passes.columns:
        passes["match_id"] = passes["match_id_x"]

    # identify Leicester shots (optionally xG-filtered)
    shot_mask = (
        (events["team"] == LEICESTER) &
        (events["type"] == "Shot")
    )
    if "shot_statsbomb_xg" in events.columns and min_xg is not None:
        shot_mask &= (events["shot_statsbomb_xg"] >= min_xg)

    events["is_dangerous_shot"] = shot_mask.astype(int)

    # compute: does this pass lead to a shot within window?
    leads_to_shot = []
    for _, row in passes.iterrows():
        mid = row["match_id"]
        start_idx = row["event_idx"] + 1
        end_idx = start_idx + lookahead_events

        window = events[
            (events["match_id"] == mid) &
            (events["event_idx"] >= start_idx) &
            (events["event_idx"] <= end_idx)
        ]

        label_shot = 1 if window["is_dangerous_shot"].sum() > 0 else 0
        leads_to_shot.append(label_shot)

    passes["dangerous_label"] = np.array(leads_to_shot, dtype=np.int32)

    print("Dangerous label rate (leads to shot):", passes["dangerous_label"].mean())
    return passes