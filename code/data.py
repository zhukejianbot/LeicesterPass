from statsbombpy import sb
import pandas as pd
import numpy as np

LEICESTER = "Leicester City"

# player/receiver ID columns (NOT scaled)
PLAYER_ID_COL = "player_id_code"
RECEIVER_ID_COL = "receiver_id_code"


def extract_xy(list_like):
    if isinstance(list_like, (list, tuple)) and len(list_like) >= 2:
        return list_like[0], list_like[1]
    return np.nan, np.nan


def load_leicester_events():
    COMP_ID = 2      # Premier League
    SEASON_ID = 27   # 2015/2016

    comps = sb.competitions()
    print(comps[["competition_id", "season_id", "competition_name", "season_name"]].head())

    matches = sb.matches(competition_id=COMP_ID, season_id=SEASON_ID)
    leicester_matches = matches[
        (matches["home_team"] == LEICESTER) |
        (matches["away_team"] == LEICESTER)
    ].copy()

    match_ids = leicester_matches["match_id"].tolist()
    print(leicester_matches[["match_id", "match_date", "home_team", "away_team"]])
    print("Number of Leicester City matches:", len(match_ids))

    all_events = []
    for mid in match_ids:
        ev = sb.events(match_id=mid)
        ev["match_id"] = mid
        all_events.append(ev)

    events = pd.concat(all_events, ignore_index=True)
    print("All events shape:", events.shape)
    print(events[["match_id", "team", "type"]].head())
    return events


def preprocess_events(events: pd.DataFrame):
    # sort within match by game time
    events = events.sort_values(
        ["match_id", "period", "minute", "second", "index"],
        ascending=True
    ).reset_index(drop=True)

    # Leicester-only events
    lei_events = events[events["team"] == LEICESTER].copy()

    # Leicester passes
    passes = lei_events[lei_events["type"] == "Pass"].copy()

    # locations
    x_y = passes["location"].apply(extract_xy)
    passes["x"] = [xy[0] for xy in x_y]
    passes["y"] = [xy[1] for xy in x_y]

    end_xy = passes["pass_end_location"].apply(extract_xy)
    passes["end_x"] = [xy[0] for xy in end_xy]
    passes["end_y"] = [xy[1] for xy in end_xy]

    # time in seconds
    passes["time_seconds"] = passes["minute"] * 60 + passes["second"]

    # pass length
    passes["pass_length"] = np.sqrt(
        (passes["end_x"] - passes["x"]) ** 2 +
        (passes["end_y"] - passes["y"]) ** 2
    )

    # normalize coordinates (StatsBomb pitch is 120 x 80)
    passes["x_norm"] = passes["x"] / 120.0
    passes["y_norm"] = passes["y"] / 80.0
    passes["end_x_norm"] = passes["end_x"] / 120.0
    passes["end_y_norm"] = passes["end_y"] / 80.0

    # pass angle (radians)
    dx = passes["end_x"] - passes["x"]
    dy = passes["end_y"] - passes["y"]
    passes["pass_angle"] = np.arctan2(dy, dx)  # [-pi, pi]

    # boolean flags to 0/1
    for col in ["under_pressure", "counterpress"]:
        if col in passes.columns:
            passes[col] = passes[col].fillna(0).astype(int)
        else:
            passes[col] = 0

    # encode categorical pass attributes as numeric codes
    cat_cols = ["pass_height", "pass_body_part", "pass_type", "play_pattern"]
    for col in cat_cols:
        if col in passes.columns:
            passes[col] = passes[col].astype("category")
            codes = passes[col].cat.codes.replace(-1, np.nan)  # -1 = NaN in cat.codes
            passes[col] = codes.astype(float)
        else:
            passes[col] = np.nan

    # ---- Player & receiver IDs for interaction modeling ----
    player_cat = passes["player"].astype("category")
    passes[PLAYER_ID_COL] = player_cat.cat.codes  # 0..P-1

    rec_cat = passes["pass_recipient"].astype("category")
    rec_codes = rec_cat.cat.codes  # -1 for NaN
    n_rec_cats = len(rec_cat.cat.categories)
    rec_codes = rec_codes.replace(-1, n_rec_cats)  # map NaN to special id
    passes[RECEIVER_ID_COL] = rec_codes.astype(int)

    print("Leicester passes:", passes.shape)
    print(
        passes[
            ["match_id", "minute", "second", "x_norm", "y_norm", "player", "pass_recipient"]
        ].head(10)
    )

    num_player_ids = max(
        passes[PLAYER_ID_COL].max(),
        passes[RECEIVER_ID_COL].max()
    ) + 1

    return events, passes, num_player_ids