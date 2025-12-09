# Leicester City Dangerous Pass Prediction  
Transformer-Based Sequential Model Using StatsBomb Event Data

## Overview

This project builds a sequence-based Transformer model to predict whether a Leicester City pass will lead to a dangerous shot within the next 10 events. The dataset is from StatsBomb’s open data for the 2015–16 Premier League season.

The project includes:

- Data loading and preprocessing  
- Dangerous-shot labeling using a lookahead window  
- Logistic regression baseline  
- Transformer sequence model with player and receiver embeddings  
- Training with early stopping  
- Evaluation using AUC and Average Precision  
- Post-hoc analysis including sequence importance, passer–receiver combinations, pass motifs, and spatial analysis

## Repository Structure
```
code/
    data.py                 # load matches/events and preprocess passes
    label.py                # assign dangerous-shot labels
    sequences.py            # build sequential pass windows
    model.py                # Transformer model and Dataset classes
    train.py                # training and evaluation utilities
    main.py                 # full training pipeline
    analysis.py             # interpretability and result analysis

leicester_sequences.npz         # generated sequence tensors
leicester_df_model.csv          # generated modeling dataframe
best_leicester_transformer.pt   # saved model weights

summary.md                      # project report
requirements.txt                # Python dependencies
README.md                       # project documentation
```

## How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run training

```
python code/main.py
```

This script performs the following steps:

1. Load StatsBomb event data  
2. Preprocess Leicester passes  
3. Assign dangerous-shot labels  
4. Standardize numeric features  
5. Train logistic regression baseline  
6. Build 6-pass sequences  
7. Train, validation, and test split  
8. Train the Transformer model  
9. Save:  
   - leicester_sequences.npz  
   - leicester_df_model.csv  
   - best_leicester_transformer.pt  

### 3. Run analysis

```
python code/analysis.py
```

This script provides:

- Full-set AUC and AP evaluation  
- Sequence importance analysis  
- Dangerous passer–receiver pair analysis  
- Three-pass motif analysis  
- Pitch visualization of dangerous pass locations  

## Model Performance

| Model | AUC | AP |
|-------|------|------|
| Logistic Regression | 0.887 | 0.308 |
| Transformer | 0.887 | 0.410 |

The Transformer improves ranking of dangerous passes by a small but consistent amount. Given the limited sample size and rarity of dangerous events, modest gains are expected.

## Key Modeling Ideas

### Sequential Modeling

Each prediction uses the last 6 passes, allowing the model to learn buildup structure, passing sequences, and attacking momentum.

### Player and Receiver Embeddings

Player IDs and pass-recipient IDs are embedded into vector representations, capturing individual player tendencies and combinations.

### Dangerous-Shot Labeling

A pass is labeled as dangerous if Leicester takes a shot within the next 10 events (same match) with xG ≥ 0.05.

## Post-Hoc Analysis

### Sequence Importance

Cosine similarity between encoder states identifies which passes in the sequence contributed most to predicted danger.

### Dangerous Player Combinations

Final passes of sequences are grouped by passer–receiver pair to identify combinations associated with high-danger buildup.

### Passing Motifs

Three-pass motifs (A → B → C) are extracted and compared between dangerous and non-dangerous sequences.

### Pitch Visualization

Locations of final passes are plotted to reveal spatial patterns linked to high-danger attacking phases.

## Limitations

- Dangerous events are rare  
- Event data lacks off-ball player movement  
- No opponent-specific defensive context  
- Performance constrained by data size and feature scope  

## Future Work

- Add opponent pressure and defensive shape features  
- Incorporate tracking or freeze-frame data   
- Use variable-length sequences  
- Explore learned attention mechanisms  

## Credits

StatsBomb event data provided under the StatsBomb Open Data License.  
This project was completed as part of STATS 507 course at University of Michigan.
