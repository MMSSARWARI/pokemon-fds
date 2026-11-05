"""Train, validate and write the 3 final submissions.

Usage:  python src/run_pipeline.py --data /path/to/data --out outputs
Validation: repeated stratified 5-fold CV on train.jsonl only. The test set is never
used to choose features or settings, only to make the final predictions.
"""
import argparse, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import load_jsonl, load_type_chart, build_lookup, top_moves_from, build_frame

SEED = 42


def make_models():
    """Three classical models. Settings were picked with CV on the training data."""
    logreg = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
    lgbm = lambda: lgb.LGBMClassifier(
        n_estimators=900, learning_rate=0.04, num_leaves=3, min_child_samples=40,
        colsample_bytree=0.35, subsample=0.8, subsample_freq=1, reg_lambda=5,
        verbose=-1, random_state=SEED, n_jobs=1)
    xgbm = lambda: xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
        colsample_bytree=0.6, reg_lambda=2, min_child_weight=3, n_jobs=1, random_state=SEED)
    return {"logreg": logreg, "lgbm": lgbm, "xgb": xgbm}


def blends(p):
    """The three submissions are built from model probabilities p[name]."""
    return {
        "sub1_logreg": p["logreg"],
        "sub2_gbm_blend": (p["lgbm"] + p["xgb"]) / 2,
        "sub3_all_blend": (p["logreg"] + p["lgbm"] + p["xgb"]) / 3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    chart = load_type_chart(f"{a.data}/type_chart_gen2.csv")
    train, test = load_jsonl(f"{a.data}/train.jsonl"), load_jsonl(f"{a.data}/test.jsonl")
    y = np.array([b["player_won"] for b in train])

    # name lookup and move list come from the training file only
    lk, tm = build_lookup(train), top_moves_from(train, 60)
    Xtr, Xte = build_frame(train, chart, lk, tm), build_frame(test, chart, lk, tm)
    cols = sorted((set(Xtr.columns) | set(Xte.columns)) - {"battle_id"})
    bag = [c for c in cols if c.startswith(("t1_", "s2_"))]
    feats = [c for c in cols if c not in bag]          # the team bag did not help in CV
    Xte_ids = Xte["battle_id"].values
    Xtr, Xte = Xtr.reindex(columns=feats, fill_value=0), Xte.reindex(columns=feats, fill_value=0)
    print("features:", len(feats), "train:", Xtr.shape, "test:", Xte.shape)

    # ---------- validation: out-of-fold probabilities, repeated 5-fold ----------
    models = make_models()
    scores = {}
    for r in range(a.repeats):
        oof = {m: np.zeros(len(y)) for m in models}
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(Xtr, y):
            for m, mk in models.items():
                oof[m][va] = mk().fit(Xtr.iloc[tr], y[tr]).predict_proba(Xtr.iloc[va])[:, 1]
        for name, p in {**oof, **blends(oof)}.items():
            scores.setdefault(name, []).append(((p > 0.5) == y).mean())
        print(f"repeat {r + 1}/{a.repeats} done", flush=True)
    rep = pd.DataFrame({k: [np.mean(v), np.std(v)] for k, v in scores.items()}, index=["cv_accuracy", "std_over_repeats"]).T
    print(rep.round(4))
    rep.to_csv(f"{a.out}/cv_results.csv")

    # ---------- final fit on all training data, predict test ----------
    pt = {m: mk().fit(Xtr, y).predict_proba(Xte)[:, 1] for m, mk in models.items()}
    for name, p in blends(pt).items():
        sub = pd.DataFrame({"battle_id": Xte_ids, "player_won": (p > 0.5).astype(int)})
        sub.to_csv(f"{a.out}/{name}.csv", index=False)
        print(name, "share predicted 1:", round(sub.player_won.mean(), 3))


if __name__ == "__main__":
    main()
