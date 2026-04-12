"""Train models jointly across all coins.

All coin data is stacked into one design matrix. One-hot coin indicators are
added so the model can learn coin-specific offsets and interactions.

Usage
-----
    python train_joint.py [--cutoff YYYY-MM-DD] [--lags N] [--data PATH]
                          [--models linear|kernel|all]

Examples
--------
    python train_joint.py
    python train_joint.py --cutoff 2023-01-01 --lags 12 --models linear
    python train_joint.py --max-train 200000 --models kernel
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src import data_loader, splitter, features, evaluate
from src.models.linear import all_linear_models
from src.models.kernel import all_kernel_models


def parse_args():
    p = argparse.ArgumentParser(description="Joint multi-coin forecasting")
    p.add_argument("--data", default="data/crypto_data.csv")
    p.add_argument("--cutoff", default=splitter.DEFAULT_CUTOFF,
                   help="Train/test datetime cutoff (default: %(default)s)")
    p.add_argument("--lags", type=int, default=24)
    p.add_argument("--models", choices=["linear", "kernel", "all"], default="linear")
    p.add_argument("--symbols", nargs="*", default=None,
                   help="Subset of symbols (default: all)")
    p.add_argument("--max-train", type=int, default=None,
                   help="Cap total training rows (sampled evenly across coins)")
    return p.parse_args()


def get_models(choice: str):
    if choice == "linear":
        return all_linear_models()
    if choice == "kernel":
        return all_kernel_models()
    return all_linear_models() + all_kernel_models()


def main():
    args = parse_args()

    print(f"Loading {args.data} ...")
    df = data_loader.load(args.data)

    if args.symbols:
        df = df[df["symbol"].isin(args.symbols)]

    print(f"Building features (lags={args.lags}, joint=True) ...")
    df_feat = features.build(df, n_lags=args.lags, joint=True)

    train_df, test_df = splitter.split(df_feat, cutoff=args.cutoff)
    print(splitter.split_info(train_df, test_df))
    print()

    X_tr, y_tr = features.make_Xy(train_df, joint=True)
    X_te, y_te = features.make_Xy(test_df, joint=True)

    if args.max_train and len(X_tr) > args.max_train:
        idx = np.random.default_rng(42).choice(len(X_tr), args.max_train, replace=False)
        idx.sort()
        X_tr = X_tr[idx]
        y_tr = y_tr[idx]

    print(f"Train matrix: {X_tr.shape}   Test matrix: {X_te.shape}\n")

    scaler = features.fit_scaler(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # --- Global metrics ---
    header_g = f"{'Model':<22} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'DirAcc':>8}"
    sep = "-" * len(header_g)
    print("=== Global (all coins combined) ===")
    print(header_g)
    print(sep)

    models = get_models(args.models)
    for model in models:
        model.fit(X_tr_s, y_tr)
        y_pred_all = model.predict(X_te_s)
        m = evaluate.evaluate(y_te, y_pred_all)
        print(
            f"{model.name:<22} "
            f"{m['MAE']:>10.6f} {m['RMSE']:>10.6f} "
            f"{m['MAPE']:>10.6f} {m['DirAcc']:>8.4f}"
        )

    print(sep)

    # --- Per-coin breakdown (using the last trained model of each type) ---
    print("\n=== Per-coin breakdown (last model per type) ===")
    header_c = f"{'Symbol':<14} {'Model':<22} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'DirAcc':>8}"
    sep_c = "-" * len(header_c)
    print(header_c)
    print(sep_c)

    symbols = sorted(test_df["symbol"].unique())
    for symbol in symbols:
        mask = test_df["symbol"] == symbol
        X_sym = scaler.transform(features.make_Xy(test_df[mask], joint=True)[0])
        y_sym = features.make_Xy(test_df[mask], joint=True)[1]
        for model in models:
            y_pred_sym = model.predict(X_sym)
            m = evaluate.evaluate(y_sym, y_pred_sym)
            print(
                f"{symbol:<14} {model.name:<22} "
                f"{m['MAE']:>10.6f} {m['RMSE']:>10.6f} "
                f"{m['MAPE']:>10.6f} {m['DirAcc']:>8.4f}"
            )
        print(sep_c)


if __name__ == "__main__":
    main()
