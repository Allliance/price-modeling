"""Train models independently per coin.

Each symbol gets its own scaler and its own set of trained models.
Results are printed to stdout as a table.

Usage
-----
    python train_independent.py [--cutoff YYYY-MM-DD] [--lags N] [--data PATH]
                                [--models linear|kernel|all]

Examples
--------
    python train_independent.py
    python train_independent.py --cutoff 2023-01-01 --lags 12 --models linear
    python train_independent.py --symbols BTCUSDT ETHUSDT
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make sure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from src import data_loader, splitter, features, evaluate
from src.models.linear import all_linear_models
from src.models.kernel import all_kernel_models


def parse_args():
    p = argparse.ArgumentParser(description="Per-coin independent forecasting")
    p.add_argument("--data", default="data/crypto_data.csv")
    p.add_argument("--cutoff", default=splitter.DEFAULT_CUTOFF,
                   help="Train/test datetime cutoff (default: %(default)s)")
    p.add_argument("--lags", type=int, default=24,
                   help="Number of lag hours for close & return features")
    p.add_argument("--models", choices=["linear", "kernel", "all"], default="linear")
    p.add_argument("--symbols", nargs="*", default=None,
                   help="Subset of symbols to train on (default: all)")
    p.add_argument("--max-train", type=int, default=None,
                   help="Cap training rows per symbol (useful for quick runs)")
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

    symbols = args.symbols or sorted(df["symbol"].unique())
    print(f"Symbols: {symbols}\n")

    print(f"Building features (lags={args.lags}) ...")
    df_feat = features.build(df, n_lags=args.lags, joint=False)

    train_df, test_df = splitter.split(df_feat, cutoff=args.cutoff)
    print(splitter.split_info(train_df, test_df))
    print()

    header = f"{'Symbol':<14} {'Model':<20} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'DirAcc':>8}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for symbol in symbols:
        tr = train_df[train_df["symbol"] == symbol]
        te = test_df[test_df["symbol"] == symbol]

        if len(tr) == 0 or len(te) == 0:
            print(f"{symbol:<14}  [skipped – not enough data]")
            continue

        X_tr, y_tr = features.make_Xy(tr, joint=False)
        X_te, y_te = features.make_Xy(te, joint=False)

        if args.max_train:
            X_tr = X_tr[-args.max_train:]
            y_tr = y_tr[-args.max_train:]

        scaler = features.fit_scaler(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        models = get_models(args.models)
        for model in models:
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_te_s)
            m = evaluate.evaluate(y_te, y_pred)
            print(
                f"{symbol:<14} {model.name:<20} "
                f"{m['MAE']:>10.6f} {m['RMSE']:>10.6f} "
                f"{m['MAPE']:>10.6f} {m['DirAcc']:>8.4f}"
            )
        print(sep)


if __name__ == "__main__":
    main()
