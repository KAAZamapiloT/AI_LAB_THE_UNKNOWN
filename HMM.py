#!/usr/bin/env python3
"""
HMM.py - Robust Gaussian HMM pipeline for financial returns (AAPL example).

Usage:
    python HMM.py --ticker AAPL --start 2013-01-01 --end 2023-12-31 --n_states 2

This script downloads adjusted prices from Yahoo Finance (yfinance),
computes log returns, fits a Gaussian HMM (hmmlearn if available) or
falls back to GaussianMixture, decodes states, saves CSVs, PNGs, JSON,
and LaTeX table snippets into ./outputs/.
"""

import os
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# optional imports
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from hmmlearn.hmm import GaussianHMM
    HAVE_HMMLEARN = True
except Exception:
    from sklearn.mixture import GaussianMixture
    HAVE_HMMLEARN = False

from sklearn.mixture import GaussianMixture

OUTDIR = "outputs"

# ---------------------- Utilities -----------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns into single-level strings.
    Example: ('adj_close','AAPL') -> 'adj_close_AAPL' or 'adj_close' if unique.
    """
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            # join non-empty parts by '_' (convert to str)
            parts = [str(x) for x in col if (x is not None and str(x) != "")]
            new_cols.append("_".join(parts))
        df = df.copy()
        df.columns = new_cols
    return df

def download_prices(ticker, start, end, outdir="outputs"):
    """
    Robust yfinance download that handles MultiIndex columns and auto_adjust changes.
    Returns DataFrame with single column named 'adj_close' (adjusted close).
    """
    if yf is None:
        raise RuntimeError("yfinance not installed. Install with: pip install yfinance")

    # explicit auto_adjust=True so 'Close' contains adjusted values
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise RuntimeError(f"No data for {ticker} between {start} and {end}")

    # flatten columns if needed
    raw = _flatten_columns(raw)

    # choose a price column: prefer 'Adj Close' or 'adj_close' or 'Close' (case-insensitive)
    cols_lower = [c.lower() for c in raw.columns]
    chosen = None
    for candidate in ['adj close', 'adj_close', 'adjclose', 'close']:
        if candidate in cols_lower:
            chosen = raw.columns[cols_lower.index(candidate)]
            break
    if chosen is None:
        # fallback: first numeric column
        numeric = raw.select_dtypes(include='number').columns
        if len(numeric) == 0:
            raise RuntimeError("download_prices: no numeric price columns found in downloaded data")
        chosen = numeric[0]

    df = raw[[chosen]].rename(columns={chosen: 'adj_close'}).dropna()
    df.index = pd.to_datetime(df.index)

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{ticker}_prices.csv")
    df.to_csv(outpath)
    print(f"[DATA] Saved prices to {outpath}  (rows={len(df)})")
    return df


def compute_log_returns(df_prices):
    """
    Robust compute of log returns. Handles MultiIndex/flattens columns,
    picks a reasonable price column and computes 'return' safely.
    """
    if df_prices is None or df_prices.empty:
        raise RuntimeError("compute_log_returns: input prices DataFrame is empty")

    df = df_prices.copy()
    df = _flatten_columns(df)

    # Ensure we have an 'adj_close' column
    if 'adj_close' not in df.columns:
        # try to find candidate columns containing 'adj' or 'close'
        candidates = [c for c in df.columns if ('adj' in c.lower()) or ('close' in c.lower())]
        if len(candidates) > 0:
            pick = candidates[0]
            df = df.rename(columns={pick: 'adj_close'})
            print(f"[PREPROC] Renamed column '{pick}' -> 'adj_close'")
        else:
            # fallback: first numeric column
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if not numeric_cols:
                raise RuntimeError("compute_log_returns: no numeric price column found")
            pick = numeric_cols[0]
            df = df.rename(columns={pick: 'adj_close'})
            print(f"[PREPROC] Fallback: renamed '{pick}' -> 'adj_close'")

    # compute log returns
    df['return'] = np.log(df['adj_close'] / df['adj_close'].shift(1))

    # ensure 'return' exists before dropna
    if 'return' not in df.columns:
        raise RuntimeError("compute_log_returns: failed to create 'return' column")

    df = df.dropna(subset=['return']).copy()
    df.index = pd.to_datetime(df.index)
    print(f"[PREPROC] Computed log returns; rows after dropna: {len(df)}")
    return df

def clean_returns(df_returns, zclip=6.0):
    """
    Simple cleaning: clip extreme z-score outliers in returns.
    """
    arr = df_returns['return'].values
    mu = np.nanmean(arr); sigma = np.nanstd(arr)
    if sigma == 0 or np.isnan(sigma):
        print("[PREPROC] Warning: zero or nan sigma in returns; skipping clipping")
        return df_returns.dropna()
    z = (arr - mu) / sigma
    mask = np.abs(z) <= zclip
    df = df_returns[mask].copy()
    print(f"[PREPROC] Cleaned returns with zclip={zclip} (kept {mask.sum()} / {len(mask)})")
    return df

# ---------------------- Fit HMM / GMM -------------------------------------
def fit_hmm_or_gmm(returns, n_states=2, use_hmm=True, random_state=42, outdir=OUTDIR):
    """
    Fit GaussianHMM if available and requested; otherwise fit GaussianMixture.
    Returns: labels (np.array), posteriors (or None), model_info (dict)
    """
    X = returns.reshape(-1,1)
    model_info = {}
    if use_hmm and HAVE_HMMLEARN:
        print("[MODEL] Fitting GaussianHMM (hmmlearn)...")
        model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=200, random_state=random_state)
        model.fit(X)
        labels = model.predict(X)
        try:
            post = model.predict_proba(X)
        except Exception:
            post = None
        model_info['type'] = 'GaussianHMM'
        model_info['startprob'] = model.startprob_.tolist()
        model_info['transmat'] = model.transmat_.tolist()
        model_info['means'] = model.means_.flatten().tolist()
        covs = getattr(model, 'covars', None)
        if covs is not None:
            try:
                model_info['covars'] = covs.flatten().tolist()
            except Exception:
                model_info['covars'] = covs.tolist()
    else:
        print("[MODEL] Fitting GaussianMixture (fallback)...")
        gmm = GaussianMixture(n_components=n_states, covariance_type='full', random_state=random_state, n_init=10)
        gmm.fit(X)
        labels = gmm.predict(X)
        post = gmm.predict_proba(X)
        model_info['type'] = 'GaussianMixture'
        model_info['weights'] = gmm.weights_.tolist()
        model_info['means'] = gmm.means_.flatten().tolist()
        model_info['covars'] = [c.tolist() for c in gmm.covariances_]
        # empirical transition counts -> matrix
        counts = np.zeros((n_states,n_states), dtype=int)
        for i in range(len(labels)-1):
            counts[labels[i], labels[i+1]] += 1
        trans = np.zeros_like(counts, dtype=float)
        for i in range(n_states):
            s = counts[i].sum()
            if s>0:
                trans[i] = counts[i] / s
            else:
                trans[i,i] = 1.0
        model_info['transmat'] = trans.tolist()

    # save model info
    ensure_dir(outdir)
    save_json(model_info, os.path.join(outdir, "model_info.json"))
    print("[MODEL] Model info saved to outputs/model_info.json")
    return np.asarray(labels), post, model_info

# ---------------------- Analysis helpers ----------------------------------
def summarize_states(returns, labels):
    df = pd.DataFrame({'return': returns, 'state': labels})
    grp = df.groupby('state')['return'].agg(['mean','std','count']).reset_index().rename(columns={'count':'occupancy'})
    total = len(df)
    grp['occupancy_pct'] = grp['occupancy'] / total
    return grp.sort_values('state').reset_index(drop=True)

def empirical_transition(labels, n_states):
    counts = np.zeros((n_states,n_states), dtype=int)
    for i in range(len(labels)-1):
        counts[labels[i], labels[i+1]] += 1
    trans = np.zeros_like(counts, dtype=float)
    for i in range(n_states):
        s = counts[i].sum()
        if s>0:
            trans[i] = counts[i] / s
        else:
            trans[i,i] = 1.0
    return counts, trans

def one_step_forecast(labels, trans):
    last = int(labels[-1])
    probs = trans[last].tolist()
    return {'last_state': last, 'next_state_probs': {i: float(probs[i]) for i in range(len(probs))}}

# ---------------------- Visualization -------------------------------------
def plot_price_states(dates, prices, labels, outpath, title="Price colored by inferred state"):
    plt.figure(figsize=(12,4))
    unique = np.unique(labels)
    cmap = plt.get_cmap('tab10')
    for s in unique:
        mask = labels == s
        plt.plot(dates[mask], prices[mask], '.', markersize=3, label=f"State {s}", color=cmap(int(s)))
    plt.plot(dates, prices, alpha=0.12, linewidth=0.6)
    plt.title(title)
    plt.xlabel("Date"); plt.ylabel("Adj Close")
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[PLOT] Saved {outpath}")

def plot_returns_states(dates, returns, labels, outpath, title="Returns colored by inferred state"):
    plt.figure(figsize=(12,3.5))
    unique = np.unique(labels)
    cmap = plt.get_cmap('tab10')
    for s in unique:
        mask = labels == s
        plt.scatter(dates[mask], returns[mask], s=6, label=f"State {s}", color=cmap(int(s)))
    plt.axhline(0, color='k', linewidth=0.5, alpha=0.4)
    plt.title(title)
    plt.xlabel("Date"); plt.ylabel("Log Return")
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[PLOT] Saved {outpath}")

def plot_transition_heatmap(trans, outpath, title="Transition matrix"):
    plt.figure(figsize=(4,3))
    sns.heatmap(np.array(trans), annot=True, fmt=".2f", cmap='Blues', cbar=True)
    plt.title(title)
    plt.xlabel("to state"); plt.ylabel("from state")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[PLOT] Saved {outpath}")

# ---------------------- LaTeX helper --------------------------------------
def df_to_latex_table(df, caption, label):
    latex = df.to_latex(index=False, float_format="%.6f")
    table = "\\begin{table}[h]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
    table += latex + "\\end{table}\n"
    return table

# ---------------------- Run pipeline -------------------------------------
def run(ticker="AAPL", start="2013-01-01", end="2023-12-31", n_states=2, use_hmm=True, zclip=6.0):
    outdir = ensure_dir(OUTDIR)
    print("[RUN] outdir:", outdir)

    # 1. Download prices
    print(f"[RUN] Downloading {ticker} from {start} to {end} ...")
    prices_df = download_prices(ticker, start, end, outdir=outdir)

    # Debug prints
    print("[DEBUG] prices_df columns:", prices_df.columns.tolist(), "rows:", len(prices_df))

    # 2. Preprocess returns
    df = compute_log_returns(prices_df)
    df = clean_returns(df, zclip=zclip)
    if df is None or len(df) < 30:
        raise RuntimeError("Not enough return data after preprocessing to fit model.")
    returns = df['return'].values
    dates = df.index.to_numpy()
    prices = df['adj_close'].values

    # Save processed
    df.to_csv(os.path.join(outdir, f"{ticker}_returns_processed.csv"))
    print(f"[RUN] Saved processed returns to {os.path.join(outdir, f'{ticker}_returns_processed.csv')}")

    # 3. Fit model
    labels, posteriors, model_info = fit_hmm_or_gmm(returns, n_states=n_states, use_hmm=use_hmm, outdir=outdir)
    labels = np.asarray(labels, dtype=int)
    print(f"[RUN] Fitted model type: {model_info.get('type', 'Unknown')}")

    # 4. Analyze
    summary_df = summarize_states(returns, labels)
    counts, trans = empirical_transition(labels, n_states)
    forecast = one_step_forecast(labels, trans)

    # 5. Save outputs CSV / JSON
    out_states = pd.DataFrame({'date': dates, 'adj_close': prices, 'return': returns, 'state': labels})
    out_states.to_csv(os.path.join(outdir, f"{ticker}_hmm_states.csv"), index=False)
    summary_df.to_csv(os.path.join(outdir, f"{ticker}_state_summary.csv"), index=False)
    pd.DataFrame(counts, columns=[f"S{i}" for i in range(n_states)], index=[f"S{i}" for i in range(n_states)]).to_csv(os.path.join(outdir, f"{ticker}_trans_counts.csv"))
    pd.DataFrame(trans, columns=[f"S{i}" for i in range(n_states)], index=[f"S{i}" for i in range(n_states)]).to_csv(os.path.join(outdir, f"{ticker}_trans_matrix.csv"))
    save_json(model_info, os.path.join(outdir, f"{ticker}_model_info.json"))
    save_json(forecast, os.path.join(outdir, f"{ticker}_forecast.json"))

    print("[RUN] Saved CSVs and JSONs.")

    # 6. Plots
    plot_price_states(dates, prices, labels, os.path.join(outdir, f"{ticker}_price_states.png"))
    plot_returns_states(dates, returns, labels, os.path.join(outdir, f"{ticker}_returns_states.png"))
    plot_transition_heatmap(trans, os.path.join(outdir, f"{ticker}_trans_heatmap.png"))

    # 7. LaTeX table snippets
    latex_state_table = df_to_latex_table(summary_df, caption=f"{ticker} HMM state summary", label=f"tab:{ticker.lower()}_state_summary")
    with open(os.path.join(outdir, f"latex_{ticker}_state_table.tex"), "w") as f:
        f.write(latex_state_table)
    latex_trans = df_to_latex_table(pd.DataFrame(trans, columns=[f"S{i}" for i in range(n_states)]), caption=f"{ticker} HMM transition matrix", label=f"tab:{ticker.lower()}_trans")
    with open(os.path.join(outdir, f"latex_{ticker}_trans_table.tex"), "w") as f:
        f.write(latex_trans)

    # 8. Markdown report
    mdpath = os.path.join(outdir, f"{ticker}_report.md")
    with open(mdpath, "w") as f:
        f.write(f"# Gaussian HMM Report for {ticker}\n\n")
        f.write(f"Period: {start} to {end}\n\n")
        f.write("See generated CSVs, PNGs and tables in this outputs directory.\n\n")
        f.write("State summary:\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\nTransition matrix:\n\n")
        f.write(pd.DataFrame(trans, columns=[f"S{i}" for i in range(n_states)], index=[f"S{i}" for i in range(n_states)]).to_markdown())
        f.write("\n\nForecast:\n\n")
        f.write(json.dumps(forecast, indent=2))
    print(f"[RUN] Markdown report saved: {mdpath}")

    print("[RUN] Complete. Outputs saved in:", outdir)

# ---------------------- CLI -----------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian HMM pipeline")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol")
    parser.add_argument("--start", type=str, default="2013-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2023-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--n_states", type=int, default=2, help="Number of hidden states")
    parser.add_argument("--use_hmm", type=int, choices=[0,1], default=1, help="use hmmlearn if available")
    parser.add_argument("--zclip", type=float, default=6.0, help="z-score clipping for returns cleaning")
    args = parser.parse_args()
    run(ticker=args.ticker, start=args.start, end=args.end, n_states=args.n_states, use_hmm=bool(args.use_hmm), zclip=args.zclip)
