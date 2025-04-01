# %% Load packages
import os
import glob
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# %% Define global variables
# Define results paths
BASE_DIR = ""
OUTCOMES_TEMPLATE = os.path.join(BASE_DIR, "{model}/{multiple}/Outcomes/*.csv")
PLOT_DIR = ""
RESULTS_DIR = ""

# Define database path
DB_PATH = "/Data/sql_database.sqlite"

# Define models and multiples
models = ["K-means", "HAC", "GMM", "FCM", "GBM", "RF"]
multiples = ["m2b", "v2a", "v2s"]

# Define months
months = pd.date_range(start="1990-01", end="2023-09", freq='ME').strftime('%Y-%m')

# Define folds
folds = range(1, 6)

# Industry classification names (Fama-French 10)
ff10 = [
    '1-Consumer NonDurables', '2-Consumer Durables', '3-Manufacturing', '4-Energy',
    '5-Chemicals', '6-Business Equipment', '7-Telecommunication', '8-Healthcare',
    '9-Utilities', '10-Other'
]

# Set number of CPU cores
cores = 7


# %% Data Loading and Processing
print("Loading data from SQLite...")

def fetch_chunk(offset, limit):
    """Fetches a chunk of data from SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = f"SELECT * FROM final_merged_data LIMIT {limit} OFFSET {offset}"
        chunk = pd.read_sql_query(query, conn)
        conn.close()
        return chunk
    except Exception as e:
        print(f"Error fetching chunk at offset {offset}: {e}")
        return pd.DataFrame()

# Get total row count
with sqlite3.connect(DB_PATH) as conn:
    row_count = pd.read_sql_query("SELECT COUNT(*) AS cnt FROM final_merged_data", conn)['cnt'].iloc[0]

# Set chunk size and parallelize loading
chunk_size = 30000
chunks = Parallel(n_jobs=cores)(
    delayed(fetch_chunk)(offset, chunk_size) for offset in range(0, row_count, chunk_size)
)

# Combine chunks into a single DataFrame
data = pd.concat(chunks, ignore_index=True)

# Convert 'month' to datetime format and filter valid months
data['month'] = pd.to_datetime(data['month'], errors='coerce')
data = data.dropna(subset=['month'])

print(f"Loaded {len(data)} rows of data.")



# %% Portfolio sorting for each Model-Multiple combination
portfolio_results = []
all_excess_returns = []

for model in models:
    
    model_cum_returns = {}
    for multiple in multiples:
        # Reload the correct percentage errors for this model-multiple
        outcome_files_pattern = OUTCOMES_TEMPLATE.format(model=model, multiple=multiple)
        percentage_error_files = glob.glob(outcome_files_pattern)

        if not percentage_error_files:
            print(f"Warning: No outcome files found for {model} - {multiple}. Skipping portfolio sorting...")
            continue

        # Read and process outcome files in parallel
        def read_and_process(file):
            df = pd.read_csv(file)
            df['month'] = pd.to_datetime(df['month'])
            return df

        all_errors = Parallel(n_jobs=cores)(
            delayed(read_and_process)(file) for file in tqdm(percentage_error_files, desc=f"Reading CSV for {model}-{multiple}")
        )

        # Concatenate into one DataFrame
        percentage_errors_df = pd.concat(all_errors, ignore_index=True)

        # Make a copy of the original data
        data_merged = data.copy()

        # Merge percentage errors for this model-multiple into the copied data
        percentage_errors_df['gvkey'] = percentage_errors_df['gvkey'].astype(str)
        data_merged['gvkey'] = data_merged['gvkey'].astype(str)

        data_merged = data_merged.merge(
            percentage_errors_df[['gvkey', 'month', 'percentage_error']],
            on=['gvkey', 'month'],
            how='left'
        ).sort_values(["gvkey", "month"])

        # Create lagged valuation error
        data_merged["percentage_error_lag"] = data_merged.groupby("gvkey")["percentage_error"].shift(1)

        # Drop rows where key variables are missing
        data_merged = data_merged.dropna(subset=["percentage_error_lag", "mktcap_lag", "ret"])
        
        print(f"\nPerforming Portfolio Sorting for {model} - {multiple}...", flush=True)

        # Assign quintiles for portfolio sorting using pd.qcut with a small jitter for bin stability
        def assign_quintile_portfolio(df, sorting_variable):
            jittered = df[sorting_variable] + np.random.normal(0, 1e-6, size=len(df))
            quantiles = jittered.quantile([0.2, 0.4, 0.6, 0.8]).values
            return pd.cut(jittered, bins=[-np.inf] + quantiles.tolist() + [np.inf], labels=range(1, 6))

        sorted_portfolios = (data_merged
            .groupby("month")
            .apply(lambda x: x.assign(portfolio=assign_quintile_portfolio(x, "percentage_error_lag")))
            .reset_index(drop=True)
        )

        # Compute value-weighted returns per portfolio per month
        portfolio_returns = (sorted_portfolios
            .groupby(["month", "portfolio"])
            .apply(lambda x: np.average(x["ret"], weights=x["mktcap_lag"], axis=0) if not x.empty else np.nan)
            .reset_index(name="portfolio_ret")
        )

        # Compute long-short portfolio return (Q5 - Q1)
        long_short_portfolio = (portfolio_returns
            .pivot(index="month", columns="portfolio", values="portfolio_ret")
            .assign(long_short=lambda x: x[5] - x[1])
            .reset_index()
        )

        # Merge Fama-French factors
        long_short_ff3 = long_short_portfolio.merge(
            data[["month", "mkt_excess", "smb", "hml", "rf"]].drop_duplicates(),
            on="month",
            how="left"
        )

        print(f"\nRunning FF3 regression for {model} - {multiple}...", flush=True)

        # Run FF3 regression with HAC standard errors
        ff3_model = sm.OLS(long_short_ff3["long_short"], sm.add_constant(long_short_ff3[["mkt_excess", "smb", "hml"]]))
        results = ff3_model.fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        print(results.summary())
        
        # Generate LaTeX formatted regression summary using summary2
        regression_summary_latex = results.summary2().as_latex()
        
        # Define the output path for the LaTeX summary file (saved as a .txt file)
        latex_file_path = os.path.join(RESULTS_DIR, f"{model}_{multiple}_regression_summary.txt")
        
        # Save the LaTeX summary to the file
        with open(latex_file_path, "w") as file:
            file.write(regression_summary_latex)
        
        print(f"\nRegression summary saved as LaTeX table to {latex_file_path}")
        
        # Compute mean return of the long-short portfolio
        mean_return = long_short_ff3["long_short"].mean()
        print(f"\nMean Monthly Return of Long-Short Portfolio: {mean_return:.4f}")
        
        # Calculate excess returns and compute Sharpe ratio
        long_short_ff3["excess_return"] = long_short_ff3["long_short"] - long_short_ff3["rf"]
        sharpe_ratio = long_short_ff3["excess_return"].mean() / long_short_ff3["excess_return"].std()
        print(f"\nSharpe Ratio of Long-Short Portfolio: {sharpe_ratio:.4f}\n")
        
        # Store excess returns with metadata for later comparison
        excess_return_series = long_short_ff3[["month", "excess_return"]].copy()
        excess_return_series["strategy"] = f"{model}-{multiple}"
        all_excess_returns.append(excess_return_series)

        # Get the full regression summary as text
        regression_summary_text = results.summary().as_text()
        
        # Store Portfolio Analysis Results along with the full regression summary
        portfolio_results.append({
            "Model": model,
            "Multiple": multiple,
            "Alpha": results.params["const"],
            "Sharpe Ratio": sharpe_ratio,
            "Mean Return": mean_return,
            "Regression Summary": regression_summary_text
        })

        # Compute Cumulative Returns
        long_short_ff3["cumulative_return"] = (1 + long_short_ff3["long_short"]).cumprod()
        
        # Instead of plotting here, store the cumulative returns for later combined plotting
        model_cum_returns[multiple] = long_short_ff3[["month", "cumulative_return"]].copy()

    # Define consistent colors and line styles
    color_map = {"m2b": "#d62728", "v2a": "#1f77b4", "v2s": "#2ca02c"}
    line_styles = {"m2b": "-", "v2a": "--", "v2s": ":"}

    # After looping through all multiples for the current model, plot all curves in one figure
    if model_cum_returns:  # ensure that there is data to plot
        plt.figure(figsize=(10, 5))
        for multiple, df_plot in model_cum_returns.items():
            plt.plot(df_plot["month"], df_plot["cumulative_return"],
                     label=multiple,
                     color=color_map[multiple],
                     linestyle=line_styles[multiple])
        plt.axhline(y=1, color="gray", linestyle="--")
        plt.title(f"Cumulative Returns of the Long-Short Portfolio ({model})")
        plt.xlabel("Year")
        plt.ylabel("Cumulative Return")
        plt.legend()
        # Save one file per model
        plt.savefig(os.path.join(PLOT_DIR, f"{model}_cumulative_returns.png"), dpi=100)
        plt.show()

# Save Portfolio Analysis Results to Excel
portfolio_df = pd.DataFrame(portfolio_results)
output_path = os.path.join(RESULTS_DIR, "portfolio_analysis_results.xlsx")
portfolio_df.to_excel(output_path, index=False)
print(f"\nPortfolio analysis results saved successfully to {output_path}.")


# %% Ledoit & Wolf test for Sharpe ratio differences
# Combine all strategy excess returns
all_excess_df = pd.concat(all_excess_returns, ignore_index=True)

# Pivot for easier access: one column per strategy
pivot_df = all_excess_df.pivot(index="month", columns="strategy", values="excess_return")

# Initialize empty matrix
pval_matrix = pd.DataFrame(index=strategies, columns=strategies)

# Fill the lower triangle with p-values from Sharpe ratio test
for i, s1 in enumerate(strategies):
    for j, s2 in enumerate(strategies):
        if i == j:
            pval_matrix.loc[s1, s2] = "---"
        elif i > j:
            r1 = pivot_df[s1].dropna()
            r2 = pivot_df[s2].dropna()
            
            # Align time index
            common_idx = r1.index.intersection(r2.index)
            r1_aligned = r1.loc[common_idx].values
            r2_aligned = r2.loc[common_idx].values

            if len(r1_aligned) < 10:  # safety check for short series
                pval = np.nan
            else:
                stat, pval = sharpe_ratio_test(r1_aligned, r2_aligned)

            pval_matrix.loc[s1, s2] = f"{pval:.4f}" if not np.isnan(pval) else "NA"
        else:
            pval_matrix.loc[s1, s2] = ""

# Save to CSV
csv_path = os.path.join(RESULTS_DIR, "sharpe_ratio_lower_triangular.csv")
pval_matrix.to_csv(csv_path)
print(f"\nLower triangular Sharpe ratio test matrix saved as:\n{csv_path}")

