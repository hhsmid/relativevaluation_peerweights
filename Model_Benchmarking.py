# %% Load packages
import os
import glob
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
from sklearn.metrics import mean_squared_error
from pandas_datareader import data as pdr

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


# %% CPI data loading
# Define start date and end date
start_date = pd.to_datetime(months[0])
end_date = pd.to_datetime(months[-1])

# Load the CPI data from FRED
cpi_monthly = (
    pdr.DataReader(
        name="CPIAUCNS",
        data_source="fred",
        start=start_date,
        end=end_date
    )
    .reset_index(names="month")
    .rename(columns={"CPIAUCNS": "cpi"})
    .assign(cpi=lambda x: x["cpi"] / x["cpi"].iloc[-1])  # Normalize to 1 at end
)

# Ensure datetime consistency
cpi_monthly['month'] = pd.to_datetime(cpi_monthly['month'])


# %% Aggregate the results for each Model-Multiple combination
aggregated_metrics = []
rmse_dict = {model: {} for model in models}
rmse_dict_cpi = {model: {} for model in models}
r2_dict = {model: {} for model in models}
mape_dict = {model: {} for model in models}
industry_errors_dict = {model: [] for model in models}

# Plot settings
sns.set(style="whitegrid")

# Define consistent colors and line styles
color_map = {"m2b": "#d62728", "v2a": "#1f77b4", "v2s": "#2ca02c"}
line_styles = {"m2b": "-", "v2a": "--", "v2s": ":"}

# Helper function to compute R-squared and RMSE
def compute_metrics_from_outcomes(outcomes_df, cpi_df):
    """
    Computes median RMSE and R² per month from fold-level outcome files.
    """
    fold_metrics = []

    # Merge CPI data
    outcomes_df = outcomes_df.merge(cpi_df, on="month", how="left")

    # Group by month and fold to calculate per-fold metrics
    grouped = outcomes_df.groupby(['month', 'fold'])

    for (month, fold), group in grouped:
        y_true = group['mktcap_actual']
        y_pred = group['mktcap_pred']
        cpi = group['cpi']

        # Filter out invalid values
        valid = (y_true > 0) & (y_pred > 0) & cpi.notna()
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        cpi = cpi[valid]

        if len(y_true) < 5:
            continue
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

        # Calculate CPI-adjusted RMSE
        y_true_real = y_true / cpi
        y_pred_real = y_pred / cpi
        rmse_cpi = np.sqrt(mean_squared_error(y_true_real, y_pred_real))

        fold_metrics.append({
            'month': month,
            'fold': fold,
            'rmse': rmse,
            'rmse_cpi': rmse_cpi,
            'r2': r2
        })

    # Convert to DataFrame and compute medians per month
    fold_metrics_df = pd.DataFrame(fold_metrics)
    monthly_medians = fold_metrics_df.groupby('month')[['rmse', 'rmse_cpi', 'r2']].median()

    return monthly_medians['rmse'], monthly_medians['rmse_cpi'], monthly_medians['r2']


# Process data and store performance metrics
for model in models:
    for multiple in multiples:
        print(f"\nProcessing Model: {model} - Multiple: {multiple}")

        # Get outcome files
        outcome_files_pattern = OUTCOMES_TEMPLATE.format(model=model, multiple=multiple)
        percentage_error_files = glob.glob(outcome_files_pattern)

        if not percentage_error_files:
            print(f"Warning: No outcome files found for {model} - {multiple}. Skipping...")
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
        

        # Compute aggregated metrics
        avg_absolute_percentage_error = np.mean(np.abs(percentage_errors_df['percentage_error']))
        med_absolute_percentage_error = np.median(np.abs(percentage_errors_df['percentage_error']))
        sd_absolute_percentage_error = np.std(np.abs(percentage_errors_df['percentage_error']))
        
        avg_percentage_error = np.mean(percentage_errors_df['percentage_error'])
        med_percentage_error = np.median(percentage_errors_df['percentage_error'])
        sd_percentage_error = np.std(percentage_errors_df['percentage_error'])
        
        # Compute RMSE and R² per month directly from outcome files
        rmse_series, rmse_cpi_series, r2_series = compute_metrics_from_outcomes(percentage_errors_df, cpi_monthly)

        # Reindex to all months to make sure time axis aligns
        rmse_series = rmse_series.reindex(pd.to_datetime(months))
        rmse_cpi_series = rmse_cpi_series.reindex(pd.to_datetime(months))
        r2_series = r2_series.reindex(pd.to_datetime(months))
        
        mean_rmse = rmse_series.mean()
        mean_rmse_cpi = rmse_cpi_series.mean()
        mean_r2 = r2_series.mean()
        
        # Display the metrics
        print("\n=== Aggregated Evaluation Metrics ===")
        print(f"Mean Absolute Percentage Error: {avg_absolute_percentage_error}")
        print(f"Median Absolute Percentage Error: {med_absolute_percentage_error}")
        print(f"Standard Deviation Absolute Percentage Error: {sd_absolute_percentage_error}")
        print(f"Mean Percentage Error (non-absolute): {avg_percentage_error}")
        print(f"Median Percentage Error (non-absolute): {med_percentage_error}")
        print(f"Standard Deviation Percentage Error (non-absolute): {sd_percentage_error}")
        print(f"Average Monthly Median RMSE: {mean_rmse}")
        print(f"Average Monthly Median CPI-adjusted RMSE: {mean_rmse_cpi}")
        print(f"Average Monthly Median R²: {mean_r2}")

        # Store results
        aggregated_metrics.append({
            "Model": model,
            "Multiple": multiple,
            "Mean_Absolute_Percentage_Error": avg_absolute_percentage_error,
            "Median_Absolute_Percentage_Error": med_absolute_percentage_error,
            "SD_Absolute_Percentage_Error": sd_absolute_percentage_error,
            "Mean_Percentage_Error": avg_percentage_error,
            "Median_Percentage_Error": med_percentage_error,
            "SD_Percentage_Error": sd_percentage_error,
            "Avg_Mon_Med_RMSE": mean_rmse,
            "Avg_Mon_Med_CPI_Adjusted_RMSE": mean_rmse_cpi,
            "Avg_Mon_Med_R2": mean_r2
        })

        
        # ==== Compute median absolute percentage error over time ====

        # Compute and store MAPE series
        median_error_over_time = percentage_errors_df['percentage_error'].abs().groupby(percentage_errors_df['month']).median()
        mape_dict[model][multiple] = median_error_over_time.reindex(pd.to_datetime(months))

        # Store RMSE and R² series for plotting
        rmse_dict[model][multiple] = rmse_series
        rmse_dict_cpi[model][multiple] = rmse_cpi_series
        r2_dict[model][multiple] = r2_series


        # ==== Compute industry-level median absolute percentage error ====
        
        percentage_errors_df['gvkey'] = percentage_errors_df['gvkey'].astype(str)
        data['gvkey'] = data['gvkey'].astype(str)

        # Merge industry data
        industry_cols = [col for col in data.columns if any(ff in col for ff in ff10)]
        percentage_errors_df = percentage_errors_df.merge(
            data[['gvkey', 'month'] + industry_cols], on=['gvkey', 'month'], how='left'
        )

        for industry in ff10:
            industry_col = next((col for col in industry_cols if industry in col), None)
            if industry_col:
                median_error = percentage_errors_df.loc[percentage_errors_df[industry_col] == 1, 'percentage_error'].abs().median()
                industry_errors_dict[model].append({
                    'Industry': industry,
                    'Median Absolute Percentage Error': median_error,
                    'Multiple': multiple
                })

# Save aggregated metrics
pd.DataFrame(aggregated_metrics).to_csv(os.path.join(RESULTS_DIR, "model_perf_benchmark.csv"), index=False)
print("\nAggregated metrics saved successfully.")

# Save industry errors
industry_errors_all = []
for model in models:
    for record in industry_errors_dict[model]:
        record['Model'] = model
        industry_errors_all.append(record)
industry_errors_df = pd.DataFrame(industry_errors_all)
industry_errors_csv_path = os.path.join(RESULTS_DIR, "industry_errors.csv")
industry_errors_df.to_csv(industry_errors_csv_path, index=False)
print(f"\nIndustry errors saved successfully to {industry_errors_csv_path}.")


# %% Plots over time and across industries
# Over time
for model in models:
    plt.figure(figsize=(10, 6))

    # RMSE Plot
    for multiple in multiples:
        if multiple in rmse_dict[model]:
            plt.plot(rmse_dict[model][multiple], label=multiple, linestyle=line_styles[multiple], color=color_map[multiple])

    plt.ylabel("RMSE")
    plt.title(f"Monthly Median RMSE ({model})")
    plt.xlabel("Time (Years)")
    plt.xticks(rotation=30)
    plt.legend(title="Multiple")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{model}_rmse_over_time.png"), dpi=100)
    plt.show()


    # CPI-adjusted RMSE Plot
    plt.figure(figsize=(10, 6))
    for multiple in multiples:
        if multiple in rmse_dict_cpi[model]:
            plt.plot(
                rmse_dict_cpi[model][multiple],
                label=multiple,
                linestyle=line_styles[multiple],
                color=color_map[multiple]
            )
    plt.ylabel("CPI-Adjusted RMSE")
    plt.title(f"Monthly Median CPI-Adjusted RMSE ({model})")
    plt.xlabel("Time (Years)")
    plt.xticks(rotation=30)
    plt.legend(title="Multiple")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{model}_rmse_cpi_over_time.png"), dpi=100)
    plt.show()


    # R² Plot
    plt.figure(figsize=(10, 6))
    for multiple in multiples:
        if multiple in r2_dict[model]:
            plt.plot(r2_dict[model][multiple], label=multiple, linestyle=line_styles[multiple], color=color_map[multiple])

    plt.ylabel("R²")
    plt.title(f"Monthly Median Out-of-Sample R² ({model})")
    plt.xlabel("Time (Years)")
    plt.xticks(rotation=30)
    plt.legend(title="Multiple")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{model}_r2_over_time.png"), dpi=100)
    plt.show()


    # MAPE Plot
    plt.figure(figsize=(10, 6))
    for multiple in multiples:
        if multiple in mape_dict[model]:
            plt.plot(mape_dict[model][multiple], label=multiple, linestyle=line_styles[multiple], color=color_map[multiple])

    plt.ylabel("Median Absolute Percentage Error")
    plt.title(f"Median Absolute Percentage Error Over Time ({model})")
    plt.xlabel("Time (Years)")
    plt.xticks(rotation=30)
    plt.legend(title="Multiple")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{model}_median_absolute_percentage_error.png"), dpi=100)
    plt.show()


# Across industries
for model in models:
    industry_errors_df = pd.DataFrame(industry_errors_dict[model])

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Median Absolute Percentage Error',
        y='Industry',
        hue='Multiple',
        data=industry_errors_df,
        palette=color_map,
        hue_order=multiples
    )
    plt.xlabel("Median Absolute Percentage Error")
    plt.ylabel("Fama-French 10 Industries")
    plt.title(f"Median Absolute Percentage Error Across Industries ({model})")
    plt.legend(title="Multiple")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{model}_industry_errors.png"), dpi=100)
    plt.show()

