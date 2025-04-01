# %% Load packages
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import combinations
import warnings
from collections import defaultdict

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# %% Define global variables
# Define results paths
BASE_DIR = ""
PEER_WEIGHTS_TEMPLATE = os.path.join(BASE_DIR, "{model}/{multiple}/Peer weights/peer_weights_{month}_fold{fold}.csv")
PLOT_DIR = ""
RESULTS_DIR = ""

# Define database path
DB_PATH = "Data/sql_database.sqlite"

# Define models and multiples
models = ["K-means", "HAC", "GMM", "FCM", "GBM", "RF"]
multiples = ["m2b", "v2a", "v2s"]

# Define a model abbreviation mapping (for table display)
model_abbr = {
    "K-means": "KM",
    "HAC": "HAC",
    "GMM": "GMM",
    "FCM": "FCM",
    "GBM": "GBM",
    "RF": "RF"
}

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


# %% Peer Weights plots and Frobenius Norm Matrix
def check_gvkey_consistency():
    """
    Ensure all files for a given month-fold combination contain the same unique train firm gvkeys (rows)
    and test firm gvkeys (columns) across models, with progress bars added.
    """
    # Gather all peer weight files
    pattern = os.path.join(BASE_DIR, "*/*/Peer weights/peer_weights_*_fold*.csv")
    all_files = glob.glob(pattern)
    
    # Group files by (month, fold)
    file_groups = defaultdict(list)
    for file in all_files:
        basename = os.path.basename(file)
        if basename.startswith("peer_weights_") and "_fold" in basename:
            temp = basename[len("peer_weights_"):-len(".csv")]
            parts = temp.split("_fold")
            if len(parts) == 2:
                month, fold = parts[0], parts[1]
                file_groups[(month, fold)].append(file)
    
    # Process each month-fold combination across models
    def process_group(key, files):
        month, fold = key
        messages = []
        
        # Helper function: load file and extract gvkeys
        def get_gvkeys(file_path):
            try:
                df = pd.read_csv(file_path, index_col=0)
                return set(df.index), set(df.columns)
            except Exception as e:
                messages.append(f"Error reading {file_path}: {e}")
                return None, None
    
        # Use the first valid file as reference (no inner progress bar)
        ref_train, ref_test = None, None
        for file in files:
            train_gvkeys, test_gvkeys = get_gvkeys(file)
            if train_gvkeys is not None and test_gvkeys is not None:
                ref_train, ref_test = train_gvkeys, test_gvkeys
                break
        if ref_train is None:
            messages.append(f"No valid file found for month {month}, fold {fold}")
            return messages
    
        # Compare all files in this group against the reference gvkeys (again, no inner progress bar)
        for file in files:
            train_gvkeys, test_gvkeys = get_gvkeys(file)
            if train_gvkeys is None or test_gvkeys is None:
                continue
            if train_gvkeys != ref_train:
                messages.append(f"Mismatch in train firm gvkeys for {file} (month {month}, fold {fold})")
            if test_gvkeys != ref_test:
                messages.append(f"Mismatch in test firm gvkeys for {file} (month {month}, fold {fold})")
        return messages

    # Process each group
    group_items = list(file_groups.items())
    results = Parallel(n_jobs=cores, backend='threading')(
        delayed(process_group)(key, files)
        for key, files in tqdm(group_items, desc="Processing groups")
    )
    
    # Flatten and print any mismatch messages
    for group_msgs in results:
        for msg in group_msgs:
            print(msg)


def load_peer_weights(model, multiple, month, fold):
    """
    Load peer weight CSV file.
    """
    file_path = PEER_WEIGHTS_TEMPLATE.format(model=model, multiple=multiple, month=month, fold=fold)
    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=0).astype(np.float32)
    else:
        print(f"File not found: {file_path}")
        return None
    

def plot_peer_weights(test_firm_gvkey, month, fold):
    """
    For each model, plot a single figure that includes peer weights for all three multiples.
    The multiples are distinguished by consistent colors and line styles.
    """
    # Define consistent colors and line styles
    color_map = {"m2b": "#d62728", "v2a": "#1f77b4", "v2s": "#2ca02c"}
    line_styles = {"m2b": "-", "v2a": "--", "v2s": ":"}
    
    for model in models:
        plt.figure(figsize=(10, 6))
        for multiple in multiples:
            df = load_peer_weights(model, multiple, month, fold)
            if df is not None and test_firm_gvkey in df.columns:
                peer_weights = df[test_firm_gvkey].sort_values(ascending=False)
                plt.plot(
                    peer_weights.values,
                    label=multiple,
                    color=color_map[multiple],
                    linestyle=line_styles[multiple]
                )
            else:
                print(f"No data for {model}-{multiple} for test firm {test_firm_gvkey} in {month}, fold {fold}.")
        plt.xlabel("Training Firms (In Descending Order By Weight)")
        plt.ylabel("Weight")
        plt.title(f"Distribution of Peer Weights for Apple Inc. in {month} ({model})")
        plt.legend(title="Multiple")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{model}_peer_weights_single.png"), dpi=100)
        plt.show()


def compute_concentration_metrics():
    """
    Compute peer weight concentration statistics across all available files.

    For each peer weight vector (each test firm column in a file), compute:
      - ENP: Effective Number of Peers = 1 / (sum of squared weights)
      - WTS: Weight Threshold Share for thresholds tau = 0.001, 0.01, 0.1
      - Gini: Gini coefficient of the weight distribution

    Aggregates are taken over all test firms (across months and folds) for each model–multiple combination.
    Returns a dictionary of DataFrames (one per multiple) with rows corresponding to models
    (using abbreviations) and columns for ENP, WTS at each threshold, and Gini.
    """
    thresholds = [0.001, 0.01, 0.1]

    # Prepare a dictionary to accumulate results per model/multiple.
    metrics = {
        model: {
            multiple: {"enp": [], "wts": {tau: [] for tau in thresholds}, "gini": []}
            for multiple in multiples
        }
        for model in models
    }

    # Define a helper function that computes metrics for a given file.
    def process_file(model, multiple, month, fold):
        file_results = []
        df = load_peer_weights(model, multiple, month, fold)
        if df is not None:
            # Process each test firm (each column)
            for col in df.columns:
                weights = df[col].values
                total = np.sum(weights)
                if total > 0:
                    weights = weights / total  # Normalize to sum to 1
                else:
                    continue  # Skip if total is zero

                # Compute ENP
                enp_val = 1.0 / np.sum(weights**2)
                # Compute WTS for each threshold
                wts_vals = {tau: np.sum(weights > tau) for tau in thresholds}
                # Compute Gini coefficient
                n = len(weights)
                if n > 0:
                    diff_sum = np.sum(np.abs(np.subtract.outer(weights, weights)))
                    gini_val = diff_sum / (2 * n * np.sum(weights))
                else:
                    gini_val = np.nan

                file_results.append((model, multiple, enp_val, wts_vals, gini_val))
        return file_results

    # Build a list of tasks (one per file)
    tasks = [
        (model, multiple, month, fold)
        for model in models
        for multiple in multiples
        for month in months
        for fold in folds
    ]

    # Process all files in parallel
    from joblib import Parallel, delayed
    all_results = Parallel(n_jobs=cores)(
        delayed(process_file)(model, multiple, month, fold)
        for model, multiple, month, fold in tqdm(tasks, desc="Processing files")
    )

    # Flatten the list of results (each element is a list of tuples)
    all_results_flat = [res for sublist in all_results for res in sublist]

    # Aggregate the results into the metrics dictionary.
    for res in all_results_flat:
        model, multiple, enp_val, wts_vals, gini_val = res
        metrics[model][multiple]["enp"].append(enp_val)
        for tau in thresholds:
            metrics[model][multiple]["wts"][tau].append(wts_vals[tau])
        metrics[model][multiple]["gini"].append(gini_val)

    # Aggregate the metrics: compute mean and std for each model and multiple combination.
    results = {
        multiple: pd.DataFrame(
            index=[model_abbr[m] for m in models],
            columns=["ENP", "WTS_0.001", "WTS_0.01", "WTS_0.1", "Gini"],
        )
        for multiple in multiples
    }

    for model in models:
        for multiple in multiples:
            enp_list = metrics[model][multiple]["enp"]
            gini_list = metrics[model][multiple]["gini"]
            wts_list = metrics[model][multiple]["wts"]

            enp_mean = np.mean(enp_list) if enp_list else np.nan
            enp_std = np.std(enp_list) if enp_list else np.nan

            gini_mean = np.mean(gini_list) if gini_list else np.nan
            gini_std = np.std(gini_list) if gini_list else np.nan

            wts_means = {}
            wts_stds = {}
            for tau in thresholds:
                lst = wts_list[tau]
                wts_means[tau] = np.mean(lst) if lst else np.nan
                wts_stds[tau] = np.std(lst) if lst else np.nan

            results[multiple].loc[model_abbr[model], "ENP"] = f"{enp_mean:.2f} ({enp_std:.2f})"
            results[multiple].loc[model_abbr[model], "WTS_0.001"] = f"{wts_means[0.001]:.2f} ({wts_stds[0.001]:.2f})"
            results[multiple].loc[model_abbr[model], "WTS_0.01"] = f"{wts_means[0.01]:.2f} ({wts_stds[0.01]:.2f})"
            results[multiple].loc[model_abbr[model], "WTS_0.1"] = f"{wts_means[0.1]:.2f} ({wts_stds[0.1]:.2f})"
            results[multiple].loc[model_abbr[model], "Gini"] = f"{gini_mean:.2f} ({gini_std:.2f})"

    return results


def compute_frobenius_norms():
    """
    Compute Frobenius norms between peer weight matrices across model-multiple combinations.
    """
    comb_keys = [f"{model}_{multiple}" for model in models for multiple in multiples]
    pair_results = {pair: [] for pair in combinations(comb_keys, 2)}

    for month in tqdm(months, desc="Computing Frobenius Norms (Months)"):
        for fold in tqdm(folds, desc="Processing Folds", leave=False):
            # Load matrices in parallel with progress bar
            matrices = dict(
                zip(
                    [(m, mu) for m in models for mu in multiples],
                    Parallel(n_jobs=cores)(
                        delayed(load_peer_weights)(model, multiple, month, fold)
                        for model in models for multiple in multiples
                    )
                )
            )

            # Filter out None values
            matrices = {
                f"{model}_{multiple}": df
                for (model, multiple), df in matrices.items()
                if df is not None
            }

            def compute_frob_norm(key1, key2):
                df1, df2 = matrices[key1], matrices[key2]
                common_index = df1.index.intersection(df2.index)
                common_columns = df1.columns.intersection(df2.columns)
                if len(common_index) == 0 or len(common_columns) == 0:
                    print(f"No common gvkeys for {key1} and {key2} in {month}, fold {fold}. Skipping.")
                    return None
                
                aligned1, aligned2 = df1.loc[common_index, common_columns], df2.loc[common_index, common_columns]
                return np.linalg.norm(aligned1.values - aligned2.values, 'fro')

            # Compute Frobenius norms in parallel with tqdm
            results = Parallel(n_jobs=cores)(
                delayed(compute_frob_norm)(key1, key2)
                for key1, key2 in tqdm(combinations(matrices.keys(), 2), 
                                       desc="Frobenius Norm Computation", leave=False)
            )

            # Store results
            for (key1, key2), frob_norm in zip(combinations(matrices.keys(), 2), results):
                if frob_norm is not None:
                    pair_results[(key1, key2)].append(frob_norm)

    # Create mean and std matrices
    mean_matrix = pd.DataFrame(index=comb_keys, columns=comb_keys, dtype=float)
    std_matrix = pd.DataFrame(index=comb_keys, columns=comb_keys, dtype=float)

    for key1, key2 in pair_results.keys():
        values = pair_results[(key1, key2)]
        mean_matrix.loc[key1, key2] = mean_matrix.loc[key2, key1] = np.mean(values) if values else np.nan
        std_matrix.loc[key1, key2] = std_matrix.loc[key2, key1] = np.std(values) if values else np.nan

    return mean_matrix, std_matrix


# %% Execution
# Check gvkey consistency
check_gvkey_consistency()

# Example Test Firm and Month Selection for Plotting
test_firm_gvkey = "001690" # This is Apple Inc.
month = "2018-04"
fold = 3
plot_peer_weights(test_firm_gvkey, month, fold)

# Compute and display peer weight concentration metrics
print("\nComputing peer weight concentration metrics...")
concentration_results = compute_concentration_metrics()

# For each multiple, print the corresponding concentration table and save
for mult in multiples:
    print(f"\n=== Concentration Metrics for {mult.upper()} ===")
    print(concentration_results[mult])
    concentration_results[mult].to_csv(os.path.join(RESULTS_DIR, f"concentration_metrics_{mult}.csv"))

# Compute Frobenius Norms Across Model multiples
frob_matrix = compute_frobenius_norms()
print("\n=== Frobenius Norms Between Peer Weight Matrices ===")
print(frob_matrix)

# Save Frobenius Norms Results
frob_matrix.to_csv(os.path.join(BASE_DIR, "frobenius_norms_results.csv"))
print("\nFrobenius norms results saved successfully.")

