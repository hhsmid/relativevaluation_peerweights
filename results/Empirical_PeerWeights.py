# %% Load packages
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import combinations, product
import warnings
from collections import defaultdict
import seaborn as sns
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter

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

# Load the gvkey to company name mapping
mapping_file = '/gvkey indexes.xlsx'
mapping_df = pd.read_excel(mapping_file)
mapping_dict = dict(zip(mapping_df.iloc[:, 0].astype(str), mapping_df.iloc[:, 1]))

# Define models and multiples
models = ["K-means", "HAC", "GMM", "FCM", "GBM", "RF"]
multiples = ["m2b", "v2a", "v2s"]
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


# %% Peer Weights utility functions
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
                    color=color_map[multiple],
                    linestyle=line_styles[multiple],
                    linewidth=2
                )
            else:
                print(f"No data for {model}-{multiple} for test firm {test_firm_gvkey} in {month}, fold {fold}.")
        
        plt.xlabel("Training Firms (In Descending Order By Weight)")
        plt.ylabel("Weight")
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{model}_peer_weights_single.png"), dpi=100)
        plt.show()
        
        
def get_top_peers(test_firm_gvkey, month, fold, num_peers):
    """
    For a given test firm, month, and fold, construct a table with the top num_peers training firm gvkeys (peers)
    for each model-multiple combination.
    
    For hard clustering models, where all nonzero weights are equal,
    the candidate list is all gvkeys with nonzero weight and then re-ordered by the frequency
    of appearance across all combinations.
    
    For soft clustering models, if more than num_peers peers share the top weight (or tie at the cutoff),
    then the tied gvkeys are re-sorted by their frequency (i.e. overlap) across all combinations.
    
    Returns:
        A pandas DataFrame with columns: 'Model', 'Multiple', 'Top num_peers Peers'
    """
    # Define which models are considered hard clustering.
    hard_clustering_models = ["K-means", "HAC"]
    
    # Dictionary to store preliminary candidate lists and corresponding weights.
    preliminary_candidates = {}
    candidate_weights_all = {}
    
    # Loop through each model and multiple to extract candidate gvkeys.
    for model in models:
        for multiple in multiples:
            key = (model, multiple)
            df = load_peer_weights(model, multiple, month, fold)
            if df is not None and test_firm_gvkey in df.columns:
                weights = df[test_firm_gvkey]
                if model in hard_clustering_models:
                    # For hard clustering models: all nonzero weights are equal.
                    candidates = list(weights[weights > 0].index)
                    # Assign an arbitrary equal weight (here, 1) for later sorting.
                    candidate_weights = {gv: 1 for gv in candidates}
                else:
                    # For soft clustering models: sort by weight descending.
                    sorted_weights = weights.sort_values(ascending=False)
                    if len(sorted_weights) < num_peers:
                        candidates = list(sorted_weights.index)
                    else:
                        # Determine the cutoff using the weight of the num_peers'th candidate.
                        cutoff = sorted_weights.iloc[num_peers-1]
                        # Include all gvkeys with weight >= cutoff.
                        candidates = list(sorted_weights[sorted_weights >= cutoff].index)
                    candidate_weights = weights.to_dict()
            else:
                candidates = []
                candidate_weights = {}
            
            preliminary_candidates[key] = candidates
            candidate_weights_all[key] = candidate_weights

    # Compute overall frequency: count in how many candidate lists each gvkey appears.
    frequency = defaultdict(int)
    for cand_list in preliminary_candidates.values():
        for gv in cand_list:
            frequency[gv] += 1

    # Now, for each combination, determine the final top num_peers using frequency as a tie-breaker.
    top_peers_list = []
    for model in models:
        for multiple in multiples:
            key = (model, multiple)
            candidates = preliminary_candidates[key]
            candidate_weights = candidate_weights_all[key]
            final_candidates = []
            if not candidates:
                final_candidates = []
            elif model in hard_clustering_models:
                # For hard clustering models, sort candidates solely by frequency.
                sorted_candidates = sorted(candidates, key=lambda x: frequency[x], reverse=True)
                final_candidates = sorted_candidates[:num_peers]
            else:
                # For soft clustering models, first sort by weight descending.
                sorted_by_weight = sorted(candidates, key=lambda x: candidate_weights.get(x, -np.inf), reverse=True)
                if len(sorted_by_weight) <= num_peers:
                    final_candidates = sorted_by_weight
                else:
                    # Identify the cutoff weight (weight of the num_peers'th candidate).
                    cutoff = candidate_weights.get(sorted_by_weight[num_peers-1], None)
                    # Split candidates into those strictly above cutoff and those tied at cutoff.
                    above_cutoff = [gv for gv in sorted_by_weight if candidate_weights.get(gv, -np.inf) > cutoff]
                    tied = [gv for gv in sorted_by_weight if candidate_weights.get(gv, -np.inf) == cutoff]
                    remaining = num_peers - len(above_cutoff)
                    # Sort the tied candidates by frequency (overlap) in descending order.
                    tied_sorted = sorted(tied, key=lambda x: frequency[x], reverse=True)
                    final_candidates = above_cutoff + tied_sorted[:remaining]
                    # In case fewer than num_peers candidates were selected, add more from the tied list.
                    if len(final_candidates) < num_peers:
                        extra = tied_sorted[remaining:num_peers - len(final_candidates)]
                        final_candidates.extend(extra)
            company_names = [mapping_dict.get(str(candidate), str(candidate)) for candidate in final_candidates]
            top_peers_list.append({
                "Model": model,
                "Multiple": multiple,
                f"Top {num_peers} Peers": ", ".join(company_names) if company_names else "No Data"
            })
    return pd.DataFrame(top_peers_list)


def get_consolidated_top_peers(test_firm_gvkey, num_peers):
    """
    Aggregate top peers for a given test firm over all month-fold combinations.
    
    For each month and fold for which the test firm appears (as determined by the presence
    in the CSV files), the function extracts the top candidates using the get_top_peers function.
    It then aggregates the candidate lists (as given per model-multiple combination) via their frequency,
    sorts them in descending order of frequency, and returns the overall top num_peers.
    
    Parameters:
        test_firm_gvkey (str): The gvkey of the test firm.
        num_peers (int): The number of peers to extract per month-fold combination and overall.
    
    Returns:
        A list of tuples (gvkey, frequency) for the overall top peers.
    """
    # Initialize an aggregator: key is (model, multiple), value is a frequency dictionary.
    aggregator = {}
    for model in models:
        for multiple in multiples:
            aggregator[(model, multiple)] = defaultdict(int)
            
    # Loop through all month-fold combinations with a progress bar.
    total_iterations = len(months) * len(folds)
    for month, fold in tqdm(product(months, folds), total=total_iterations, desc="Processing month-fold combos"):
        try:
            # Extract top peers for the given test firm, month, and fold.
            df_top = get_top_peers(test_firm_gvkey, month, fold, num_peers)
        except Exception as e:
            print(f"Error processing month {month}, fold {fold}: {e}")
            continue
        
        # If no data, skip this combination.
        if df_top.empty:
            continue
        
        # Loop through each row (each row is one model-multiple combination).
        for _, row in df_top.iterrows():
            model = row.get("Model")
            multiple = row.get("Multiple")
            peers_str = row.get(f"Top {num_peers} Peers", "No Data")
            if peers_str == "No Data":
                continue
            # Split the candidate list string into individual gvkeys.
            candidates = [peer.strip() for peer in peers_str.split(",") if peer.strip()]
            # Update the frequency counter for the corresponding (model, multiple) key.
            for candidate in candidates:
                aggregator[(model, multiple)][candidate] += 1

    # Build the consolidated result for each (model, multiple) combination.
    consolidated_results = []
    for (model, multiple), freq_dict in aggregator.items():
        if freq_dict:
            # Sort candidates by frequency in descending order.
            sorted_candidates = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
            top_candidates = sorted_candidates[:num_peers]
            # Map the gvkeys to company names if available.
            consolidated_str = ", ".join([mapping_dict.get(str(candidate), str(candidate)) for candidate, _ in top_candidates])
        else:
            consolidated_str = "No Data"
        consolidated_results.append({
            "Model": model,
            "Multiple": multiple,
            f"Consolidated Top {num_peers} Peers": consolidated_str
        })
        
    return pd.DataFrame(consolidated_results)


def plot_heatmaps(test_firm_gvkey, num_peers):
    """
    Plot lower-triangle heatmaps for top peer overlaps
    """
    models_abbr_list = [model_abbr[model] for model in models]
    aggregated = {mult: pd.DataFrame(0, index=models_abbr_list, columns=models_abbr_list) for mult in multiples}
    plotted_matrices = {}

    # Build aggregated overlap matrices
    for month, fold in tqdm(product(months, folds), total=len(months) * len(folds), desc="Aggregating overlaps"):
        try:
            df_top = get_top_peers(test_firm_gvkey, month, fold, num_peers)
        except Exception as e:
            print(f"Error for month {month}, fold {fold}: {e}")
            continue
        if df_top.empty:
            continue

        for mult in multiples:
            sub_df = df_top[df_top['Multiple'] == mult]
            if sub_df.empty:
                continue

            top_peers_dict = {}
            for _, row in sub_df.iterrows():
                abbr = model_abbr.get(row['Model'], row['Model'])
                peers_str = row[f"Top {num_peers} Peers"]
                top_peers_dict[abbr] = set(peer.strip() for peer in peers_str.split(",") if peer.strip()) if peers_str != "No Data" else set()

            for m1 in top_peers_dict:
                for m2 in top_peers_dict:
                    overlap = len(top_peers_dict[m1].intersection(top_peers_dict[m2]))
                    aggregated[mult].loc[m1, m2] += overlap

    # Filter out empty matrices and compute color limits only from lower triangles
    lower_triangle_values = []
    for mult, matrix in aggregated.items():
        tril_mask = np.tril(np.ones_like(matrix.values, dtype=bool), k=-1)
        visible_values = matrix.where(tril_mask).stack()
        if not visible_values.empty:
            plotted_matrices[mult] = matrix
            lower_triangle_values.append(visible_values)

    if not lower_triangle_values:
        print("No valid heatmap data — skipping plots and colorbar.")
        return

    # Compute color scale
    all_visible = pd.concat(lower_triangle_values)
    vmin, vmax = all_visible.min(), all_visible.max()

    # Plot heatmaps with no colorbar
    for mult, matrix in plotted_matrices.items():
        mask = np.triu(np.ones_like(matrix.values, dtype=bool))
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            matrix,
            mask=mask,
            cmap="rocket_r",
            annot=True,
            fmt=".0f",
            annot_kws={"size": 20},
            linewidths=0.5,
            linecolor="white",
            square=True,
            cbar=False,
            vmin=vmin,
            vmax=vmax
        )
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"aggregated_heatmap_overlap_top{num_peers}_{mult}.png"), dpi=300, bbox_inches='tight')
        plt.show()

    # Save shared colorbar
    fig, ax = plt.subplots(figsize=(6, 1))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cb = ColorbarBase(ax, cmap="rocket_r", norm=norm, orientation='horizontal')
    cb.set_label("Peer Overlap Count", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"heatmap_colorbar_legend_top{num_peers}.png"), dpi=300, bbox_inches='tight')
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
                    weights = weights / total  # Normalize to sum to exactly one (this should already be the case)
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

    return mean_matrix, std_matrix, pair_results


def permutation_test_frobenius(n_permutations=1000, random_state=27, n_jobs=cores):
    """
    Global permutation test for normalized Frobenius norms
    """
    comb_keys = [f"{model}_{multiple}" for model in models for multiple in multiples]
    tasks = list(combinations(comb_keys, 2))
    pval_matrix = pd.DataFrame(np.nan, index=comb_keys, columns=comb_keys)

    for idx, (key1, key2) in enumerate(tqdm(tasks, desc="Model‑Pair Tests")):
        # Serial preload of peer weights
        model1, mult1 = key1.split("_", 1)
        model2, mult2 = key2.split("_", 1)
        peer1, peer2 = {}, {}
        for month in months:
            for fold in folds:
                df1 = load_peer_weights(model1, mult1, month, fold)
                if df1 is not None: peer1[(month, fold)] = df1
                df2 = load_peer_weights(model2, mult2, month, fold)
                if df2 is not None: peer2[(month, fold)] = df2

        # Build tasks_data list
        tasks_data = []
        for key, A_df in peer1.items():
            B_df = peer2.get(key)
            if B_df is None: continue
            idx_common = A_df.index.intersection(B_df.index)
            col_common = A_df.columns.intersection(B_df.columns)
            if idx_common.empty or col_common.empty: continue
            A = A_df.loc[idx_common, col_common].values
            B = B_df.loc[idx_common, col_common].values
            tasks_data.append((A, B))

        if not tasks_data:
            pval_matrix.loc[key2, key1] = np.nan
            continue

        # Pbserved distance
        obs_vals = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(frob_norm)(A, B) for A, B in tasks_data
        )
        obs_global = np.mean(obs_vals)

        # Permuted distance
        def single_perm(b):
            rng = np.random.default_rng(random_state + idx * n_permutations + b)
            permuted_means = []
            for A, B in tasks_data:
                perm = rng.permutation(A.shape[0])       # a permutation of row‑indices
                A_perm = A[perm, :]                     # reorder rows
                permuted_means.append(frob_norm(A_perm, B))
            return np.mean(permuted_means)

        perm_globals = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(single_perm)(b) for b in range(n_permutations)
        )

        # p‐value
        p_val = (1 + sum(pg >= obs_global for pg in perm_globals)) / (1 + n_permutations)
        pval_matrix.loc[key2, key1] = p_val

    return pval_matrix


# %% Execution
# Check gvkey consistency
check_gvkey_consistency()

# Example Test Firm and Month Selection for Plotting
test_firm_gvkey = "001690" # This is Apple Inc.
month = "2018-04"
fold = 3
plot_peer_weights(test_firm_gvkey, month, fold)

# Get and display the top 5 peers table for the specified test firm over all months
consolidated_top = get_consolidated_top_peers(test_firm_gvkey, 5)
print("\nOverall Consolidated Top 5 Peers:")
print(consolidated_top)

# Save the top peers table
output_table_path = os.path.join(RESULTS_DIR, "top5_peers_table.csv")
consolidated_top.to_csv(output_table_path, index=False)
print(f"\nTop 5 peers table saved to {output_table_path}")
    
# Plot aggregated heatmaps for top peers overlap
for num in [10, 20, 50]:
    print(f"\nPlotting heatmaps for Top {num} Peers")
    plot_heatmaps(test_firm_gvkey, num)

# Compute and display peer weight concentration metrics
print("\nComputing peer weight concentration metrics...")
concentration_results = compute_concentration_metrics()

# For each multiple, print the corresponding concentration table and save
for mult in multiples:
    print(f"\n=== Concentration Metrics for {mult.upper()} ===")
    print(concentration_results[mult])
    concentration_results[mult].to_csv(os.path.join(RESULTS_DIR, f"concentration_metrics_{mult}.csv"))

# Compute Frobenius norms
mean_matrix, std_matrix, pair_results = compute_frobenius_norms()
print("\n=== Frobenius Norms Between Peer Weight Matrices ===")
print(mean_matrix, std_matrix)

mean_matrix.to_csv(os.path.join(RESULTS_DIR, "frobenius_norms_mean_results.csv"))
std_matrix.to_csv(os.path.join(RESULTS_DIR, "frobenius_norms_std_results.csv"))

# Compute the permutation test p-values
pvals_df = permutation_test_frobenius(n_permutations=1000, random_state=27)
pvals_df.to_csv(os.path.join(RESULTS_DIR, "frobenius_permtest_pvalues.csv"))
