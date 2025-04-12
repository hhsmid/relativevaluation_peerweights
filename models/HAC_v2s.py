# %% Imports and Configuration
import numpy as np
import pandas as pd
import sqlite3
from sklearn.metrics import mean_squared_error
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import warnings
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# %% Global Parameters
# Set random seed 
SEED = 27

# Set number of cores
cores = 7


# %% Data loading and processing
print("Loading data...")

########## Load data from SQLite with parallelized chunks ##########
db_path = '/Data/sql_database.sqlite'

def fetch_chunk(offset, limit):
    try:
        conn = sqlite3.connect(db_path)
        query = f"""
            SELECT * 
            FROM final_merged_data
            LIMIT {limit} OFFSET {offset}
        """
        chunk = pd.read_sql_query(query, conn)
        conn.close()
        return chunk
    except Exception as e:
        print(f"Error fetching chunk at offset {offset}: {e}")
        return pd.DataFrame()

# Get total row count
with sqlite3.connect(db_path) as conn:
    row_count = pd.read_sql_query("SELECT COUNT(*) AS cnt FROM final_merged_data", conn)['cnt'].iloc[0]

# Set chunk size and parallelize loading
chunk_size = 30000

chunks = Parallel(n_jobs=cores)(
    delayed(fetch_chunk)(offset, chunk_size) for offset in range(0, row_count, chunk_size)
)

# Combine all chunks into a single DataFrame
data = pd.concat(chunks, ignore_index=True)
data = data.sort_values(["gvkey", "month"]).reset_index(drop=True)

# Drop rows missing target values
data = data.dropna(subset=['ln_m2b', 'ln_v2a', 'ln_v2s'])
    
# Convert 'month' to a datetime object
data['month'] = pd.to_datetime(data['month'], errors='coerce')

# Filter for valid months
data = data.dropna(subset=['month'])
unique_months_total = data['month'].unique()


########## Define predictors and targets ##########

# Define the predictors
predictors = [
    # Panel A: WRDS Financial Ratios
    'accrual_wr', 'aftret_eq_wr', 'aftret_equity_wr', 'aftret_invcapx_wr', 'at_turn_wr',
    'capital_ratio_wr', 'cash_conversion_wr', 'cash_debt_wr', 'cash_lt_wr', 'cash_ratio_wr',
    'cfm_wr', 'curr_debt_wr', 'curr_ratio_wr', 'de_ratio_wr', 'debt_assets_wr', 'debt_at_wr',
    'debt_capital_wr', 'debt_ebitda_wr', 'debt_invcap_wr', 'dltt_be_wr', 'dpr_wr',
    'efftax_wr', 'equity_invcap_wr', 'fcf_ocf_wr', 'gpm_wr', 'gprof_wr', 'int_debt_wr',
    'int_totdebt_wr', 'intcov_ratio_wr', 'intcov_wr', 'invt_act_wr', 'inv_turn_wr', 'lt_debt_wr',
    'lt_ppent_wr', 'npm_wr', 'ocf_lct_wr', 'opmad_wr', 'opmbd_wr', 'pay_turn_wr',
    'pretret_earnat_wr', 'pretret_noa_wr', 'profit_lct_wr', 'ptpm_wr', 'quick_ratio_wr',
    'rd_sale_wr', 'rect_act_wr', 'rect_turn_wr', 'roa_wr', 'roce_wr', 'roe_wr',
    'short_debt_wr', 'totdebt_invcap_wr',
    # Panel B: Accounting Anomalies
    'assetgrowth_an', 'assetturnover2_an', 'assetturnover_an', 'bookequitygrowth_an',
    'capexgrowth_an', 'cash2assets_an', 'cf2debt_an', 'chbe_an', 'chca_an', 'chceq_an',
    'chcl_an', 'chcurrentratio_an', 'chfnl_an', 'chlt_an', 'chnccl_an', 'debt2tang_an',
    'deprn_an', 'ebitda2revenue_an', 'grossmargin_an', 'grossprofit_an',
    'inventorychange_an', 'inventorygrowth_an', 'investment_an', 'liquid2assets_an',
    'marginch_an', 'opleverage_an', 'pchdeprn_an', 'pchgm2pchsale_an',
    'pchquickratio_an', 'pchsale2pchinvt_an', 'pchsale2pchrect_an',
    'pchsale2pchxsga_an', 'pchsales2inv_an', 'profitability_an', 'roic_an',
    'sales2cash_an', 'sales2inv_an', 'sales2rec_an', 'salesgrowth_an',
    # Panel C: Size Proxies
    'total_assets', 'book_equity', 'sales', 'total_debt',
    # Panel D: CAPM Beta
    'beta_monthly',
    # Panel E: FF49
    '1-Agriculture', '2-Food Products', '3-Candy & Soda', '4-Beer & Liquor', '5-Tobacco Products',
    '6-Recreation', '7-Entertainment', '8-Printing and Publishing', '9-Consumer Goods', '10-Apparel',
    '11-Healthcare', '12-Medical Equipment', '13-Pharmaceutical Products', '14-Chemicals',
    '15-Rubber and Plastic Products', '16-Textiles', '17-Construction Materials', '18-Construction',
    '19-Steel Works Etc', '20-Fabricated Products', '21-Machinery', '22-Electrical Equipment',
    '23-Automobiles and Trucks', '24-Aircraft', '25-Shipbuilding', '26-Defense', '27-Precious Metals',
    '28-Non-Metallic and Industrial Metal Mining', '29-Coal', '30-Petroleum and Natural Gas', '31-Utilities',
    '32-Communication', '33-Personal Services', '34-Business Services', '35-Computers',
    '36-Computer Software', '37-Electronic Equipment', '38-Measuring and Control Equipment',
    '39-Business Supplies', '40-Shipping Containers', '41-Transportation', '42-Wholesale', '43-Retail',
    '44-Restaurants', '45-Banking', '46-Insurance', '47-Real Estate', '48-Trading', '49-Almost Nothing or Missing'
]

# Ensure predictors variables are numeric
for col in predictors:
    if not np.issubdtype(data[col].dtype, np.number):
        print(f"Warning: Non-numeric column detected: {col}")

# Define the target variable
target_variable = ['ln_v2s']

print(f"\nData successfully processed: {data.shape[0]} rows, {data.shape[1]} columns, and {len(unique_months_total)} unique months.")


# %% Sample smaller subset
# Filter data for all months from January 1990 to December 2023
data = data[
    (data['month'].dt.year >= 1990) & 
    (data['month'].dt.year <= 2023)
]

# Filter small subset
#data = data[
#    (data['month'].dt.year == 2018) & 
#    (data['month'].dt.month.isin([1]))
#]

unique_months = data['month'].unique()

print(f"Filtered data for subsample: {len(unique_months)} unique months and {data.shape[0]} rows remaining.")


# %% Forward-filling and scaling
# Forward fill NAs in predictors on the firm level
data = data.sort_values(["gvkey", "month"]).copy()
data[predictors] = data.groupby("gvkey")[predictors].ffill()

# Fill remaining NAs using the average for that predictor across all firms in that month
data = data.copy()
data[predictors] = data.groupby("month")[predictors].transform(lambda x: x.fillna(x.mean()))
        
# Save an unscaled copy of total_debt and assets before scaling
data = pd.concat([
    data,
    data['net_debt'].rename('net_debt_unscaled'),
    data['sales'].rename('sales_unscaled')
], axis=1)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the entire dataset
data[predictors] = scaler.fit_transform(data[predictors])


# %% Utility functions
# Function to determine the optimal number of clusters for HAC clustering
def optimize_clusters_hac(train_set, val_set, predictors, max_clusters=None, iterations=5, n_jobs=cores):
    """
    Determines the optimal number of clusters for HAC clustering using validation RMSE.
    Candidate cluster numbers are randomly selected from the available range.

    Parameters:
    - train_set (pd.DataFrame): Training dataset containing firm data.
    - val_set (pd.DataFrame): Validation dataset containing firm data.
    - predictors (list): List of column names used as features for clustering.
    - target_variable (list or str): The target variable name (e.g. 'ln_v2s').
    - max_clusters (int, optional): Maximum number of clusters to test.
    - iterations (int, optional): Number of clustering trials.
    - n_jobs (int, optional): Number of parallel jobs.

    Returns:
    - tuple: (best_model, best_n_clusters, best_rmse)
    """
    # Prepare training features and validation actual market cap
    X_train = train_set[predictors].values
    y_val_actual = val_set.set_index('gvkey')['mktcap'].values
    
    # Set minimum clusters
    min_clusters = 2
    
    # Count unique rows (distinct firms in the training set)
    unique_data_points = np.unique(X_train, axis=0).shape[0]
    
    # Set the dynamic upper limit for clusters
    if max_clusters is None:
        max_clusters = max(2, unique_data_points // 3)
    else:
        max_clusters = min(max_clusters, unique_data_points)
    
    # Create a list of candidate cluster numbers
    available_clusters = list(range(min_clusters, max_clusters + 1))

    def evaluate_hac(iteration):
        try:
            if not available_clusters:
                return iteration, None, None, np.inf

            # Randomly select a candidate number of clusters from those available
            k = available_clusters.pop(np.random.randint(len(available_clusters)))
            
            # Fit the HAC model (deterministic once k is set)
            hac = AgglomerativeClustering(n_clusters=k, linkage='ward')
            hac.fit(X_train)
            
            # Compute peer weights on the validation set using the HAC model
            peer_weights = compute_peer_weights(train_set, val_set, hac, predictors)
            if peer_weights is None:
                return iteration, None, k, np.inf

            # Compute out-of-sample predictions using the peer weights
            y_val_hat, _ = compute_oos_predictions(train_set, val_set, peer_weights, target_variable)
            v2s_pred = np.exp(y_val_hat)
            
            # Calculate predicted market cap
            sales_val = val_set['sales_unscaled'].values
            net_debt_val = val_set['net_debt_unscaled'].values
            mktcap_pred = np.maximum(v2s_pred * sales_val - net_debt_val, 0)
            
            # Compute RMSE between predicted market cap and actual market cap
            val_rmse = np.sqrt(mean_squared_error(y_val_actual, mktcap_pred))
            return iteration, hac, k, val_rmse

        except Exception as e:
            print(f"Error encountered for {k} clusters: {e}")
            return iteration, None, k, np.inf

    # Run multiple iterations in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(evaluate_hac)(i) for i in range(iterations))
    results.sort(key=lambda x: x[0])
    
    # Report RMSE for each iteration
    for i, _, k, rmse in results:
        if k is not None:
            print(f"Iteration {i+1}: {k} clusters → Validation RMSE: {rmse:.5f}", flush=True)
    
    # Select the model with the lowest RMSE
    best_iteration, best_model, best_n_clusters, best_rmse = min(results, key=lambda x: x[3])
    print(f"Best model: {best_n_clusters} clusters (Validation RMSE: {best_rmse:.5f})\n", flush=True)
    
    return best_model, best_n_clusters, best_rmse


# Helper function to assign clusters to test data based on a KNN classifier trained on HAC clustering results
def assign_clusters_hac(train_data, test_data, predictors, hac_model):
    """
    Assign clusters to test data using a KNN classifier trained on the training data's HAC labels.
    
    Parameters:
    - train_data (pd.DataFrame): The training data with features used for clustering.
    - test_data (pd.DataFrame): The test data to assign clusters to.
    - predictors (list): List of column names used for clustering.
    - hac_model: The fitted HAC model (provides labels for training data).
    
    Returns:
    - pd.Series: Predicted cluster labels for test_data.
    """
    # Get training features and labels (from HAC)
    X_train = train_data[predictors]
    labels_train = hac_model.labels_
    
    # Initialize a 1-Nearest Neighbor classifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, labels_train)
    
    # Predict clusters for test data
    test_clusters = knn.predict(test_data[predictors])
    return pd.Series(test_clusters, index=test_data.index)


# Function to compute peer weights based on HAC cluster assignment
def compute_peer_weights(train_set, test_set, clustering_model, predictors):
    """
    Computes peer weights for each test firm based on HAC clustering similarity.

    Parameters:
    - train_set (pd.DataFrame): Training dataset containing firm data.
    - test_set (pd.DataFrame): Test dataset containing firm data.
    - clustering_model: A fitted HAC model from optimize_clusters.
    - predictors (list): List of column names used for clustering.

    Returns:
    - pd.DataFrame: A matrix of peer weights, where rows represent training firms 
                  and columns represent test firms.
    """
    train_set = train_set.copy()
    test_set = test_set.copy()
    
    train_set['cluster'] = clustering_model.labels_
    test_set['cluster'] = assign_clusters_hac(train_set, test_set, predictors, clustering_model)
    
    peer_weights = pd.DataFrame(0.0, index=train_set['gvkey'], columns=test_set['gvkey'])
    
    for cluster in train_set['cluster'].unique():
        cluster_train = train_set[train_set['cluster'] == cluster]
        cluster_test = test_set[test_set['cluster'] == cluster]
        
        if not cluster_test.empty:
            weight = 1 / len(cluster_train)
            peer_weights.loc[cluster_train['gvkey'], cluster_test['gvkey']] = weight
    
    col_sums = peer_weights.sum(axis=0)
    if not all(abs(col_sums - 1) < 1e-6):
        print("Warning: Some columns do not sum to 1.")
    
    return peer_weights


# Function to compute OOS predictions using medians and averages of the clusters
def compute_oos_predictions(train_set, test_set, peer_weights, target_variable):
    """
    Computes OOS target predictions using both peer-weighted average and median.
    
    Parameters:
    - train_set (pd.DataFrame): Training dataset containing firm data.
    - test_set (pd.DataFrame): Test dataset containing firm data.
    - peer_weights (pd.DataFrame): A matrix of peer weights where rows represent training firms 
                                   and columns represent test firms.
    - target_variable (str): The target variable name.
    
    Returns:
    - pd.Series: Peer-weighted average predictions.
    - pd.Series: Peer-median predictions.
    """
    y_train = train_set.set_index('gvkey')[target_variable]
    
    y_test_hat = peer_weights.T.dot(y_train.to_numpy())
    
    train_set = train_set.copy()
    train_set.loc[:, 'gvkey'] = train_set['gvkey'].astype(str)
    train_set = train_set.set_index('gvkey')
    
    peer_medians = {}
    for test_firm in test_set['gvkey'].astype(str):
        if test_firm in peer_weights.columns:
            peers = peer_weights[test_firm] > 0
            aligned_index = peer_weights.index.intersection(train_set.index)
            peer_group = train_set.loc[aligned_index, target_variable].squeeze()[peers.loc[aligned_index]]
        else:
            peer_group = pd.Series(dtype='float64')

        if not peer_group.empty:
            peer_medians[test_firm] = peer_group.median()
        else:
            peer_medians[test_firm] = y_train.median()
    
    return pd.Series(y_test_hat.values.flatten(), index=test_set['gvkey']), pd.Series(peer_medians, index=test_set['gvkey'])


# %% Start the loop over each month
# Results and valuation errors containers
results = []

for month in tqdm(unique_months, desc="Processing months", position=0, leave=True):
    month_data = data[data['month'] == month]
    
    # Find the number of unique firms
    unique_firms = month_data['gvkey'].nunique()
    print("\n")
    print(f"\n===== Processing Month: {month.strftime('%Y-%m')} | Unique Firms: {unique_firms} =====\n")

    # Check if there's enough data for splitting
    if month_data.shape[0] < 10:
        logger.warning(f"Skipping month {month}: insufficient data ({month_data.shape[0]} rows).")
        continue
    
    # Randomly shuffle observations
    month_data = month_data.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Partition the data into five blocks, each ~20% of the data
    n = len(month_data)
    indices = np.arange(n)
    splits = np.array_split(indices, 5)
    block_ids = np.empty(n, dtype=int)
    for i, split in enumerate(splits):
        block_ids[split] = i + 1
    month_data['block'] = block_ids

    # Results containers
    oos_predictions = []
    rmse_list = []
    r2_oos_list = []
    rmse_median_list = []
    r2_oos_median_list = []
    
    for fold in tqdm(range(1, 6), desc="Processing folds", position=1, leave=True):
        print(f"\nProcessing fold {fold}/5 of month {month.strftime('%Y-%m')}.")
            
        # Test set: the current test block
        test_set = month_data[month_data['block'] == fold]
    
        # Validation set: one of the remaining blocks at random
        remaining_blocks = month_data[month_data['block'] != fold]
        np.random.seed(SEED)
        val_block = np.random.choice(remaining_blocks['block'].unique(), 1)[0]
        val_set = remaining_blocks[remaining_blocks['block'] == val_block]
        
        # Training set: all remaining blocks except the selected validation block
        train_set = remaining_blocks[remaining_blocks['block'] != val_block]
    
        # Extract features and target for each set
        X_train, y_train = train_set[predictors], train_set[target_variable]
        X_val, y_val = val_set[predictors], val_set[target_variable]
        X_test, y_test = test_set[predictors], test_set[target_variable]
        
        # Dataset sizes
        N_train = X_train.shape[0]
        N_val = X_val.shape[0]
        N_test = X_test.shape[0]


        ########## HAC Model Training ##########

        # Optimize clusters for the current fold using HAC with Ward linkage
        clustering_model, best_n_clusters, _ = optimize_clusters_hac(train_set, val_set, predictors)
        print(f"   ---> Clustering completed for {month.strftime('%Y-%m')}, Fold {fold} with {best_n_clusters} clusters.")


        ########## Peer Weights Calculation ##########
        
        # Calculate clusters and peer weights
        peer_weights = compute_peer_weights(train_set, test_set, clustering_model, predictors)
        if peer_weights is None:
            print("Warning: Peer weights could not be computed. Skipping this iteration.")
            continue
        print("   ---> Peer weights calculated.", flush=True)
        
        # Ensure all test firms have assigned peer weights
        for test_firm in test_set['gvkey']:
            if peer_weights[test_firm].sum() == 0:
                print(f"Warning: Test firm {test_firm} has no peer weights. Assigning equal weight to all training firms.")
                peer_weights[test_firm] = 1 / len(train_set)
        
        # Save the test weight matrix
        W_path = f"/Results/HAC/v2s/Peer weights/peer_weights_{month.strftime('%Y-%m')}_fold{fold}.csv"
        gvkey_train = train_set['gvkey'].values
        gvkey_test = test_set['gvkey'].values
        W_df = pd.DataFrame(peer_weights, index=gvkey_train, columns=gvkey_test)
        W_df.to_csv(W_path)
        print("   ---> Peer weights saved.", flush=True)


        ########## Evaluate on Test Set ##########
        
        # Compute OOS predictions
        y_test_hat, y_test_medians_hat = compute_oos_predictions(train_set, test_set, peer_weights, target_variable)
        print("   ---> OOS predictions calculated.", flush=True)
        
        # Transform predictions back to `v2s`
        v2s_pred = np.exp(y_test_hat)
        v2s_median_pred = np.exp(y_test_medians_hat)
        
        # Calculate predicted market cap
        sales_test = test_set['sales_unscaled'].copy()
        net_debt_test = test_set['net_debt_unscaled'].copy()
        mktcap_pred = np.maximum(v2s_pred * sales_test.values - net_debt_test.values, 0)
        mktcap_median_pred = np.maximum(v2s_median_pred * sales_test.values - net_debt_test.values, 0)        
        mktcap_actual = test_set.set_index('gvkey')['mktcap']
    
        # Calculate metrics for weighted averages
        test_rmse = np.sqrt(mean_squared_error(mktcap_actual, mktcap_pred))
        r2_oos = 1 - (np.sum((mktcap_actual - mktcap_pred) ** 2) / np.sum((mktcap_actual - mktcap_actual.mean()) ** 2))
        rmse_list.append(test_rmse)
        r2_oos_list.append(r2_oos)
    
        # Calculate metrics for median-based predictions
        test_rmse_median = np.sqrt(mean_squared_error(mktcap_actual, mktcap_median_pred))
        r2_oos_median = 1 - (np.sum((mktcap_actual - mktcap_median_pred) ** 2) / np.sum((mktcap_actual - mktcap_actual.mean()) ** 2))
        rmse_median_list.append(test_rmse_median)
        r2_oos_median_list.append(r2_oos_median)
    
        # Calculate percentage error
        percentage_error = (mktcap_pred / mktcap_actual.values) - 1
        percentage_error_median = (mktcap_median_pred / mktcap_actual.values) - 1
        
        # Save outcomes for the test set
        fold_outcomes = pd.DataFrame({
            'fold': fold,
            'month': month,
            'gvkey': test_set['gvkey'].values,
            'mktcap_actual': mktcap_actual.values,
            'mktcap_pred': mktcap_pred,
            'mktcap_median_pred': mktcap_median_pred,
            'percentage_error': percentage_error,
            'percentage_error_median': percentage_error_median
        })
        oos_predictions.append(fold_outcomes)
        
        # Save fold outcomes for each firm (gvkey) for the current fold
        fold_outcomes_path = f"/Results/HAC/v2s/Outcomes/outcomes_{month.strftime('%Y-%m')}_fold{fold}.csv"
        fold_outcomes.set_index('gvkey', inplace=True)
        fold_outcomes.to_csv(fold_outcomes_path)
        print("   ---> Fold outcomes saved.", flush=True)
    
    
    ########## Aggregate Results for the Month ##########
    
    month_med_rmse = np.median(rmse_list)
    month_med_r2 = np.median(r2_oos_list)
    month_med_rmse_median = np.median(rmse_median_list)
    month_med_r2_median = np.median(r2_oos_median_list)
        
    # Results
    results.append({
        "month": month,
        "med_rmse": month_med_rmse,
        "med_r2_oos": month_med_r2,
        "med_rmse_median": month_med_rmse_median,
        "med_r2_oos_median": month_med_r2_median,
    })
    print(f"\n===== Results for {month.strftime('%Y-%m')}: Median RMSE={month_med_rmse:.3f}, Median R²={month_med_r2:.3f} =====\n", flush=True)

# Convert the results to a DataFrame
results = pd.DataFrame(results)

# Save the results
results.to_csv("/Results/HAC/v2s/results.csv", index=False)
      
