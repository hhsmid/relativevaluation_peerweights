# %% Imports and Configuration
import numpy as np
import pandas as pd
import sqlite3
import lightgbm as lgb
from lightgbm.callback import early_stopping
from sklearn.metrics import mean_squared_error
from numba import jit
from joblib import Parallel, delayed
import logging
import warnings
from tqdm import tqdm
import gc

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

# Set GBM parameters
LEARNING_RATE = 0.1
LEAVES = 31
EARLY_STOPPING_ROUNDS = 30


# %% Data loading and processing
print("Loading data...")

########## Load data from SQLite with parallelized chunks ##########
db_path = '/Users/hhsmid/Desktop/Hugo EUR/0. Master Thesis/03. Data/sql_database.sqlite'

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
        return pd.DataFrame()  # Return an empty DataFrame on failure

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
target_variable = ['ln_v2a']

print(f"\nData successfully processed: {data.shape[0]} rows, {data.shape[1]} columns, and {len(unique_months_total)} unique months.")


# %% Sample smaller subset
# Filter data for all months from January 1990 to December 2023
data = data[
    (data['month'].dt.year >= 1990) & 
    (data['month'].dt.year <= 2023)
]

# Filter small subset of first three months of 2018
#data = data[
#    (data['month'].dt.year == 2018) & 
#    (data['month'].dt.month.isin([1, 2, 3]))
#]

unique_months = data['month'].unique()

print(f"Filtered data for subsample: {len(unique_months)} unique months and {data.shape[0]} rows remaining.")


# %% Utility functions
def tree_prediction(model, data, train_sample_average, tree_iteration, learning_rate):
    """
    Extracts individual tree predictions from a LightGBM model.

    Parameters:
        model: LightGBM model object
        data: Input data
        train_sample_average: Mean of training labels
        tree_iteration: Index of the tree to extract
        learning_rate: Learning rate of the model

    Returns:
        tree_prediction: Predictions from the specified tree
    """
    if tree_iteration == 0:
        tree_prediction = train_sample_average.values[0]
    elif tree_iteration == 1:
        gbm1_prediction = model.predict(data, start_iteration=0, num_iteration=1)
        tree_prediction = (gbm1_prediction - train_sample_average.values[0]) / learning_rate
    else:
        tree_prediction = model.predict(data, start_iteration=tree_iteration-1, num_iteration=1) / learning_rate
    return tree_prediction


@jit(nopython=True)
def getD(tree):
    """
    Generates a leaf membership matrix.
    
    Parameters:
        tree: Array of leaf memberships

    Returns:
        D: Membership matrix where D[f, c] = 1 if observations f and c are in the same leaf, else 0
    """
    tree = np.asarray(tree)
    return (tree[:, None] == tree[None, :]).astype(np.float32)


def compute_peer_weights_gbm(model, X_train, X_test, y_train, learning_rate):
    """
    Computes the peer weight matrix.

    Parameters:
        model: Trained LightGBM model
        X_train: Training data
        X_test: Test data
        y_train: Training labels
        learning_rate: Learning rate of the model

    Returns:
        W: nxS optimized peer weight matrix for test samples
    """
    n_train = X_train.shape[0]
    S_test = X_test.shape[0]
    M = model.best_iteration + 1

    I = np.identity(n_train)

    # Extract instance leaf membership for train and test
    instance_leaf_membership_train = model.predict(X_train, pred_leaf=True)
    instance_leaf_membership_test = model.predict(X_test, pred_leaf=True)

    train_average = y_train.mean()

    # Extract tree predictions
    tree = [np.full(n_train, train_average)] 
    tree_results = Parallel(n_jobs=cores)(
        delayed(tree_prediction)(model, X_train, train_average, idx, learning_rate)
        for idx in range(1, M)
    )
    tree.extend([t.astype(np.float32) for t in tree_results])

    # Validate Ensemble Predictions
    ensemble = tree[0]
    for index in range(1, M):
        ensemble = ensemble + learning_rate * tree[index]

    y_hat = model.predict(X_train)
    diff1 = ensemble - y_hat
    assert np.allclose(diff1, np.zeros_like(diff1), atol=1e-6), "Mismatch in ensemble predictions"
    print("   ---> Ensemble predictions validated.")

    # Validate Membership Matrices
    myD1 = getD(tree[1])
    D1 = getD(instance_leaf_membership_train[:, 0])
    diff2 = myD1 - D1
    assert np.allclose(diff2, np.zeros_like(diff2), atol=1e-3), "Mismatch in membership matrices"
    print("   ---> Membership matrices validated.")

    # Initialize final weight matrices
    W_train = np.zeros((n_train, n_train), dtype=np.float32)
    W = np.zeros((n_train, S_test), dtype=np.float32)

    # First tree weights
    L = np.ones((n_train, n_train), dtype=np.float32)
    L_test = np.ones((n_train, S_test), dtype=np.float32)
    P = np.ones((n_train, n_train), dtype=np.float32) / n_train
    W_train += P.T @ (L / (np.sum(L, axis=0, keepdims=True) + 1e-10))
    W += P.T @ (L_test / (np.sum(L_test, axis=0, keepdims=True) + 1e-10))
    G = P.copy()
    v = y_train.values.reshape(n_train, 1)

    # Iterative weights
    for i in range(1, M):
        D = getD(tree[i]).astype(np.float32)
        D_sum = np.sum(D, axis=0, keepdims=True) + 1e-10
        V = D / D_sum

        P = learning_rate * (V @ (I - G))
        G = G + P

        L = D
        del D, V
        gc.collect()

        # Test leaf membership
        L_test = (instance_leaf_membership_train[:, i - 1][:, np.newaxis] ==
                  instance_leaf_membership_test[:, i - 1][np.newaxis, :]).astype(np.float32)
        L_test /= (np.sum(L_test, axis=0, keepdims=True) + 1e-10)

        W_train += P.T @ (L / (np.sum(L, axis=0, keepdims=True) + 1e-10))
        W += P.T @ (L_test / (np.sum(L_test, axis=0, keepdims=True) + 1e-10))

        del P, L_test
        gc.collect()

    # Final prediction with peer weights
    k = v.T @ W_train
    k_test = y_train.values.T @ W
    print("   ---> Peer weights iterations completed.")

    # Validate W_train predictions
    diff_pred = k - y_hat
    assert np.allclose(diff_pred, np.zeros((1, n_train)), atol=1e-5), "Mismatch in GBM predictions"
    print("   ---> GBM predictions validated.")

    # Validate W predictions
    y_test_hat = model.predict(X_test)
    diff_pred = k_test - y_test_hat
    assert np.allclose(diff_pred, np.zeros((1, S_test)), atol=1e-5), "Mismatch in out-of-sample GBM predictions!"
    print("   ---> Out-of-sample GBM predictions validated.")

    return W


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
        

        ########## LightGBM Model Training ##########
        
        # Parameters
        params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": LEAVES,
            "learning_rate": LEARNING_RATE,
            "verbose": -1,
            "min_data": 2,
            "num_threads": cores
        }
        
        # Create datasets for LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model with early stopping using callback
        model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)]
        )


        ########## Evaluate on Test Set ##########
        
        # Generate OOS predictions for the test set
        y_test_hat = model.predict(X_test)

        # Transform predictions back to `v2a`
        v2a_pred = np.exp(y_test_hat)
        
        # Calculate predicted market cap
        assets_test = test_set['total_assets']
        net_debt_test = test_set['net_debt']
        mktcap_pred = np.maximum(v2a_pred * assets_test.values - net_debt_test.values, 0)
        mktcap_actual = test_set['mktcap']

        # Calculate metrics
        test_rmse = np.sqrt(mean_squared_error(mktcap_actual, mktcap_pred))
        r2_oos = 1 - (np.sum((mktcap_actual - mktcap_pred) ** 2) / np.sum((mktcap_actual - mktcap_actual.mean()) ** 2))
        rmse_list.append(test_rmse)
        r2_oos_list.append(r2_oos)

        # Calculate percentage error
        percentage_error = (mktcap_pred / mktcap_actual.values) - 1
        
        # Save outcomes for the test set
        fold_outcomes = pd.DataFrame({
            'fold': fold,
            'month': month,
            'gvkey': test_set['gvkey'].values,
            'mktcap_actual': mktcap_actual.values,
            'mktcap_pred': mktcap_pred,
            'percentage_error': percentage_error
        })
        oos_predictions.append(fold_outcomes)
        
        # Save fold outcomes for each firm (gvkey) for the current fold
        fold_outcomes_path = f"/Volumes/HugoHardDrive/05. Results/GBM/v2a/Outcomes/outcomes_{month.strftime('%Y-%m')}_fold{fold}.csv"
        fold_outcomes.set_index('gvkey', inplace=True)
        fold_outcomes.to_csv(fold_outcomes_path)
        print("   ---> Fold outcomes saved.")


        ########## Peer Weights Calculation ##########
        
        W = compute_peer_weights_gbm(model, X_train, X_test, y_train, LEARNING_RATE)

        # Save the W matrix
        W_path = f"/Volumes/HugoHardDrive/05. Results/GBM/v2a/Peer weights/peer_weights_{month.strftime('%Y-%m')}_fold{fold}.csv"
        gvkey_train = train_set['gvkey'].values
        gvkey_test = test_set['gvkey'].values
        W_df = pd.DataFrame(W, index=gvkey_train, columns=gvkey_test)
        W_df.to_csv(W_path)
        print("   ---> Peer weights saved.", flush=True)
         
        
    ########## Aggregate Results for the Month ##########
    
    month_med_rmse = np.median(rmse_list)
    month_med_r2 = np.median(r2_oos_list)
        
    # Results
    results.append({
        "month": month,
        "med_rmse": month_med_rmse,
        "med_r2_oos": month_med_r2,
    })
    print(f"\n===== Results for {month.strftime('%Y-%m')}: Median RMSE={month_med_rmse:.3f}, Median R²={month_med_r2:.3f} =====\n", flush=True)

# Convert the results to a DataFrame
results = pd.DataFrame(results)

# Save the results
results.to_csv("/Volumes/HugoHardDrive/05. Results/GBM/v2a/results.csv", index=False)

