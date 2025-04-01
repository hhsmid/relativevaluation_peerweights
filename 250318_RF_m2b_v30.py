# %% Imports and Configuration
import numpy as np
import pandas as pd
import sqlite3
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import logging
import warnings
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor

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

# RF hyperparameters
n_estimators_max = 300
early_stopping_rounds = 30


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
target_variable = ['ln_m2b']

print(f"\nData successfully processed: {data.shape[0]} rows, {data.shape[1]} columns, and {len(unique_months_total)} unique months.")


# %% Sample smaller subset
# Filter data for all months from January 1990 to December 2023
data = data[
    (data['month'].dt.year >= 1993) & 
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
data[predictors] = data.groupby("month")[predictors].transform(lambda x: x.fillna(x.median()))


# %% Utility Functions
def optimize_model_rf(X_train, y_train, X_val, y_val, n_estimators_max=n_estimators_max, early_stopping_rounds=early_stopping_rounds):
    """
    Trains a Random Forest Regressor with early stopping based on validation RMSE.

    Parameters:
    - X_train, y_train: Training features and target.
    - X_val, y_val: Validation features and target.
    - n_estimators_max: Maximum number of trees.
    - early_stopping_rounds: Number of rounds without improvement before stopping.
    - seed: Random seed for reproducibility.
    - cores: Number of CPU cores for parallel training.

    Returns:
    - final_model: Trained RandomForestRegressor with optimized n_estimators.
    - best_n_estimators: Optimal number of trees.
    """

    best_n_estimators = None
    best_val_rmse = float("inf")
    no_improvement_count = 0
    
    # Initialize model
    model = RandomForestRegressor(
        n_estimators=50,
        random_state=SEED,
        n_jobs=cores,
        warm_start=True
    )
    
    for n_estimators in range(1, n_estimators_max + 1):
        model.n_estimators = n_estimators
        model.fit(X_train, y_train.values.ravel())
    
        # Predict on the validation set
        y_val_pred = model.predict(X_val)
        
        # Compute validation RMSE
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        # Check for improvement
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_n_estimators = n_estimators
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Early stopping condition
        if no_improvement_count >= early_stopping_rounds:
            print(f"\nEarly stopping at n_estimators={n_estimators} (Best RMSE={best_val_rmse:.4f})", flush=True)
            break
    
    # Retrain final model with optimal n_estimators
    final_model = RandomForestRegressor(
        n_estimators=best_n_estimators,
        random_state=SEED,
        n_jobs=cores
    )
    
    final_model.fit(X_train, y_train.values.ravel())

    print(f"Optimized RF Model trained with {best_n_estimators} estimators (Best RMSE={best_val_rmse:.4f})\n", flush=True)

    return final_model, best_n_estimators


def compute_peer_weights_rf(model, X_train, X_test, y_train):
    """
    Computes the n x S peer weight matrix for a Random Forest.
    
    Parameters:
        model : Trained RandomForestRegressor model (with bootstrap=True)
        X_train : Training features (DataFrame or array)
        X_test  : Test features (DataFrame or array)
        y_train : Training targets (provided for prediction check)
    
    Returns:
        W : A NumPy array of shape (n, S) containing the peer weights.
    """
    X_test_df = X_test.copy()
    
    # Ensure input arrays are numpy arrays
    X_train = np.asarray(X_train)
    X_test  = np.asarray(X_test)
    
    n = X_train.shape[0]
    S = X_test.shape[0]
    M = len(model.estimators_)
    
    # Initialize the peer weight matrix with zeros
    W = np.zeros((n, S))
    
    # Loop over each tree in the forest
    for m, tree in enumerate(model.estimators_):
        bootstrap_indices = model.estimators_samples_[m]
        v = np.bincount(bootstrap_indices, minlength=n)
    
        leaf_train = tree.apply(X_train)
        leaf_test = tree.apply(X_test)
        
        zero_denom_count = 0
        
        # Loop over each test observation s
        for s in range(S):
            leaf_id = leaf_test[s]
            in_leaf = (leaf_train == leaf_id)
            numerator = v * in_leaf
            denom = np.sum(numerator)
            
            if denom > 0:
                W[:, s] += (numerator / denom) / M
            else:
                # Increase count and print details for a few cases
                zero_denom_count += 1
                if zero_denom_count <= 5:
                    print(f"DEBUG: Tree {m}, Test Sample {s}: leaf_id={leaf_id}, "
                          f"Count in leaf={np.sum(in_leaf)}, v[in_leaf]={v[in_leaf] if np.sum(in_leaf)>0 else '[]'}, "
                          f"denom={denom}")
                W[:, s] += 0.0
        
        if zero_denom_count > 0:
            print(f"DEBUG: Tree {m} had {zero_denom_count} out of {S} test samples with denominator 0.")
    
    # Check 1: Check that each column sums to one
    col_sums = np.sum(W, axis=0)
    if not np.allclose(col_sums, np.ones(S), atol=1e-6):
        print("Error: Peer weights columns do not sum to one within tolerance. Column sums: {}".format(col_sums))
    
    # Check 2: Verify that the predictions from the peer weights match those of the Random Forest.
    y_train_arr = np.array(y_train).flatten()
    peer_preds = np.dot(W.T, y_train_arr)
    model_preds = model.predict(X_test_df)
    if not np.allclose(peer_preds, model_preds, atol=1e-6):
        print("Error: Predictions from peer weights do not match model predictions within tolerance. "
              "Max difference: {:.6f}".format(np.max(np.abs(peer_preds - model_preds))))
    
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


        ########## Random Forest training ##########
        
        # Train optimized model
        model, best_n_estimators = optimize_model_rf(X_train, y_train, X_val, y_val, n_estimators_max, early_stopping_rounds)


        ########## Evaluate on Test Set ##########
        
        # Generate OOS predictions for the test set
        y_test_hat = model.predict(X_test)

        # Transform predictions back to `m2b`
        m2b_pred = np.exp(y_test_hat)
        
        # Calculate predicted market cap
        book_equity_test = test_set['book_equity']
        mktcap_pred = np.maximum(m2b_pred * book_equity_test.values, 0)
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
        fold_outcomes_path = f"/Volumes/HugoHardDrive/05. Results/RF/m2b/Outcomes/outcomes_{month.strftime('%Y-%m')}_fold{fold}.csv"
        fold_outcomes.set_index('gvkey', inplace=True)
        fold_outcomes.to_csv(fold_outcomes_path)
        print("   ---> Fold outcomes saved.")


        ########## Peer Weights Calculation ##########
        
        W = compute_peer_weights_rf(model, X_train, X_test, y_train)

        # Save the W matrix
        W_path = f"/Volumes/HugoHardDrive/05. Results/RF/m2b/Peer weights/peer_weights_{month.strftime('%Y-%m')}_fold{fold}.csv"
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
results.to_csv("/Volumes/HugoHardDrive/05. Results/RF/m2b/results.csv", index=False)
    
