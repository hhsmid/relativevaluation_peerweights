# %% Import required libraries
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import sqlite3
import pandas_datareader as pdr
from statsmodels.regression.rolling import RollingOLS
from joblib import Parallel, delayed
from itertools import product
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis


# %% Suppress warnings
# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# %% Login to WRDS
# Load credentials from a .env file located at the specified path.
load_dotenv('WRDS_credentials.env')

# Get WRDS_USER and WRDS_PASSWORD
wrds_user = os.getenv('WRDS_USER')
wrds_password = os.getenv('WRDS_PASSWORD')

# Check if credentials are missing; raise an error if either is not found.
if wrds_user is None or wrds_password is None:
    raise ValueError(
        "WRDS credentials not loaded properly. Check your .env file."
    )

# Create a connection string for the WRDS database using the PostgreSQL dialect (psycopg2).
connection_string = f"postgresql+psycopg2://{wrds_user}:{wrds_password}@wrds-pgdata.wharton.upenn.edu:9737/wrds"

# Create an SQLAlchemy engine object, which manages the database connection.
wrds = create_engine(connection_string, pool_pre_ping=True)

# Try WRDS connection
try:
    conn = wrds.connect()
    print("Connection to WRDS is successful.")
except Exception as e:
    print("Error connecting to WRDS:", e)


# %% Create and manage an SQLite database
# Connect to a local SQLite database using the specified file path.
sql_database = sqlite3.connect(
    '/Data/sql_database.sqlite')

# Fetch all table names
tables = sql_database.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
table_names = [table[0] for table in tables]

# Drop each table
for table_name in table_names:
    sql_database.execute(f"DROP TABLE IF EXISTS {table_name};")
    print(f"Table {table_name} has been dropped.")

# Commit and close to finalize the deletion, then reopen for the main script
sql_database.commit()
sql_database.close()
print("All tables cleared from SQLite database.")

# Reopen SQLite connection for the main script
sql_database = sqlite3.connect(
    '/Data/sql_database.sqlite')


# %% Set start date and end date
start_date = '1980-01-01'
end_date = '2023-12-31'


# %% WRDS ratios query
# Query to retrieve the specific WRDS financial ratios
wrds_ratios_query = f"""
SELECT 
    public_date, gvkey, permno, accrual, aftret_eq, aftret_equity, aftret_invcapx, at_turn, 
    capital_ratio, cash_conversion, cash_debt, cash_lt, cash_ratio, cfm, curr_debt, 
    curr_ratio, de_ratio, debt_assets, debt_at, debt_capital, debt_ebitda, debt_invcap, 
    dltt_be, dpr, efftax, equity_invcap, fcf_ocf, gpm, gprof, int_debt, int_totdebt, 
    intcov_ratio, intcov, invt_act, inv_turn, lt_debt, lt_ppent, npm, ocf_lct, opmad, 
    opmbd, pay_turn, pretret_earnat, pretret_noa, profit_lct, ptpm, quick_ratio, rd_sale,
    rect_act, rect_turn, roa, roce, roe, short_debt, totdebt_invcap, ffi49, ffi49_desc, 
    ffi10, ffi10_desc
FROM wrdsapps.firm_ratio
WHERE public_date BETWEEN '{start_date}' AND '{end_date}'
"""

# Fetch the WRDS financial ratios
wrds_ratios_raw = pd.read_sql_query(
    sql=wrds_ratios_query,
    con=wrds,
    dtype={"gvkey": str},
    parse_dates=['public_date'])

# Create a copy of the raw data for reference
wrds_ratios = wrds_ratios_raw.copy()

print("WRDS ratios query successfully executed")


# %% WRDS ratios processing
########## Change date to month ##########

# Convert public_date to month-level granularity and rename it to 'month'
wrds_ratios['month'] = wrds_ratios['public_date'].dt.to_period('M').dt.to_timestamp()

# Drop the original 'public_date' column
wrds_ratios.drop(columns=['public_date'], inplace=True)


########## Process and define FF49 and FF10 industries ##########

# Ensure ffi49 is numeric and clean any string formatting
try:
    # Convert to numeric (float if necessary), then cast to int
    wrds_ratios['ffi49'] = pd.to_numeric(wrds_ratios['ffi49'], errors='coerce').fillna(49).astype(int)
except Exception as e:
    print(f"Error converting ffi49 to int: {e}")

# Fama French 49 industries
ff49 = [
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

# Create mapping for ffi49
ff49_mapping = {i: name for i, name in enumerate(ff49, start=1)}

# Map ffi49 to descriptive names for one-hot encoding
wrds_ratios['ffi49_desc'] = wrds_ratios['ffi49'].map(ff49_mapping)

# One-hot encode ffi49_desc
ffi49_encoded = pd.get_dummies(wrds_ratios['ffi49_desc'], prefix='', prefix_sep='')

# Ensure all one-hot encoded columns are numeric
ffi49_encoded = ffi49_encoded.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0).astype(int)

# Merge one-hot encoded columns with the main DataFrame
wrds_ratios = pd.concat([wrds_ratios, ffi49_encoded], axis=1)

# Drop the original ffi49_desc columns
wrds_ratios.drop(columns=['ffi49_desc'], inplace=True)

# Ensure ffi10 is numeric and handle any non-numeric entries
try:
    # Convert to numeric, coerce errors to NaN, fill NaN with a default value (e.g., 10), and cast to int
    wrds_ratios['ffi10'] = pd.to_numeric(wrds_ratios['ffi10'], errors='coerce').fillna(10).astype(int)
except Exception as e:
    print(f"Error converting ffi10 to int: {e}")

# Define Fama-French 10 industry names
ff10 = [
    '1-Consumer NonDurables',
    '2-Consumer Durables',
    '3-Manufacturing',
    '4-Energy',
    '5-Chemicals',
    '6-Business Equipment',
    '7-Telecommunication',
    '8-Healthcare',
    '9-Utilities',
    '10-Other'
]

# Create mapping for ffi10
ff10_mapping = {i: name for i, name in enumerate(ff10, start=1)}

# Map ffi10 to descriptive names
wrds_ratios['ffi10_desc'] = wrds_ratios['ffi10'].map(ff10_mapping)

# One-hot encode ffi10_desc
ffi10_encoded = pd.get_dummies(wrds_ratios['ffi10_desc'], prefix='', prefix_sep='')

# Ensure all one-hot encoded columns are numeric
ffi10_encoded = ffi10_encoded.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0).astype(int)

# Merge one-hot encoded columns with the main DataFrame
wrds_ratios = pd.concat([wrds_ratios, ffi10_encoded], axis=1)

# Drop the original ffi10 and ffi10_desc columns if they are no longer needed
wrds_ratios.drop(columns=['ffi10', 'ffi10_desc'], inplace=True)


########## Add _wr suffix ##########

# List of columns to exclude from renaming
exclude_from_wr_columns = ['gvkey', 'permno', 'month', 'ffi49'] + ff49 + ff10

# Create a dictionary for renaming columns (adding _wr suffix)
rename_dict = {
    col: f"{col}_wr" for col in wrds_ratios.columns if col not in exclude_from_wr_columns}

# Rename the columns using the dictionary
wrds_ratios.rename(columns=rename_dict, inplace=True)


########## Clean the WRDS ratios from outliers ##########

def create_boxplot(df, columns):
    """
    Generate separate boxplots for each column in the list.
    
    Parameters:
    df : DataFrame
        The dataframe containing the data.
    columns : list of str
        List of column names to visualize with boxplots.
    """
    for column in columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.show()

# Exclude 'public_date', 'gvkey', 'permno', 'month', and ff49 / ff10 from summary stats
variables_to_exclude_wrds = ['gvkey', 'permno', 'month'] + ff49 + ff10
numeric_columns_wrds = wrds_ratios.drop(columns=variables_to_exclude_wrds)

# Generate boxplots for all columns
# create_boxplot(wrds_ratios, numeric_columns_wrds.columns)

# Replacing outliers with NaN based on boxplots
wrds_ratios.loc[wrds_ratios['cash_conversion_wr']
                > 1e18, 'cash_conversion_wr'] = np.nan
wrds_ratios.loc[wrds_ratios['cfm_wr'] < -200000, 'cfm_wr'] = np.nan
wrds_ratios.loc[wrds_ratios['de_ratio_wr'] > 0.4e6, 'de_ratio_wr'] = np.nan
wrds_ratios.loc[wrds_ratios['debt_ebitda_wr']
                > 0.5e18, 'debt_ebitda_wr'] = np.nan
wrds_ratios.loc[wrds_ratios['dpr_wr'] > 0.1e16, 'dpr_wr'] = np.nan
wrds_ratios.loc[wrds_ratios['fcf_ocf_wr'] < -0.2e17, 'fcf_ocf_wr'] = np.nan
wrds_ratios.loc[wrds_ratios['intcov_ratio_wr']
                < -0.5e17, 'intcov_ratio_wr'] = np.nan
wrds_ratios.loc[wrds_ratios['intcov_wr'] < -0.5e17, 'intcov_wr'] = np.nan
wrds_ratios.loc[wrds_ratios['npm_wr'] < -0.26e16, 'npm_wr'] = np.nan
wrds_ratios.loc[wrds_ratios['opmad_wr'] < -0.26e16, 'opmad_wr'] = np.nan
wrds_ratios.loc[wrds_ratios['opmbd_wr'] < -0.26e16, 'opmbd_wr'] = np.nan
wrds_ratios.loc[wrds_ratios['pretret_noa_wr']
                > 1.0e16, 'pretret_noa_wr'] = np.nan
wrds_ratios.loc[wrds_ratios['pretret_noa_wr']
                < -0.5e14, 'pretret_noa_wr'] = np.nan
wrds_ratios.loc[wrds_ratios['ptpm_wr'] < -0.5e16, 'ptpm_wr'] = np.nan

# List of columns to visualize based on summary statistics
columns_to_plot_wrds = ['cash_conversion_wr', 'cfm_wr', 'de_ratio_wr', 'debt_ebitda_wr',
                        'dpr_wr', 'fcf_ocf_wr', 'intcov_ratio_wr', 'intcov_wr', 'inv_turn_wr',
                        'lt_ppent_wr', 'npm_wr', 'opmad_wr', 'opmbd_wr', 'pretret_noa_wr', 'ptpm_wr']

# Generate boxplots for cleaned columns
# create_boxplot(wrds_ratios, columns_to_plot_wrds)


########## Check for month-firm duplicates ##########

# Check for duplicate combinations of 'month' and 'gvkey'
duplicates_wrds = wrds_ratios[wrds_ratios.duplicated(
    subset=['month', 'gvkey'], keep=False)]
if not duplicates_wrds.empty:
    print(
        f"Duplicate entries found for 'month' and 'gvkey': {len(duplicates_wrds)} duplicates.")
else:
    print("No duplicate entries for 'month' and 'gvkey'.")

# Check for unique combinations of 'month' and 'gvkey'
unique_combinations_wrds = wrds_ratios.drop_duplicates(
    subset=['month', 'gvkey'])

# Count the number of unique months, unique gvkeys, and unique month-gvkey combinations
num_unique_months_wrds = wrds_ratios['month'].nunique()
num_unique_gvkeys_wrds = wrds_ratios['gvkey'].nunique()
num_unique_combinations_wrds = unique_combinations_wrds.shape[0]

print(f"Number of unique months: {num_unique_months_wrds}")
print(f"Number of unique gvkeys: {num_unique_gvkeys_wrds}")
print(f"Number of unique month-gvkey combinations: {num_unique_combinations_wrds}")


########## Summary statistics ##########

# Exclude 'public_date', 'gvkey', 'permno', 'month', and 'ffi49' from summary stats
variables_to_exclude_wrds = ['gvkey', 'permno', 'month']
numeric_columns_wrds = wrds_ratios.drop(columns=variables_to_exclude_wrds)

# Generate summary statistics for numeric columns
summary_stats_numeric_wrds = numeric_columns_wrds.describe()

print("WRDS ratios processing successfully executed.")


# %% Save WRDS ratios
# Store WRDS ratios data into SQLite
wrds_ratios.to_sql(name='wrds_ratios',
                   con=sql_database, if_exists='replace', index=False)

print("WRDS ratios successfully stored in the SQLite database.")


# %% Compustat query
# Query to retrieve Compustat Fundamentals Quarterly
compustat_query = f"""
SELECT 
    gvkey, datadate, atq, saleq, ceqq, capxy, cheq, ltq, dlcq, dlttq, dpq, oibdpq, 
    revtq, cogsq, invtq, ibq, rectq, lctq, actq, xsgaq, ppentq, teqq, nopiq, pstkq, 
    xintq, seqq, txdbq, txditcq, itccy, pstkrq, chq, mibtq
FROM 
    comp.fundq
WHERE 
    indfmt = 'INDL' AND datafmt = 'STD' AND consol = 'C'
    AND datadate BETWEEN '{start_date}' AND '{end_date}'
    AND atq > 0;
"""

# Fetch Compustat data
compustat_data_raw = pd.read_sql_query(
    sql=compustat_query,
    con=wrds,
    dtype={"gvkey": str},
    parse_dates={"datadate"}
)

# Create a copy of the raw data for reference
compustat_data = compustat_data_raw.copy()

print("Compustat query successfully executed.")


# %% Compustat processing
########## Change date to month ##########

# Convert datadate to month-level granularity and rename it to 'month'
compustat_data['month'] = compustat_data['datadate'].dt.to_period('M').dt.to_timestamp()

# Drop the original 'datadate' column
compustat_data.drop(columns=['datadate'], inplace=True)


########## Month-firm duplicates cleaning ##########

# Identify duplicate combinations of 'month' and 'gvkey'
duplicates_compustat = compustat_data[compustat_data.duplicated(subset=['month', 'gvkey'], keep=False)]

if not duplicates_compustat.empty:
    print(f"Duplicate entries found for 'month' and 'gvkey': {len(duplicates_compustat)} duplicates.")
else:
    print("No duplicate entries for 'month' and 'gvkey'.")

# Group by 'month' and 'gvkey' and check if all other columns are identical
def check_equivalence_compustat(group):
    # If all rows are identical (except 'month' and 'gvkey'), return True, otherwise False
    return group.drop(columns=['month', 'gvkey']).nunique().max() == 1

duplicate_groups_compustat = duplicates_compustat.groupby(['month', 'gvkey'])

# Separate the equivalent and non-equivalent duplicates
equivalent_duplicates_compustat = []
non_equivalent_duplicates_compustat = []

for name, group in duplicate_groups_compustat:
    if check_equivalence_compustat(group):
        equivalent_duplicates_compustat.append(group)
    else:
        non_equivalent_duplicates_compustat.append(group)

# Handle equivalent duplicates
if equivalent_duplicates_compustat:
    # Combine all equivalent duplicate groups
    equivalent_duplicates_df_compustat = pd.concat(equivalent_duplicates_compustat)
    # Remove duplicates, keeping only one
    compustat_data = compustat_data.drop_duplicates(subset=['month', 'gvkey'])
    print(f"Deleted {len(equivalent_duplicates_df_compustat)} duplicate rows (equivalent duplicates).")

# Handle non-equivalent duplicates by keeping rows with the least missing values
if non_equivalent_duplicates_compustat:
    non_equivalent_duplicates_df_compustat = pd.concat(non_equivalent_duplicates_compustat)

    # Function to count missing values per row and keep the row with the least missing values
    def keep_least_missing_compustat(group):
        # Count missing values in each row, sort by missing count, and keep the row with the fewest missing values
        return group.loc[group.isnull().sum(axis=1).idxmin()]

    # Apply the function to each group of non-equivalent duplicates
    resolved_non_equivalent_compustat = non_equivalent_duplicates_df_compustat.groupby(
        ['month', 'gvkey']).apply(keep_least_missing_compustat)

    # Step 1: Reset the index for both dataframes to ensure alignment
    non_equivalent_duplicates_df_compustat.reset_index(drop=True, inplace=True)
    compustat_data.reset_index(drop=True, inplace=True)

    # Step 2: Drop the existing non-equivalent duplicates from the original data by using 'month' and 'gvkey'
    compustat_data = compustat_data.merge(non_equivalent_duplicates_df_compustat[['month', 'gvkey']],
                                          how='left', indicator=True)
    compustat_data = compustat_data[compustat_data['_merge'] == 'left_only'].drop(
        columns='_merge')

    # Step 3: Append the resolved rows with least missing values back to the main dataset
    compustat_data = pd.concat(
        [compustat_data, resolved_non_equivalent_compustat])

    print(
    f"Resolved {len(non_equivalent_duplicates_df_compustat)} non-equivalent duplicate rows "
    f"by keeping rows with the least missing values."
    )
else:
    print("No non-equivalent duplicates found.")

# Final check for any remaining duplicates
final_duplicates_compustat = compustat_data[compustat_data.duplicated(
    subset=['month', 'gvkey'], keep=False)]

if final_duplicates_compustat.empty:
    print("Final check: No duplicates found for 'month' and 'gvkey' after cleaning.")
else:
    print(
        f"Final check: There are still {len(final_duplicates_compustat)} duplicate rows remaining.")
    print(final_duplicates_compustat)

# Check for unique combinations of 'month' and 'gvkey'
unique_combinations_compustat = compustat_data.drop_duplicates(subset=['month', 'gvkey'])

# Count the number of unique months, unique gvkeys, and unique month-gvkey combinations
num_unique_months_compustat = compustat_data['month'].nunique()
num_unique_gvkeys_compustat = compustat_data['gvkey'].nunique()
num_unique_combinations_compustat = unique_combinations_compustat.shape[0]

print(f"Number of unique months: {num_unique_months_compustat}")
print(f"Number of unique gvkeys: {num_unique_gvkeys_compustat}")
print(f"Number of unique month-gvkey combinations: {num_unique_combinations_compustat}")


########## Convert compustat data to monthly by filling ##########

# Ensure the compustat data is sorted by 'gvkey' and 'month'
compustat_data = compustat_data.sort_values(['gvkey', 'month'])

# Function to fill missing months for each gvkey based on its own range
def fill_missing_months(gvkey_group):
    # Get the minimum and maximum dates for the gvkey group
    min_date = gvkey_group['month'].min()
    max_date = gvkey_group['month'].max()

    # Generate a full monthly range with month-start dates within the min and max date of the gvkey
    full_months = pd.date_range(min_date, max_date, freq='MS').to_frame(index=False, name='month')

    # Filter out months that already exist in gvkey_group to avoid duplicates
    missing_months = full_months[~full_months['month'].isin(gvkey_group['month'])]

    # Add the gvkey column to missing_months to match the group structure
    missing_months['gvkey'] = gvkey_group['gvkey'].iloc[0]

    # Combine the existing data with missing months, then sort
    full_data = pd.concat([gvkey_group, missing_months], ignore_index=True).sort_values('month')

    # Forward fill missing values within the gvkey group
    full_data_filled = full_data.ffill()

    return full_data_filled

# Apply the function to fill missing months for each gvkey
compustat_data_monthly = compustat_data.groupby('gvkey', group_keys=False).apply(fill_missing_months)

# Reset index if needed
compustat_data_monthly.reset_index(drop=True, inplace=True)

print("Missing months filled for each gvkey.")

# After filling the missing months, ensure no duplicate entries for the same 'gvkey' and 'month'
duplicates_compustat_monthly = compustat_data_monthly[compustat_data_monthly.duplicated(
    subset=['gvkey', 'month'], keep=False)]

if not duplicates_compustat_monthly.empty:
    print(f"Duplicate entries found for 'gvkey' and 'month': {len(duplicates_compustat_monthly)} duplicates.")
else:
    print("No duplicate entries for 'gvkey' and 'month' after filling missing months.")

# Verify that all months between the min and max date for each 'gvkey' are filled without gaps.
min_max_dates_per_gvkey = compustat_data_monthly.groupby(
    'gvkey')['month'].agg(['min', 'max'])
months_per_gvkey = compustat_data_monthly.groupby('gvkey')['month'].nunique()
expected_months_per_gvkey = min_max_dates_per_gvkey.apply(
    lambda x: pd.date_range(x['min'], x['max'], freq='M').nunique(), axis=1)

gvkeys_with_missing_months = months_per_gvkey[months_per_gvkey < expected_months_per_gvkey]

if not gvkeys_with_missing_months.empty:
    print(f"gvkeys with missing months: {len(gvkeys_with_missing_months)}")
else:
    print("All gvkeys have complete months filled between their available date range.")


########## Calculate and add size proxy variables to compustat_data_monthly ##########

# Define the function to calculate 'book_equity' for each row in the DataFrame
def calculate_book_equity(row):
    # First attempt: use 'seqq' directly if available
    book_equity = row['seqq']
    
    # Second attempt: if 'seqq' is missing, use 'ceqq' + 'pstkq'
    if pd.isna(book_equity):
        book_equity = row['ceqq'] + row['pstkq'] if not pd.isna(row['ceqq']) and not pd.isna(row['pstkq']) else np.nan
    
    # Third attempt: if 'seqq' and 'ceqq' + 'pstkq' are missing, use 'atq' - 'ltq'
    if pd.isna(book_equity):
        if not pd.isna(row['atq']):
            book_equity = row['atq'] if pd.isna(row['ltq']) else row['atq'] - row['ltq']
        else:
            book_equity = np.nan

    # Add deferred taxes if available, prioritize 'txditcq', then 'txdbq' + 'itccy'
    deferred_taxes = row['txditcq'] if not pd.isna(row['txditcq']) else (
        (row['txdbq'] + row['itccy']) if not pd.isna(row['txdbq']) and not pd.isna(row['itccy']) else 0
    )

    # Deduct preferred stock; prioritize 'pstkrq', then 'pstklq', and finally 'pstkq'
    preferred_stock = row['pstkrq'] if not pd.isna(row['pstkrq']) else (
        row['pstkq'] if not pd.isna(row['pstkq']) else 0
    )
    
    # Calculate final book equity
    book_equity = book_equity + deferred_taxes - preferred_stock
    
    # Return book_equity
    return book_equity

# Apply the function to create the 'book_equity' column
compustat_data_monthly['book_equity'] = compustat_data_monthly.apply(calculate_book_equity, axis=1)

# Define 'total_assets'
compustat_data_monthly['total_assets'] = compustat_data_monthly['atq']

# Define 'sales'
compustat_data_monthly['sales'] = (
    compustat_data_monthly.groupby('gvkey')['saleq']
    .rolling(window=4, min_periods=1)
    .sum()
    .reset_index(level=0, drop=True)
)

# Calculate 'total_debt' as long-term debt + debt in current liabilities
compustat_data_monthly['total_debt'] = compustat_data_monthly['dlttq'].fillna(0) + compustat_data_monthly['dlcq'].fillna(0)

print("Size proxy variables successfully calculated.")


########## Calculate net debt ##########

# Calculate Net Debt
compustat_data_monthly['net_debt'] = (
    compustat_data_monthly['total_debt'] 
    + compustat_data_monthly['mibtq'].fillna(0)  # Non-controlling interest
    + compustat_data_monthly['pstkq'].fillna(0)  # Preferred stock
    - compustat_data_monthly['chq'].fillna(0)    # Cash and cash equivalents
)


########## Size proxy variables EDA ##########

# Generate summary statistics for the size proxy variables
size_proxy_vars = ['total_assets', 'book_equity', 'sales', 'total_debt']
summary_stats_size_proxies = compustat_data_monthly[size_proxy_vars].describe()

# Print summary statistics for size proxy variables
print("Summary statistics for size proxy variables:\n", summary_stats_size_proxies)

# Check for missing values in size proxy variables
missing_values_size_proxies = compustat_data_monthly[size_proxy_vars].isnull().sum()
print("Missing values in size proxy variables:\n", missing_values_size_proxies)

# Generate boxplots for size proxy variables
# create_boxplot(compustat_data_monthly, size_proxy_vars)


########## Calculate and add _an variables to compustat_data_monthly ##########

# Ensure the compustat data is sorted by 'gvkey' and 'month'
compustat_data_monthly = compustat_data_monthly.sort_values(['gvkey', 'month'])

# Function to calculate _an variables within each gvkey group
def calculate_an_variables_parallel(group):
    group['assetgrowth_an'] = group['atq'].pct_change(fill_method=None)
    
    group['assetturnover2_an'] = (
        group['saleq'] / group['atq'].rolling(window=2).mean()
    )
    
    group['assetturnover_an'] = group['saleq'] / group['atq']
    
    group['bookequitygrowth_an'] = group['book_equity'].pct_change(fill_method=None)
    
    group['capexgrowth_an'] = group['capxy'].pct_change(fill_method=None)
    
    group['cash2assets_an'] = group['cheq'] / group['atq']
    
    group['cf2debt_an'] = (
        (group['ibq'] + group['dpq']) / group['ltq'].rolling(window=2).mean()
    )
    
    group['chbe_an'] = group['book_equity'].diff() / group['atq']
    
    group['chca_an'] = (
        (group['actq'].diff() - group['cheq'].diff()) / group['atq'].shift(1)
    )
    
    group['chceq_an'] = group['ceqq'].diff() / group['ceqq'].shift(1)
    
    group['chcl_an'] = (
        (group['lctq'].diff() - group['dlcq'].diff()) / group['atq'].shift(1)
    )
    
    group['chcurrentratio_an'] = (
        ((group['actq'] / group['lctq']) - (group['actq'].shift(1) / group['lctq'].shift(1))) /
        (group['actq'].shift(1) / group['lctq'].shift(1))
    )
    
    group['chfnl_an'] = (
        (group['dlttq'].diff().fillna(0) + group['dlcq'].diff().fillna(0) + 
         group['pstkq'].diff().fillna(0)) / group['atq'].shift(1)
    )
    
    group['chlt_an'] = group['ltq'].pct_change(fill_method=None)
    
    group['chnccl_an'] = (
        (group['ltq'].diff() - group['lctq'].diff() - group['dlttq'].diff().fillna(0)) / 
        group['atq'].shift(1)
    )
    
    group['debt2tang_an'] = (
        (group['cheq'] + group['rectq'] * 0.715 + group['invtq'] * 0.547 + 
         group['ppentq'] * 0.535) / group['atq']
    )
    
    group['deprn_an'] = group['dpq'] / group['ppentq']
    
    group['ebitda2revenue_an'] = group['oibdpq'] / group['revtq']
    
    group['grossmargin_an'] = (
        (group['revtq'] - group['cogsq']) / group['revtq']
    )
    
    group['grossprofit_an'] = (
        (group['revtq'] - group['cogsq']) / group['atq']
    )
    
    group['inventorychange_an'] = (
        group['invtq'].diff() / 
        ((0.5 * group['atq'].shift(1)) + (0.5 * group['atq']))
    )
    
    group['inventorygrowth_an'] = group['invtq'].pct_change(fill_method=None)
    
    group['investment_an'] = group['atq'].diff() / group['atq']
    
    group['liquid2assets_an'] = (
        (group['cheq'] + 0.75 * (group['actq'] - group['cheq']) +
         0.5 * (group['atq'] - group['actq'])) / group['atq'].shift(1)
    )
    
    group['marginch_an'] = (
        (group['ibq'] / group['saleq']) - (group['ibq'].shift(1) / group['saleq'].shift(1))
    )
    
    group['opleverage_an'] = (
        (group['cogsq'] + group['xsgaq']) / group['atq']
    )
    
    group['pchdeprn_an'] = (
        ((group['dpq'] / group['ppentq']) - (group['dpq'].shift(1) / group['ppentq'].shift(1))) /
        (group['dpq'].shift(1) / group['ppentq'].shift(1))
    )
    
    group['pchgm2pchsale_an'] = (
        (((group['saleq'] - group['cogsq']) - 
          (group['saleq'].shift(1) - group['cogsq'].shift(1))) / 
         (group['saleq'].shift(1) - group['cogsq'].shift(1))) -
        ((group['saleq'] - group['saleq'].shift(1)) / group['saleq'].shift(1))
    )
    
    group['pchquickratio_an'] = (
        (((group['actq'] - group['invtq']) / group['lctq']) - 
         ((group['actq'].shift(1) - group['invtq'].shift(1)) / group['lctq'].shift(1))) /
        ((group['actq'].shift(1) - group['invtq'].shift(1)) / group['lctq'].shift(1))
    )
    
    group['pchsale2pchinvt_an'] = (
        ((group['saleq'] - group['saleq'].shift(1)) / group['saleq'].shift(1)) - 
        ((group['invtq'] - group['invtq'].shift(1)) / group['invtq'].shift(1))
    )
    
    group['pchsale2pchrect_an'] = (
        ((group['saleq'] - group['saleq'].shift(1)) / group['saleq'].shift(1)) - 
        ((group['rectq'] - group['rectq'].shift(1)) / group['rectq'].shift(1))
    )
    
    group['pchsale2pchxsga_an'] = (
        ((group['saleq'] - group['saleq'].shift(1)) / group['saleq'].shift(1)) - 
        ((group['xsgaq'] - group['xsgaq'].shift(1)) / group['xsgaq'].shift(1))
    )
    
    group['pchsales2inv_an'] = (
        ((group['saleq'] / group['invtq']) - (group['saleq'].shift(1) / group['invtq'].shift(1))) /
        (group['saleq'].shift(1) / group['invtq'].shift(1))
    )
    
    group['profitability_an'] = (
        (group['saleq'] - group['cogsq'] - group['xintq'] - group['xsgaq']) / group['book_equity']
    )
    
    group['roic_an'] = (
        (group['oibdpq'] - group['nopiq']) / 
        (group['ceqq'] + group['ltq'] - group['cheq'])
    )
    
    group['sales2cash_an'] = group['saleq'] / group['cheq']
    
    group['sales2inv_an'] = group['saleq'] / group['invtq']
    
    group['sales2rec_an'] = group['saleq'] / group['rectq']
    
    group['salesgrowth_an'] = group['saleq'].pct_change(fill_method=None)
    
    return group

# Set number of cores for parallel processing
n_cores = 7

# Group data by 'gvkey' and parallelize the calculations
compustat_data_monthly = pd.concat(
    Parallel(n_jobs=n_cores)(
        delayed(calculate_an_variables_parallel)(group)
        for _, group in compustat_data_monthly.groupby('gvkey')
    )
)

print("_an variables successfully calculated per gvkey.")


########## _an variables EDA ##########

# Define the list of _an variables
an_vars = ['assetgrowth_an', 'assetturnover2_an', 'assetturnover_an', 'bookequitygrowth_an', 
           'capexgrowth_an','cash2assets_an', 'cf2debt_an', 'chbe_an', 'chca_an', 
           'chceq_an', 'chcl_an', 'chcurrentratio_an','chfnl_an', 'chlt_an', 'chnccl_an', 
           'debt2tang_an', 'deprn_an', 'ebitda2revenue_an', 'grossmargin_an',
           'grossprofit_an', 'inventorychange_an', 'inventorygrowth_an', 'investment_an', 
           'liquid2assets_an', 'marginch_an', 'opleverage_an', 'pchdeprn_an', 'pchgm2pchsale_an', 
           'pchquickratio_an','pchsale2pchinvt_an', 'pchsale2pchrect_an', 'pchsale2pchxsga_an', 
           'pchsales2inv_an', 'profitability_an','roic_an', 'sales2cash_an', 'sales2inv_an', 
           'sales2rec_an', 'salesgrowth_an']

# Replace inf and -inf with NaN
compustat_data_monthly[an_vars] = compustat_data_monthly[an_vars].replace(
    [np.inf, -np.inf], np.nan)

# Generate summary statistics for _an variables
summary_stats_an_vars = compustat_data_monthly[an_vars].describe()

# Generate boxplots for _an variables
# create_boxplot(compustat_data_monthly, an_vars)

print("Compustat processing successfully executed.")


# %% Store Compustat data (size proxy variables & _an variables)
# Store the final data into SQLite
compustat_data_monthly.to_sql(name='compustat_sizeproxies_and_an',
                              con=sql_database, if_exists='replace', index=False)

print("Compustat data with size proxies and _an variables successfully stored in the SQLite database.")


# %% CRSP query
# Query to retrieve CRSP Stock Data Monthly
crsp_query = f"""
SELECT 
    msf.permno, date_trunc('month', msf.mthcaldt)::date AS date, msf.mthret AS ret, 
    msf.shrout, msf.mthprc AS prc, ssih.primaryexch, ssih.siccd 
FROM 
    crsp.msf_v2 AS msf 
LEFT JOIN 
    crsp.stksecurityinfohist AS ssih 
ON
    msf.permno = ssih.permno 
AND 
    ssih.secinfostartdt <= msf.mthcaldt 
AND 
    msf.mthcaldt <= ssih.secinfoenddt 
WHERE 
    msf.mthcaldt BETWEEN '{start_date}' AND '{end_date}'
    AND ssih.sharetype = 'NS' 
    AND ssih.securitytype = 'EQTY' 
    AND ssih.securitysubtype = 'COM' 
    AND ssih.usincflg = 'Y' 
    AND ssih.issuertype in ('ACOR', 'CORP') 
    AND ssih.primaryexch in ('N', 'A', 'Q')
    AND msf.mthprc > 0
    AND msf.shrout > 0 
    AND (msf.siccd <= 6000 OR msf.siccd >= 6799)
"""

# Fetch, filter, and process CRSP data
crsp_data_raw = pd.read_sql_query(
    sql=crsp_query,
    con=wrds,
    dtype={"permno": int, "siccd": str},
    parse_dates=["date"]
)

# Create a copy of the raw data for reference
crsp_data = crsp_data_raw.copy()

print("CRSP query successfully executed.")


# %% CRSP processing
########## Change date to month ##########

# Convert date to month-level granularity and rename it to 'month'
crsp_data['month'] = crsp_data['date'].dt.to_period('M').dt.to_timestamp()

# Drop the original 'date' column
crsp_data.drop(columns=['date'], inplace=True)


########## Calculate market capitalization variables ##########

# Calculate market capitalization
crsp_data = (crsp_data
             .assign(mktcap=lambda x: (x["shrout"] * x["prc"]).abs() / 1000)
             .assign(mktcap=lambda x: x["mktcap"].replace(0, np.nan))
             )

# Sanity check for market capitalization (ensure it's non-negative)
invalid_mktcap = crsp_data[crsp_data['mktcap'] < 0]
if not invalid_mktcap.empty:
    print(
        f"Invalid (negative) market capitalization values found: {len(invalid_mktcap)} instances.")
else:
    print("No invalid market capitalization values (non-negative check passed).")

# Calculate lagged market capitalization
mktcap_lag = (crsp_data
              .assign(
                  date=lambda x: x["month"] + pd.DateOffset(months=1),
                  mktcap_lag=lambda x: x["mktcap"]
              )
              .get(["permno", "month", "mktcap_lag"])
              )

# Merge the lagged market capitalization back into the original data
crsp_data = (crsp_data
             .merge(mktcap_lag, how="left", on=["permno", "month"])
             )


########## Check for month-firm duplicates ##########

# Check for duplicate combinations of 'month' and 'permno'
duplicates_crsp = crsp_data[crsp_data.duplicated(
    subset=['month', 'permno'], keep=False)]
if not duplicates_crsp.empty:
    print(
        f"Duplicate entries found for 'month' and 'permno': {len(duplicates_crsp)} duplicates.")
else:
    print("No duplicate entries for 'month' and 'permno'.")

# Check for unique combinations of 'month' and 'permno'
unique_combinations_crsp = crsp_data.drop_duplicates(
    subset=['month', 'permno'])

# Count the number of unique months, unique permnos, and unique month-permno combinations
num_unique_months_crsp = crsp_data['month'].nunique()
num_unique_permnos_crsp = crsp_data['permno'].nunique()
num_unique_combinations_crsp = unique_combinations_crsp.shape[0]

print(f"Number of unique months: {num_unique_months_crsp}")
print(f"Number of unique permnos: {num_unique_permnos_crsp}")
print(f"Number of unique month-permno combinations: {num_unique_combinations_crsp}")


########## Calculate and add CAPM Beta and FF 3 factors to crsp_data ##########

# Query to retrieve FF 3 factors
factors_ff3_monthly_raw = pdr.DataReader(
    name="F-F_Research_Data_Factors",
    data_source="famafrench",
    start=start_date,
    end=end_date
)[0]

# Parse columns and scale FF 3 factors
factors_ff3_monthly = (factors_ff3_monthly_raw
                       .divide(100)
                       .reset_index(names="month")
                       .assign(month=lambda x: pd.to_datetime(x["month"].astype(str)))
                       .rename(str.lower, axis="columns")
                       .rename(columns={"mkt-rf": "mkt_excess"})
                       )

# Merge CRSP data with FF risk free rate and calculate excess returns
crsp_data = (crsp_data
             .merge(factors_ff3_monthly, how="left", on="month")
             .assign(ret_excess=lambda x: x["ret"]-x["rf"])
             .assign(ret_excess=lambda x: x["ret_excess"].clip(lower=-1))
             )

# Drop missing values
crsp_data = crsp_data.dropna(subset=["ret_excess"])

# Set parameters for rolling window and minimum number of observations
window_size = 48
min_obs = 36

# Identify 'permno' (stock identifiers) that have more than the required number of observations
valid_permnos = (crsp_data
                 .dropna()
                 .groupby("permno")["permno"]
                 .count()
                 .reset_index(name="counts")
                 .query(f"counts > {window_size}+1")
                 )

# Gather information about the first and last month of each 'permno'
permno_information = (crsp_data
                      .merge(valid_permnos, how="inner", on="permno")
                      .groupby(["permno"])
                      .aggregate(first_month=("month", "min"),
                                 last_month=("month", "max"))
                      .reset_index()
                      )

# Generate combinations of 'permno' and 'month'
unique_permno = crsp_data["permno"].unique()
unique_month = factors_ff3_monthly["month"].unique()
all_combinations = pd.DataFrame(
    product(unique_permno, unique_month), columns=["permno", "month"])

# Merge necessary columns from CRSP and Fama-French data
returns_monthly = (all_combinations
                   .merge(crsp_data[["permno", "month", "ret_excess"]],
                          how="left", on=["permno", "month"])
                   .merge(permno_information, how="left", on="permno")
                   .query("(month >= first_month) & (month <= last_month)")
                   .drop(columns=["first_month", "last_month"])
                   .merge(factors_ff3_monthly, how="left", on="month")
                   )

# Define a function to perform rolling CAPM estimation
def roll_capm_estimation(data, window_size, min_obs):
    data = data.sort_values("month")

    # Perform the rolling regression of 'ret_excess' on 'mkt_excess' using OLS
    result = (RollingOLS.from_formula(
        formula="ret_excess ~ mkt_excess",
        data=data,
        window=window_size,
        min_nobs=min_obs,
        missing="drop")
        .fit()
        .params.get("mkt_excess")
    )

    result.index = data.index
    return result

# Define a function to calculate rolling CAPM estimation in parallel for each 'permno'
def roll_capm_estimation_for_joblib(permno, group):
    if "date" in group.columns:
        group = group.sort_values(by="date")
    else:
        group = group.sort_values(by="month")

    # Perform rolling OLS regression for CAPM
    beta_values = (RollingOLS.from_formula(
        formula="ret_excess ~ mkt_excess",
        data=group,
        window=window_size,
        min_nobs=min_obs,
        missing="drop"
    )
        .fit()
        .params.get("mkt_excess")
    )

    # Prepare the result DataFrame
    result = pd.DataFrame(beta_values)
    result.columns = ["beta"]
    result["month"] = group["month"].values
    result["permno"] = permno

    # Try to handle 'date' column if present (for higher frequency data)
    try:
        result["date"] = group["date"].values
        result = result[
            (result.groupby("month")["date"].transform(
                "max")) == result["date"]
        ]
    except (KeyError):
        pass  # Ignore if 'date' is not present

    return result

# Group the data by 'permno' for parallelized computation
permno_groups = returns_monthly.groupby("permno")

# Set the number of CPU cores for parallel processing
n_cores = 7

# Calculate rolling CAPM beta estimates for each 'permno' in parallel
beta_monthly = (
    pd.concat(
        Parallel(n_jobs=n_cores)
        (delayed(roll_capm_estimation_for_joblib)(name, group)
         for name, group in permno_groups)
    )
    .dropna()
    .rename(columns={"beta": "beta_monthly"})
)

# Merge the rolling CAPM betas (beta_monthly) back into crsp_data
crsp_data = crsp_data.merge(beta_monthly, how="left", on=["permno", "month"])

print("CAPM betas succesfully calculated.")


# %% Store CRSP data (market capitalization, returns, beta)
crsp_data.to_sql(name='crsp_beta_ff3',
                 con=sql_database, if_exists='replace', index=False)

print("CRSP data successfully stored in the SQLite database.")


# %% Merge WRDS ratios, Compustat, and CRSP
########## Merging ##########

# Standardize 'gvkey' and 'permno' as string data types across datasets
wrds_ratios['gvkey'] = wrds_ratios['gvkey'].astype(str)
wrds_ratios['permno'] = wrds_ratios['permno'].astype(str)
compustat_data_monthly['gvkey'] = compustat_data_monthly['gvkey'].astype(str)
crsp_data['permno'] = crsp_data['permno'].astype(str)

# Step 1: Merge Compustat and WRDS Ratios on 'gvkey' and 'month'
compustat_wrds_merged = compustat_data_monthly.merge(
    wrds_ratios,
    how='outer',
    on=['gvkey', 'month']
)

# Step 2: Shift the 'month' in CRSP data to align with Compustat and WRDS data
crsp_data['shifted_month'] = crsp_data['month'] + pd.DateOffset(months=3)

# Step 3: Merge the CRSP data with the Compustat and WRDS merged dataset on 'permno' and 'shifted_month'
final_merged_data = crsp_data.merge(
    compustat_wrds_merged,
    how='inner',
    left_on=['permno', 'shifted_month'],
    right_on=['permno', 'month']
)

# Step 4: Drop or rename columns as needed to avoid duplicates and keep only relevant columns
final_merged_data.drop(columns=['month_y'], errors='ignore', inplace=True)
final_merged_data.rename(columns={'month_x': 'month'}, inplace=True)

print("WRDS ratios, Compustat and CRSP data succesfully merged.")


########## Merged data EDA ##########

# Check for duplicate combinations of 'gvkey', 'permno', and 'month'
duplicates_final_merged = final_merged_data[
    final_merged_data.duplicated(
        subset=['gvkey', 'permno', 'month'], keep=False)
]

if not duplicates_final_merged.empty:
    print(f"Duplicate entries found for 'gvkey', 'permno', and 'month': {len(duplicates_final_merged)} duplicates.")
else:
    print("No duplicate entries for 'gvkey', 'permno', and 'month'.")

# Count the number of unique gvkeys, permnos, and month-permno-gvkey combinations
num_unique_months_merged = final_merged_data['month'].nunique()
num_unique_gvkeys_merged = final_merged_data['gvkey'].nunique()
num_unique_combinations_merged = final_merged_data.drop_duplicates(
    subset=['gvkey', 'month']).shape[0]

print(f"Number of unique months: {num_unique_months_merged}")
print(f"Number of unique gvkeys: {num_unique_gvkeys_merged}")
print(f"Number of unique month-gvkey combinations: {num_unique_combinations_merged}")

# Generate summary statistics for the merged data
summary_stats_merged = final_merged_data.describe()


# %% Target variables
########## Calculate Enterprise Value ##########
final_merged_data['enterprise_value'] = final_merged_data['mktcap'] + final_merged_data['net_debt']


########## Calculate target variables ##########

# Calculate Market-to-Book (m2b), allowing negative book equity but avoiding division by zero
final_merged_data['m2b'] = np.where(
    final_merged_data['book_equity'] != 0,
    final_merged_data['mktcap'] / final_merged_data['book_equity'], 
    np.nan
)

# Calculate Enterprise Value to Assets (v2a), allowing negatives but avoiding division by zero
final_merged_data['v2a'] = np.where(
    final_merged_data['atq'] != 0, 
    final_merged_data['enterprise_value'] / final_merged_data['atq'], 
    np.nan
)

# Calculate Enterprise Value to Sales (v2s), allowing negatives but avoiding division by zero
final_merged_data['v2s'] = np.where(
    final_merged_data['sales'] != 0, 
    final_merged_data['enterprise_value'] / final_merged_data['sales'], 
    np.nan
)

# Calculate natural logarithms, handling non-positive values by setting them to NaN
final_merged_data['ln_m2b'] = final_merged_data['m2b'].apply(lambda x: np.log(x) if x > 0 else np.nan)
final_merged_data['ln_v2a'] = final_merged_data['v2a'].apply(lambda x: np.log(x) if x > 0 else np.nan)
final_merged_data['ln_v2s'] = final_merged_data['v2s'].apply(lambda x: np.log(x) if x > 0 else np.nan)

print("Target variables successfully calculated.")


########## Target variables EDA ##########

# List of target variables
target_variables = ['m2b', 'v2a', 'v2s', 'ln_m2b', 'ln_v2a', 'ln_v2s']

# Generate summary statistics for the target variables
summary_stats_targets = final_merged_data[target_variables].describe()

# Check for negative values in target variables
neg_m2b = final_merged_data[final_merged_data['m2b'] < 0]
neg_v2a = final_merged_data[final_merged_data['v2a'] < 0]
neg_v2s = final_merged_data[final_merged_data['v2s'] < 0]

# Print the counts of invalid (negative) values
print(f"Negative Market-to-Book values found: {len(neg_m2b)} instances.")
print(f"Negative Enterprise Value to Assets values found: {len(neg_v2a)} instances.")
print(f"Negative Enterprise Value to Sales values found: {len(neg_v2s)} instances.")


# %% Final merged data EDA
# Check for missing values in the merged dataset
missing_values_final = final_merged_data.isnull().sum()
print("Missing values per column in the final merged data:\n", missing_values_final)

# Generate summary statistics for the merged data
summary_stats_final = final_merged_data.describe()
print("Summary statistics for the final merged data:\n", summary_stats_final)


# %% Store the final merged data (including the calculated target variables) into the SQLite database
final_merged_data.to_sql(name='final_merged_data',
                         con=sql_database, if_exists='replace', index=False)

print("Final merged data successfully stored in the SQLite database.")


# %% Clean, describe, and close the database
sql_database.execute('VACUUM;')

# Fetch the list of tables in the database
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", sql_database)
print("\nDatabase contains the following tables:\n", tables['name'].tolist())

sql_database.close()

print("Database connection closed.")

