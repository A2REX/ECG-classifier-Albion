import pandas as pd
import numpy as np # Still needed for split later, but not for signal loading
import ast

# Define the path and constants
path = r'C:\Users\albio\Downloads\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\\'
test_fold = 10

# 1. Load and convert annotation data (The main labels DataFrame)
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# 2. Skip Raw Signal Data Loading (The original X = load_raw_data(Y, ...))

# 3. Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

# 4. Define and Apply Diagnostic Superclass Aggregation
def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# 5. Split Labels into Train and Test Sets
# Note: Since X is not loaded, we only apply the mask to Y

# Train Labels
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test Labels
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

# The final results you need are y_train and y_test
print(f"Total Labels (Y): {len(Y)}")
print(f"Training Labels (y_train): {len(y_train)}")
print(f"Testing Labels (y_test): {len(y_test)}")