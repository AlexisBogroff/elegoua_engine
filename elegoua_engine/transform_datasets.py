"""
Transform datasets (of medium size) to make them different for each student.
Thus increasing the number of datasets without requiring any additional real data.
The transformations are all applied a random manner.

transformations:
- add_missing_values
- add_outliers
- add_duplicate_variables
- add_empty_columns
- remove_variables (not implemented)
- remove_individuals (to test)
- mess_up_columns_names (not implemented)
- shuffle_columns (not implemented)
- add_duplicate_individuals (not implemented)
- change_columns_names (not implemented)
- deform_variables (not implemented)
"""
import numpy as np
import pandas as pd

def load_dataset(path, delim=','):
    return pd.read_csv(path, delimiter=delim)


def add_missing_values(dataset, prop_na):
    """
    Add a proportion of missing values to the dataset or to each column
    tested 2023.01.01

    args:
        prop_na:
        - float: affects linearly the whole dataframe
        - list: affects each column based on the list values
        - dictionnary: affects the specified columns
    """
    if isinstance(prop_na, float):
        _add_missing_values_everywhere(dataset, prop_na)
    elif isinstance(prop_na, list):
        _add_missing_values_to_all_cols(dataset, prop_na)
    elif isinstance(prop_na, dict):
        _add_missing_values_to_specific_cols(dataset, prop_na)


def add_outliers(dataset, prop, cols=None):
    """
    Add outliers that can be detected using a IQR or Z-score method with threshold 2
    tested 2023.01.01
    """
    # Use threshold way greater than classic threshold of 2 (or 1.6)
    THRESHOLD = 10
    
    for col in cols:
        num = _compute_num_of_rows_from_prop(dataset, prop=prop)
        rows = _select_random_rows(dataset, num=num)
        
        mean = dataset[col].mean()
        std = dataset[col].std()
        bound_up = mean + THRESHOLD * std
        bound_low = mean - THRESHOLD * std
        
        # Indicator: define the sign randomly (positive or negative)
        rands = np.random.randint(0, 2, num)

        # Compute value of each outlier
        outliers_low = rands * bound_low - abs(dataset.loc[rows, col] * rands)
        outliers_up = (1 - rands) * bound_up + abs(dataset.loc[rows, col] * (1 - rands))
        
        # Assign computed values to original dataset
        dataset.loc[rows, col] = outliers_low + outliers_up


def add_duplicate_variables(dataset:pd.DataFrame, proportion_to_duplicate=.1):
    """
    Add duplicate variables randomly
    tested 2022.12.31
    """
    n_cols_to_dup = _proportion_to_num_cols(dataset, proportion_to_duplicate)
    cols_names = _select_columns_randomly(dataset, n_cols_to_dup)
    for col_name in cols_names:
        _duplicate_column(dataset, col_name)


def add_empty_columns(dataset:pd.DataFrame, proportion_to_add:float=.1):
    """
    Add empty variables and shuffle columns order
    tested 2022.12.31
    """
    n_cols_to_add = _proportion_to_num_cols(dataset, proportion_to_add)
    for _ in range(n_cols_to_add):
        _add_empty_col(dataset)
    dataset = shuffle_columns(dataset)


def remove_variables(dataset):
    """
    Drop variables randomly on large datasets
    """
    raise NotImplementedError


def remove_individuals(dataset, proportion_to_drop=.3):
    n_rows = _proportion_to_num_rows(dataset, proportion_to_drop)
    idx = _select_rows_randomly(dataset, n_rows)
    dataset.drop(idx, inplace=True)


def mess_up_columns_names(dataset):
    raise NotImplementedError


def shuffle_columns(dataset):
    """
    Reorder dataset columns in a random order
    tested 2022.12.31
    """
    return dataset.sample(frac=1, axis=1)


def add_duplicate_individuals(dataset):
    raise NotImplementedError


def change_columns_names(dataset):
    raise NotImplementedError


def deform_variables(dataset):
    """ Use log, exp, and other transformations to defor variables randomly, yet on a coherent basis """
    raise NotImplementedError


def _select_random_rows(dataset, num):
    """ tested 2023.01.01 """
    return np.random.choice(dataset.index, num, replace=False)


def _select_random_columns(dataset, num):
    """ tested 2023.01.01 """
    return np.random.choice(dataset.columns, num, replace=False)


def _compute_num_of_rows_from_prop(dataset, prop):
    """ tested 2023.01.01 """
    num_of_rows = round(dataset.shape[0] * prop)
    return num_of_rows


def _compute_num_of_columns_from_prop(dataset, prop):
    """ tested 2023.01.01 """
    num_of_columns = round(dataset.shape[1] * prop)
    return num_of_columns


def _add_nas(dataset, rows, cols):
    """
    Add missing values to specified rows and cols
    tested 2023.01.01
    
    Case 1:
    - multiple rows and a unique column
    - multiple columns and a unique row
    Case 2:
    - multiple rows and multiple columns. This would add grids of NAs

    args:
        rows: list of rows names (not iloc)
        cols: list of cols names (not iloc)
    """
    dataset.loc[rows, cols] = np.nan


def _add_random_na_to_col(dataset, col, n_rows):
    """
    Add missing values to a specific column
    tested 2023.01.01

    args:
        col: should be a col name (not iloc)
    """
    rows = _select_random_rows(dataset, n_rows)
    _add_nas(dataset, rows, col)


def _add_missing_values_to_all_cols(dataset, props_na):
    """
    Add missing values at a specific proportion for each column
    tested 2023.01.01
    """
    for i_col, col in enumerate(dataset.columns):
        n_rows = _compute_num_of_rows_from_prop(dataset, props_na[i_col])
        _add_random_na_to_col(dataset, col, n_rows)


def _add_missing_values_everywhere(dataset, prop_na):
    """
    Add missing values anywhere based on the proportion specified
    tested 2023.01.01
    """
    n_rows = _compute_num_of_rows_from_prop(dataset, prop_na)
    n_cols = _compute_num_of_columns_from_prop(dataset, prop_na)
    cols_to_add_na = _select_random_columns(dataset, n_cols)
    for col in cols_to_add_na:
        _add_random_na_to_col(dataset, col, n_rows)


def _add_missing_values_to_specific_cols(dataset, props_na):
    """
    Add missing values at a specific proportion for a specified columns only
    tested 2023.01.01
    """
    for col, prop_na in props_na.items():
        n_rows = _compute_num_of_rows_from_prop(dataset, prop_na)
        _add_random_na_to_col(dataset, col, n_rows)


def _duplicate_column(dataset, column_name):
    """ Duplicate and add a specified column (new col having a random name) """
    dataset[_gen_rand_txt()] = dataset[column_name]


def _add_empty_col(dataset:pd.DataFrame):
    """
    Add empty column (with NaNs only) with a random name
    tested 2022.12.31
    """
    n_samples = dataset.shape[0]
    dataset[_gen_rand_txt()] = np.ones(n_samples) * np.nan


def _gen_rand_txt():
    return hex(np.random.randint(10000))


def _proportion_to_num_cols(dataset, proportion:float):
    """
    Count number of columns corresponding to expected proportion
    tested 2022.12.31
    """
    assert proportion >= 0 and proportion <= 1
    n_cols = dataset.shape[1]
    n_cols_prop = round(n_cols * proportion)
    return n_cols_prop


def _proportion_to_num_rows(dataset, proportion:float):
    """
    Count number of columns corresponding to expected proportion
    tested 2022.12.31
    """
    assert proportion >= 0 and proportion <= 1
    tot_rows = dataset.shape[0]
    n_rows = round(tot_rows * proportion)
    return n_rows


def _select_columns_randomly(dataset, n_to_select, replacement=False):
    """
    Select columns randomly 
    tested 2022.12.31
    """
    n_cols = dataset.shape[1]
    ids = np.random.choice(n_cols, size=n_to_select, replace=replacement)
    cols_names = dataset.columns[ids]
    return cols_names


def _select_rows_randomly(dataset, n_to_select, replacement=False):
    """
    Select rows randomly
    tested 2022.12.31
    """
    n = dataset.shape[0]
    ids = np.random.choice(n, size=n_to_select, replace=replacement)
    rows_idx = dataset.index[ids]
    return rows_idx
