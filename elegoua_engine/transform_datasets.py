"""
Transform datasets (of medium size) to make them different for each student.
Thus increasing the number of datasets without requiring any additional real data.
The transformations are all applied a random manner.
"""
import numpy as np
import pandas as pd

def load_dataset(path, delim=','):
    return pd.read_csv(path, delimiter=delim)


def select_random_rows(dataset, num):
    """ tested 2023.01.01 """
    return np.random.choice(dataset.index, num, replace=False)


def select_random_columns(dataset, num):
    """ tested 2023.01.01 """
    return np.random.choice(dataset.columns, num, replace=False)


def compute_num_of_rows_from_prop(dataset, prop):
    """ tested 2023.01.01 """
    num_of_rows = round(dataset.shape[0] * prop)
    return num_of_rows


def compute_num_of_columns_from_prop(dataset, prop):
    """ tested 2023.01.01 """
    num_of_columns = round(dataset.shape[1] * prop)
    return num_of_columns


def add_nas(dataset, rows, cols):
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


def add_random_na_to_col(dataset, col, n_rows):
    """
    Add missing values to a specific column
    tested 2023.01.01

    args:
        col: should be a col name (not iloc)
    """
    rows = select_random_rows(dataset, n_rows)
    add_nas(dataset, rows, col)


def add_missing_values_to_all_cols(dataset, props_na):
    """
    Add missing values at a specific proportion for each column
    tested 2023.01.01
    """
    for i_col, col in enumerate(dataset.columns):
        n_rows = compute_num_of_rows_from_prop(dataset, props_na[i_col])
        add_random_na_to_col(dataset, col, n_rows)


def add_missing_values_everywhere(dataset, prop_na):
    """
    Add missing values anywhere based on the proportion specified
    tested 2023.01.01
    """
    n_rows = compute_num_of_rows_from_prop(dataset, prop_na)
    n_cols = compute_num_of_columns_from_prop(dataset, prop_na)
    cols_to_add_na = select_random_columns(dataset, n_cols)
    for col in cols_to_add_na:
        add_random_na_to_col(dataset, col, n_rows)


def add_missing_values_to_specific_cols(dataset, props_na):
    """
    Add missing values at a specific proportion for a specified columns only
    tested 2023.01.01
    """
    for col, prop_na in props_na.items():
        n_rows = compute_num_of_rows_from_prop(dataset, prop_na)
        add_random_na_to_col(dataset, col, n_rows)


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
        add_missing_values_everywhere(dataset, prop_na)
    elif isinstance(prop_na, list):
        add_missing_values_to_all_cols(dataset, prop_na)
    elif isinstance(prop_na, dict):
        add_missing_values_to_specific_cols(dataset, prop_na)


def add_outliers(dataset):
    raise NotImplementedError


def duplicate_column(dataset, column_name):
    """ Duplicate and add a specified column (new col having a random name) """
    dataset[gen_rand_txt()] = dataset[column_name]


def add_duplicate_variables(dataset:pd.DataFrame, proportion_to_duplicate=.1):
    """
    Add duplicate variables randomly
    tested 2022.12.31
    """
    n_cols_to_dup = proportion_to_num_cols(dataset, proportion_to_duplicate)
    cols_names = select_columns_randomly(dataset, n_cols_to_dup)
    for col_name in cols_names:
        duplicate_column(dataset, col_name)


def add_empty_col(dataset:pd.DataFrame):
    """
    Add empty column (with NaNs only) with a random name
    tested 2022.12.31
    """
    n_samples = dataset.shape[0]
    dataset[gen_rand_txt()] = np.ones(n_samples) * np.nan


def shuffle_columns(dataset):
    """
    Reorder dataset columns in a random order
    tested 2022.12.31
    """
    return dataset.sample(frac=1, axis=1)


def add_empty_columns(dataset:pd.DataFrame, proportion_to_add:float=.1):
    """
    Add empty variables and shuffle columns order
    tested 2022.12.31
    """
    n_cols_to_add = proportion_to_num_cols(dataset, proportion_to_add)
    for _ in range(n_cols_to_add):
        add_empty_col(dataset)
    dataset = shuffle_columns(dataset)


def add_duplicate_individuals(dataset):
    raise NotImplementedError


def change_columns_names(dataset):
    raise NotImplementedError


def deform_variables(dataset):
    """ Use log, exp, and other transformations to defor variables randomly, yet on a coherent basis """
    raise NotImplementedError


def gen_rand_txt():
    return hex(np.random.randint(10000))


def mess_up_columns_names(dataset):
    raise NotImplementedError


def proportion_to_num_cols(dataset, proportion:float):
    """
    Count number of columns corresponding to expected proportion
    tested 2022.12.31
    """
    assert proportion >= 0 and proportion <= 1
    n_cols = dataset.shape[1]
    n_cols_prop = round(n_cols * proportion)
    return n_cols_prop


def remove_variables(dataset):
    """
    Drop variables randomly on large datasets
    """
    raise NotImplementedError


def remove_individuals(dataset, proportion_to_drop=.3):
    raise NotImplementedError


def select_columns_randomly(dataset, n_to_select, replacement=False):
    """
    Select columns randomly 
    tested 2022.12.31
    """
    n_cols = dataset.shape[1]
    ids = np.random.choice(n_cols, size=n_to_select, replace=replacement)
    cols_names = dataset.columns[ids]
    return cols_names


def select_rows_randomly(dataset, n_to_select, replacement=False):
    """
    Select rows randomly
    tested 2022.12.31
    """
    n = dataset.shape[0]
    ids = np.random.choice(n, size=n_to_select, replace=replacement)
    rows_idx = dataset.index[ids]
    return rows_idx
