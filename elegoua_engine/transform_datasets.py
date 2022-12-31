"""
Transform datasets (of medium size) to make them different for each student.
Thus increasing the number of datasets without requiring any additional real data.
The transformations are all applied a random manner.
"""
import numpy as np
import pandas as pd

def load_dataset(path, delim=','):
    return pd.read_csv(path, delimiter=delim)

def add_missing_values(dataset):

    raise NotImplementedError

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
