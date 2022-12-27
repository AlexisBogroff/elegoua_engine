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


def add_duplicate_variables(dataset:pd.DataFrame, proportion_to_duplicate=.1):
    """ Add duplicate variables randomly """
    n_cols_to_dup = proportion_to_num_cols(dataset, proportion_to_duplicate)
    cols_names = select_columns_randomly(dataset, n_cols_to_dup)

    # Duplicate the selected columns (use random names)
    for cols_names in [n_cols_to_dup]:
        dataset[gen_rand_txt()] = dataset[cols_names]

    return dataset


def add_empty_col(dataset:pd.DataFrame):
    n_samples = dataset.shape[0]
    dataset[gen_rand_txt()] = np.ones(n_samples) * np.nan
    return dataset


def reorder_columns(dataset):
    cols = list(dataset.columns)
    np.random.shuffle(cols)
    dataset = dataset[cols]
    return dataset


def add_empty_columns(dataset:pd.DataFrame, proportion_to_add=.1):
    """ Add empty variables at a ramdom position """
    # Compute number of columns to add proportionally
    n_cols_to_add = proportion_to_num_cols(dataset, proportion_to_add)
    
    # Add empty columns
    for _ in range(n_cols_to_add):
        dataset = add_empty_col(dataset)
    
    # Reorder columns
    dataset = reorder_columns(dataset)

    return dataset


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


def proportion_to_num_cols(dataset, proportion):
    """ Count number of columns corresponding to expected proportion """
    n_cols = dataset.shape[1]
    n_cols_prop = int(round(n_cols * proportion))
    return n_cols_prop


def remove_variables(dataset):
    """
    Drop variables randomly on large datasets
    """
    raise NotImplementedError


def remove_individuals(dataset, proportion_to_drop=.3):
    raise NotImplementedError


def select_columns_randomly(dataset, n_to_select, replacement=False):
    """ Select columns randomly """
    n_cols = dataset.shape[1]
    ids = np.random.choice(n_cols, size=n_to_select, replace=replacement)
    cols_names = dataset.columns[ids]
    return cols_names
