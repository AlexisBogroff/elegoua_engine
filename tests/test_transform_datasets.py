import pytest
import pandas as pd
import numpy as np
from elegoua_engine import transform_datasets as td
np.random.seed(42)


class TestTransformDatasets:
    @classmethod
    def setup_class(cls):
        """ setup code before tests """
        cls.df = pd.DataFrame(
            {
                'a': np.random.randn(100),
                'b': np.random.randn(100),
                'c': np.random.randn(100),
                'd': np.random.randn(100),
                'e': np.random.randn(100),
                'f': np.random.randn(100),
                'g': np.random.randn(100),
                'h': np.random.randn(100),
                'i': np.random.randn(100),
                'j': np.random.randn(100),
                # 10 columns
            }
        )


    @pytest.fixture
    def test_df(self):
        return self.df.copy()


    def test_select_random_rows_selects_the_right_number_of_rows(self):
        result_10 = td.select_random_rows(self.df, 10)
        result_50 = td.select_random_rows(self.df, 50)
        assert self.df.index.isin(result_10).sum() == 10
        assert self.df.index.isin(result_50).sum() == 50


    def test_select_random_columns_selects_the_right_number_of_cols(self):
        result_2 = td.select_random_columns(self.df, 2)
        result_5 = td.select_random_columns(self.df, 5)
        assert self.df.columns.isin(result_2).sum() == 2
        assert self.df.columns.isin(result_5).sum() == 5


    def test_compute_num_of_rows_from_prop(self):
        result_10 = td.compute_num_of_rows_from_prop(self.df, .1)
        result_50 = td.compute_num_of_rows_from_prop(self.df, .5)
        result_80 = td.compute_num_of_rows_from_prop(self.df, .8)
        assert result_10 == 10
        assert result_50 == 50
        assert result_80 == 80


    def test_compute_num_of_columns_from_prop(self):
        result_1 = td.compute_num_of_columns_from_prop(self.df, .1)
        result_3 = td.compute_num_of_columns_from_prop(self.df, .3)
        result_8 = td.compute_num_of_columns_from_prop(self.df, .8)
        assert result_1 == 1
        assert result_3 == 3
        assert result_8 == 8


    def test_add_random_na_to_col(self, test_df):
        td.add_random_na_to_col(test_df, col='e', n_rows=10)
        assert test_df['e'].isna().sum() == 10


    def test_add_missing_values_everywhere(self, test_df):
        # 30%: selects 3 columns out of 10
        # 30%: selects 30 rows out of 100
        # 3*30 = 90
        td.add_missing_values_everywhere(test_df, prop_na=.3)
        assert test_df.isna().sum().sum() == 90


    def test_add_missing_values_to_specific_cols(self, test_df):
        props_na = {'b': .2, 'g': .9}
        td.add_missing_values_to_specific_cols(test_df, props_na)
        assert test_df['b'].isna().sum() == 20
        assert test_df['g'].isna().sum() == 90


    def test_add_missing_values_with_prop_as_float(self, test_df):
        td.add_missing_values(test_df, prop_na=.3)
        assert test_df.isna().sum().sum() == 90


    def test_add_outliers(self, test_df):
        td.add_outliers(test_df, prop=.1, cols=['a', 'c'])
        ma_sup = test_df['a'] > (test_df.mean() + test_df.std() * 2)[0]
        ma_inf = test_df['a'] < (test_df.mean() - test_df.std() * 2)[0]
        assert ma_sup.sum() == 4
        assert ma_inf.sum() == 6


    def test_add_missing_values_with_prop_as_list(self, test_df):
        props_na = [.2, .8, .1, .9, .5, .4, .3, .7, .9, .4]
        td.add_missing_values(test_df, props_na)
        assert test_df['b'].isna().sum() == 80
        assert test_df['c'].isna().sum() == 10


    def test_add_missing_values_with_prop_as_dict(self, test_df):
        props_na = {'b': .2, 'g': .9}
        td.add_missing_values(test_df, props_na)
        assert test_df['b'].isna().sum() == 20
        assert test_df['g'].isna().sum() == 90


    def test_add_missing_values_to_all_cols(self, test_df):
        props_na = [.2, .8, .1, .9, .5, .4, .3, .7, .9, .4]
        td.add_missing_values_to_all_cols(test_df, props_na)
        assert test_df['a'].isna().sum() == 20
        assert test_df['b'].isna().sum() == 80
        assert test_df['c'].isna().sum() == 10
        assert test_df['f'].isna().sum() == 40


    def test_add_nas(self, test_df):
        td.add_nas(test_df, [1, 3, 5], ['a'])
        td.add_nas(test_df, [4, 8, 1], ['g'])
        assert test_df.loc[[1, 3, 5], 'a'].isna().all()
        assert test_df.loc[[4, 8, 1], 'g'].isna().all()
        

    def test_select_rows_randomly_is_expected_size(self):
        result_2 = td.select_rows_randomly(self.df, 2)
        result_10 = td.select_rows_randomly(self.df, 10)
        result_30 = td.select_rows_randomly(self.df, 30)
        assert result_2.shape[0] == 2
        assert result_10.shape[0] == 10
        assert result_30.shape[0] == 30


    def test_select_rows_randomly_is_within_df_rows(self):
        result = td.select_rows_randomly(self.df, 10)
        assert result.isin(self.df.index).all()

    
    def test_select_columns_randomly_is_expected_size(self):
        result_2 = td.select_columns_randomly(self.df, 2)
        result_8 = td.select_columns_randomly(self.df, 8)
        assert result_2.shape[0] == 2
        assert result_8.shape[0] == 8

    def test_select_columns_randomly_is_within_df_columns(self):
        result = td.select_columns_randomly(self.df, 8)
        assert result.isin(self.df.columns).all()


    def test_proportion_to_num_cols(self):
        result_20_pct = td.proportion_to_num_cols(self.df, .2)
        result_40_pct = td.proportion_to_num_cols(self.df, .4)
        result_80_pct = td.proportion_to_num_cols(self.df, .8)
        expected_20_pct = round(self.df.shape[1] * .2)
        expected_40_pct = round(self.df.shape[1] * .4)
        expected_80_pct = round(self.df.shape[1] * .8)
        assert result_20_pct == expected_20_pct
        assert result_40_pct == expected_40_pct
        assert result_80_pct == expected_80_pct


    def test_add_empty_columns_has_increased_size(self, test_df):
        td.add_empty_columns(test_df, .2)
        assert test_df.shape[1] > self.df.shape[1]

    
    def test_shuffle_columns_has_different_order(self, test_df):
        test_df = td.shuffle_columns(test_df)
        assert (test_df.columns != self.df.columns).any()


    def test_add_empty_col_has_1_more_cols(self, test_df):
        td.add_empty_col(test_df)
        assert test_df.shape[1] > self.df.shape[1]


    def test_add_duplicate_variables_has_more_cols(self, test_df):
        td.add_duplicate_variables(test_df, .3)
        assert test_df.shape[1] > self.df.shape[1]


    def test_add_duplicate_variables_has_duplicates(self, test_df):
        td.add_duplicate_variables(test_df, .3)
        assert test_df.T.duplicated().any()


    @classmethod
    def teardown_class(cls):
        """ Clean up code after tests"""
        del cls.df
