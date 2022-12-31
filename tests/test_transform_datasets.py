import pytest
import pandas as pd
import numpy as np
from elegoua_engine import transform_datasets as td


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


    @pytest.fixture
    def test_df(self):
        return self.df.copy()


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
