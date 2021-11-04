import re
from typing import Any, Pattern
from itertools import product

import pandas as pd
import numpy as np


class cnnTools:
    def __init__(self,
                 df: pd.DataFrame,
                 input_cols: list,
                 output_cols: str,
                 test_ratio: float = .5,
                 val_ratio: float = .1) -> None:
        """ Class that treat data for cnn predition in time series """
        self.df = df[input_cols + [output_cols]]
        # calculate n_test
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        
        n_test = int(df.shape[0] * test_ratio)
        self.df_test = self.df[-n_test:]
        self.df_train = self.df[:-n_test]
        # Calculate Validation
        n_val = int(self.df_test.shape[0] * test_ratio)
        self.df_val = self.df_test[:n_val]
        # input and outputs
        self.input_cols = input_cols
        self.output_cols = output_cols
        
    def __eq__(self, other):
        if isinstance(other, cnnTools):
            if other.input_cols != self.input_cols:
                return False
            elif other.output_cols != self.output_cols:
                return False
            elif other.test_ratio != self.test_ratio:
                return False
            elif other.val_ratio != self.val_ratio:
                return False
            elif set(other.df.columns) != set(self.df.columns):
                return False
            for col in other.df.columns:
                if not np.array_equal(other.df[col], self.df[col]):
                    return False
        return True
            
            
    @staticmethod
    def series_to_supervised(data, window=1, lag=1, dropnan=True):
        cols, names = list(), list()
        # Input sequence (t-n, ... t-1)
        for i in range(window, 0, -1):
            cols.append(data.shift(i))
            names += [('%s(t-%d)' % (col, i)) for col in data.columns]
        # Current timestep (t=0)
        cols.append(data)
        names += [('%s(t+0)' % (col)) for col in data.columns]
        # Target timestep (t=lag)
        cols.append(data.shift(-lag))
        names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
        # Put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # Drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    
    @staticmethod
    def create_col_name(col_name: str,
                        windows_to_get: list) -> list:
        """ Create columns based on root name and time to get """
        return [f"{col_name}(t{w})" for w in windows_to_get]
    
    @staticmethod
    def select_data(df :pd.DataFrame,
                    col_name: str,
                    windows_to_get: list) -> pd.DataFrame:
        """ Select dataframe to get based on column name and time window """
        # Create columns name
        cols = cnnTools.create_col_name(col_name, windows_to_get)
        return df[cols]
    
    @staticmethod
    def order_cols(cols: list) -> list:
        """ Order columns based on time steps """
        pattern = "[+-]\d+\)"
        order_dict = {}
        for col in cols:
            order_dict[col] = int(re.findall(pattern, 
                                             col)[0].replace(")", ""))
            
        sorted_dict = {k: v for k, v in sorted(order_dict.items(), 
                                               key=lambda item: item[1])}
        return [v for v in sorted_dict.keys()]
    
    @staticmethod
    def prepare_data_to_cnn(data: dict) -> np.array:
        """ Prepare data to input in keras cnn """
        n_features = len(data.keys())
        if n_features == 1:
            for k in data:
                df = data[k]
                return df.values.reshape(df.shape[0], 
                                         df.shape[1], 
                                         1)
        cnn_values = []
        for k in data:
            df = data[k]
            val = df.values.reshape(df.shape[0], 
                                    df.shape[1], 
                                    1)
            cnn_values.append(val)
            
        cnn_values = tuple(cnn_values)
        return np.stack(cnn_values, axis=2).reshape(df.shape[0],
                                                    df.shape[1],
                                                    n_features)
        
    @staticmethod
    def prepare_output(df: pd.DataFrame,
                       output_col: str) -> np.array:
        """ Create output col"""
        d = df[output_col]
        return d.values.reshape(d.shape[0])
    
    def pipeline(self,
                 df_sup: pd.DataFrame,
                 columns: list,
                 window: int = 1,
                 ):
        
        windows_to_get = [f"-{i + 1}" for i in range(window)]
        windows_to_get += ["+0", "+1"]
        data = {}
        # for each in put create data
        for col in columns:
            cols = cnnTools.create_col_name(col, windows_to_get)
            cols = cnnTools.order_cols(cols)
            data[col] = df_sup[cols]
        
        return cnnTools.prepare_data_to_cnn(data)
    
    
    def main(self,
             window: int = 1,   
             dropnan: bool = True):
        """ Pipeline to create dataset for feed cnn model """
        
        
        self.df_sup_train = cnnTools.series_to_supervised(self.df_train,
                                                          window=window,
                                                          lag=1,
                                                          dropnan=dropnan)
        self.df_sup_test = cnnTools.series_to_supervised(self.df_test,
                                                         window=window,
                                                         lag=1,
                                                         dropnan=dropnan)
        self.df_sup_val = cnnTools.series_to_supervised(self.df_val,
                                                        window=window,
                                                        lag=1,
                                                        dropnan=dropnan)
        data = {}
        data["x_train"] = self.pipeline(self.df_sup_train,
                                        columns=self.input_cols,
                                        window=window)
        data["x_test"] = self.pipeline(self.df_sup_test,
                                       columns=self.input_cols,
                                       window=window)
        data["x_val"] = self.pipeline(self.df_sup_val,
                                      columns=self.input_cols,
                                      window=window)
        out_col = f"{self.output_cols}(t+1)"
        data["y_train"] = cnnTools.prepare_output(self.df_sup_train,
                                                  output_col=out_col)
        data["y_test"] = cnnTools.prepare_output(self.df_sup_test,
                                                 output_col=out_col)
        data["y_val"] = cnnTools.prepare_output(self.df_sup_val,
                                                output_col=out_col)
        
        
        return data
    
class MultiCnnTool:
    def __init__(self, 
                 cnn_obj: Any,
                 cat_var: list,
                 scaler: callable) -> None:
        """ Class that will deal with non time dependet variables """
        
        self.cnn_obj = cnn_obj
        self.cat_var = cat_var
        # Create cases of categorical columns
        data = self.get_values_to_split()
        # Possibilities of categorical values 
        self.possib = MultiCnnTool.get_df_cat_var_possibilities(data)
        
    def get_values_to_split(self):
        """ Get values that will be combined"""
        data = {}
        for col in self.cat_var:
            vals = [v for v in self.cnn_obj.df[col].unique()]
            data[col] = vals
        return data
    
    @staticmethod
    def get_df_cat_var_possibilities(data: dict):
        """ Create dataframe of possibilities of cat values to mask later 
            Args:
                data: Dictionary of possibilities 
        """
        a = list(data.values())
        possibilities = list(product(*a))
        return pd.DataFrame(data=possibilities,
                            columns=data.keys())
    
    @staticmethod 
    def data_select(possib: pd.Series,
                    database:pd.DataFrame):
        """ Get full data based on cat values for one goal case 
            args:
                possib (pandas.Series): A serie that is a row of the 
                                        dataframes of possibilities
                database (pandas.DataFrame): Dataframe of the fulldaset
            output:
                result (pandas.DataFrame): Dataframe of the selected data
                code (str): String of the code created
            """
        
        result = database.copy()
        code = str() # code name for the case
        for col in possib.index:
            result = result[np.equal(result[col], possib[col])]
            code += f"{col}_{possib[col]}_"
            
        return result, code
    
    
    def create_cat_objs(self):
        """ Get a dictionary of cnnTools objects for each cat variable 
            output:
                split_data(dict): Dictionary with cnnTools objected with
                                  the spllited dataframe.
        """
        
        # Splitted data that will stored 
        split_data = {}
        for i in range(self.possib.shape[0]):
            serie = self.possib.iloc[i] # Possibility that will be slected
            result, code = MultiCnnTool.data_select(serie, 
                                                    self.cnn_obj.df)
            if not result.empty:
                split_data[code] = cnnTools(df=result,
                                            input_cols=self.cnn_obj.input_cols,
                                            output_cols=self.cnn_obj.output_cols,
                                            test_ratio=self.cnn_obj.test_ratio,
                                            val_ratio=self.cnn_obj.val_ratio)
        return split_data
            
        
        
    
    
        