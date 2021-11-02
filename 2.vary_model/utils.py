import re
from typing import Pattern

import pandas as pd
import numpy as np


class cnnTools:
    def __init__(self,
                 df: pd.DataFrame,
                 input_cols: list,
                 output_cols: list,
                 test_ratio: float = .5) -> None:
        """ Class that treat data for cnn predition in time series """
        self.df = df
        # calculate n_test
        n_test = int(df.shape[0] * test_ratio)
        self.df_test = self.df[-n_test:]
        self.df_train = self.df[:-n_test]
        self.input_cols = input_cols
        self.output_cols = output_cols
        
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
        data ={}
        data["x_train"] = self.pipeline(self.df_sup_train,
                                        columns=self.input_cols,
                                        window=window)
        data["x_test"] = self.pipeline(self.df_sup_test,
                                       columns=self.input_cols,
                                       window=window)
        out_col = f"{self.output_cols}(t+1)"
        data["y_train"] = cnnTools.prepare_output(self.df_sup_train,
                                                  output_col=out_col)
        data["y_test"] = cnnTools.prepare_output(self.df_sup_test,
                                                 output_col=out_col)
        
        
        return data
        
        
    
    
        