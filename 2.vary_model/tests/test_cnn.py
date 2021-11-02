from utils import cnnTools
import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
import re


VEC = np.array([1, 2, 3, 4, 5, 6])
DATA = {"x1": VEC,
        "x2": 2 * VEC}
DATA["y"] = DATA["x1"] + DATA["x2"]

INPUT_COLS = ["x1", "x2"]
OUTPUT_COLS = "y"
DF = pd.DataFrame(data=DATA)
N = DF.shape

OBJ = cnnTools(df=DF,
               input_cols=INPUT_COLS,
               output_cols=OUTPUT_COLS)

def test_series_to_supervised():
    """ Happy test for data preparation """
    PATTERN = "\(t[+-]\d+\)"
    
    ans = {"x1(t-3)": np.array([1, 2]),
           "x1(t-2)": np.array([2, 3]),
           "x1(t-1)": np.array([3, 4]),
           "x1(t+0)": np.array([4, 5]),
           "x1(t+1)": np.array([5, 6])}
    aux_d = {}
    for k in ans:
        new_k = k.replace("x1", "x2")
        aux_d[new_k] = 2 * ans[k]
        try:
            time = re.findall(PATTERN, new_k)[0]
        except IndexError:
            time = "(t)"
        aux_d[f"y{time}"] = aux_d[new_k] + ans[k]
    
    data = {}
    data.update(ans)
    data.update(aux_d)
    df_ans = pd.DataFrame(data=data)
    df_ans = df_ans.astype("float32")
    df2check = cnnTools.series_to_supervised(DF, 3, 1, dropnan=True)
    df2check = df2check.astype("float32")
    
    for col in df_ans:
        assert (df_ans[col].values == df2check[col].values).all()
        
def test_create_col_name():
    """ Happy test to get correct columns names """
    ans = ["c1(t-1)", "c1(t+0)", "c1(t+999)"]
    col_name = "c1"
    windows_to_get = [-1, "+0", "+999"]
    ans2check = cnnTools.create_col_name(col_name, windows_to_get)
    
    assert ans == ans2check
    
def test_select_data():
    """ Happy test to get correct dataframe """
    col_name = "x1"
    windows_to_get = ["-3", "+1"]
    df = cnnTools.series_to_supervised(DF, 3, 1, dropnan=True)
    df_ans = df[["x1(t-3)", "x1(t+1)"]]
    df2check = cnnTools.select_data(df, col_name, windows_to_get)
    
    for col in df_ans:
        assert (df_ans[col].values == df2check[col].values).all()
        
def test_order_cols():
    """ Happy test for ordering cols in ascending order """
    cols = ["aa(t+0)", "x1(t-3)", "asdasd(t+222)"]
    ans = [ "x1(t-3)", "aa(t+0)", "asdasd(t+222)"]
    list2heck = cnnTools.order_cols(cols)
    
    for i in range(len(cols)):
        assert ans[i] == list2heck[i]
        
def test_prepare_data_to_cnn_1_col():
    """ Happy test for data preparation in cnn  for 1 col only"""
    df = cnnTools.series_to_supervised(DF, 3, 1, dropnan=True)
    windows_to_get = ["-3", "-2", "-1", "+0"]
    col_name = "x1"
    data = {}
    data[col_name] = cnnTools.select_data(df, col_name, windows_to_get)
    
    x2check = cnnTools.prepare_data_to_cnn(data=data)
    
    ans = np.array([[[1], [2], [3], [4]], [[2], [3], [4], [5]]])
    
    assert np.array_equal(ans, x2check)
    
    
def test_prepare_data_to_cnn_2_col():
    """ Happy test for data preparation in cnn  for 1 col only"""
    df = cnnTools.series_to_supervised(DF, 3, 1, dropnan=True)
    windows_to_get = ["-3", "-2", "-1", "+0"]
    data = {}
    data["x1"] = cnnTools.select_data(df, "x1", windows_to_get)
    data["x2"] = cnnTools.select_data(df, "x2", windows_to_get)
    
    x2check = cnnTools.prepare_data_to_cnn(data=data)
    
    ans = np.array([[[1, 2], [2, 4], [3, 6], [4, 8]], 
                    [[2, 4], [3, 6], [4, 8], [5, 10]]])
    
    assert np.array_equal(ans, x2check)
    
def test_prepare_output():
    """ Prepare data to output """
    df = cnnTools.series_to_supervised(DF, 3, 1, dropnan=True)
    y2check = cnnTools.prepare_output(df, "y(t+1)")
    ans = np.array([15, 18]).astype("float32")
    
    assert np.array_equal(ans, y2check)
    
def test_pipeline():
    """ Happy test for pipeline routine """
    
    df = cnnTools.series_to_supervised(DF, 3, 1, dropnan=True)
    x = OBJ.pipeline(df, ["x1", "x2"], 3)
    
    ans = np.array([[[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]], 
                    [[2, 4], [3, 6], [4, 8], [5, 10], [6, 12]]])
    
    assert np.array_equal(ans, x)
    
    
    
    
   
    
        
        
    
    