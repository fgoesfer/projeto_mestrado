from utils import cnnTools, MultiCnnTool
import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
import re


VEC = np.array([1, 2, 3, 4, 5, 6])
DATA = {"x1": VEC,
        "x2": 2 * VEC}
DATA["y"] = DATA["x1"] + DATA["x2"]
DATA["hs"] = np.array([.2, .2, .2, .3, .3, .3])
DATA["tp"] = np.array([12, 12, 12, 13, 13, 13])

INPUT_COLS = ["x1", "x2", "hs", "tp"]
CAT_VAR = ["hs", "tp"]
OUTPUT_COLS = "y"
DF = pd.DataFrame(data=DATA)
N = DF.shape

CNN_OBJ = cnnTools(df=DF,
                   input_cols=INPUT_COLS,
                   output_cols=OUTPUT_COLS)

MULT_OBJ = MultiCnnTool(CNN_OBJ, 
                        CAT_VAR,
                        StandardScaler)

def test_get_values_to_split():
    """ Happy test to get cat variables unique values """
    ans = {"hs": np.array([.2, .3]),
           "tp": np.array([12, 13])}
    
    val = MULT_OBJ.get_values_to_split()
    
    for k in ans:
        assert np.array_equal(val[k], ans[k])
        
        
def test_df_of_possibilities():
    """ Happy test for the creation of the datafrane for keyin"""
    
    data = MULT_OBJ.get_values_to_split()
    df_check = MultiCnnTool.get_df_cat_var_possibilities(data)
    poss = [(.2, 12), (.2, 13), (.3, 12), (.3, 13)]
    df_ans = pd.DataFrame(data=poss, columns=CAT_VAR)
    
    for c in df_ans.columns:
        assert np.array_equal(df_ans[c], df_check[c])
        
def test_get_split_database():
    """ Happy test to get the masked case """
    
    possib = pd.Series(data=[.2, 12], index=CAT_VAR)
    df_check, _ = MultiCnnTool.data_select(possib, DF)
    df_ans = pd.DataFrame(data={"x1": [1, 2, 3],
                                "x2": [2, 4, 6],
                                "y": [3, 6, 9],
                                "hs": [.2, .2, .2],
                                "tp": [12, 12, 12]})
    
    for c in df_ans.columns:
        assert np.array_equal(df_check[c], df_ans[c])
        
def test_get_split_database_empty():
    """ Happy test to get new empty dataframe """
    
    possib = pd.Series(data=[.2, 13], index=CAT_VAR)
    df_check, _ = MultiCnnTool.data_select(possib, DF)
    df_ans = pd.DataFrame(data={"x1": [],
                                "x2": [],
                                "y": [],
                                "hs": [],
                                "tp": []})
    for c in df_ans.columns:
        assert np.array_equal(df_check[c], df_ans[c])
        
def test_get_split_database_code_name():
    """ Happy test for name of the splitted case """
    possib = pd.Series(data=[.2, 12], index=CAT_VAR)
    df_check, code_check = MultiCnnTool.data_select(possib, DF)
    code_ans = "hs_0.2_tp_12.0_"
    
    assert code_ans == code_check
    
def test_main():
    """ Happy test to check main method """
    d1 = DF[(DF.hs == .2) & (DF.tp == 12)]
    d2 = DF[(DF.hs == .3) & (DF.tp == 13)]
    # create objects
    obj1 = cnnTools(df=d1,
                    input_cols=INPUT_COLS,
                    output_cols=OUTPUT_COLS)
    obj2 = cnnTools(df=d2,
                    input_cols=INPUT_COLS,
                    output_cols=OUTPUT_COLS)
    objs = {}
    objs["hs_0.2_tp_12.0_"] = obj1
    objs["hs_0.3_tp_13.0_"] = obj2
    
    ans_dic = MULT_OBJ.create_cat_objs()
    for k in objs:
        assert objs[k] == ans_dic[k]

    
    
    