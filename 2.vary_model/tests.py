from utils import cnnTools
import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

VEC = np.array([1, 2, 3, 4, 5, 6, 7,  8, 9, 10])
DATA = {"x1": VEC,
        "x2": 2 * VEC,
        "t": .1 * VEC}
DATA["y"] = DATA["x1"] + DATA["x2"]

INPUT_COLS = ["x1", "x2"]
OUTPUT_COLS = ["y"]
DF = pd.DataFrame(data=DATA)
N = DF.shape

OBJ = cnnTools(df=DF,
               input_cols=INPUT_COLS,
               output_cols=OUTPUT_COLS)

def test_data_prep():
    """ Happy test for data preparation """
    ans ={}
    ans["x1"] = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    ans["x2"] = 2 * ans["x1"]
    ans["y"] = ans["x1"] + ans["x2"]
    
    to_check = OBJ.data_prep(DF, INPUT_COLS, OUTPUT_COLS)
    for v in to_check:
        assert np.array_equal(to_check[v], ans[v])
    
def test_create_inputs():
    """ Happy test to create inputs for cnn """
    ans = np.array([[[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]], [[4], [5], [6]], 
                    [[5], [6], [7]], [[6], [7], [8]], [[7], [8], [9]], [[8], [9], [10]]])
                    
    vector = OBJ.data_prep(DF, INPUT_COLS, OUTPUT_COLS)
    to_check = SmartDynamicPredictor.create_inputs(vector['x1'], n_steps=3)
    assert np.array_equal(to_check, ans)
    
    
def test_create_outputs():
    ans_prev = np.array([[[3], [6], [9]],
                        [[6], [9], [12]],
                        [[9], [12], [15]],
                        [[12], [15], [18]],
                        [[15], [18], [21]],
                        [[18], [21], [24]],
                        [[21], [24], [27]]])
    ans_y = np.array([12, 15, 18, 21, 24, 27, 30])
    vector = OBJ.data_prep(DF, INPUT_COLS, OUTPUT_COLS)
    to_check_prev, to_check_y = SmartDynamicPredictor.create_output(vector["y"], 3, True)
    pytest.set_trace()
    