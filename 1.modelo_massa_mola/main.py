#%%
from os.path import join

import numpy as np
import pandas as pd

from utils.create_case import CreateCase

# Inputs
hs = 7.8
tz = 11.8
t = np.arange(0, 3 * 3600, .125)
m = .05
k = 2
k1 = 0.3
c = .2

def main(G: list):
    """ Main function that will run the cases """
    print(f"Caso: {G}")
    case = CreateCase(hs, tz, m, c, k, k1, t, 1, altered=False)
    result = {}
    aux = case.get_xt()
    result["xt"] = aux[:,0]
    result["xt_dot"] = aux[:,1]
    result["yt"] = case.yt
    result["G"] = G
    result["t"] = t
    result["m"] = m
    result["k"] = k
    result["k1"] = k1
    result["c"] = c
    print(f"Completo")
    print(80 * "-")
    return result

if __name__ == "__main__":
    # Run cases to run
    g_vec = np.arange(0, 5, step=.5)
    results = [main(G) for G in g_vec]
    df = pd.DataFrame(data=results)
    save_file = join(r"F:\User\Documents\trabalho_mestrado\1.modelo_massa_mola", "dabase_g_cases_alterated.pkl")
    df.to_pickle(save_file)
# %%



