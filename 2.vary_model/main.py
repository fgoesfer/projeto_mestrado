import pandas as pd
import numpy as np

class SmartDynamicPredictor:
    def __init__(self, 
                 data: pd.DataFrame, 
                 train_ratio: float,
                 transient2cut: float,
                 scaler: callable,
                 val_split: float = .1,
                 input_cols: list = [],
                 output_cols: list = [],
                 use_prev: bool = True
                 ):
        """ Class that do the treatment of the dataset for prediction 
            in the model
            Args:
                data (Dataframe): Data frame of the results
                train_ration (float): Ratio that will be used for train de model
                transient2curt (float): Part of the data that will be ignored
                scaler (callable): Scaler function to call and scale
                val_split (float): Validation split data
        """
        self.data = data
        self.train_ratio = train_ratio
        self.transient2cut = transient2cut
        # Data without transient
        self.df = self.data[self.data.t > self.transient2cut]
        # Dataframe to train
        ttrain = self.df.t.max() * self.train_ratio
        self.df2train = self.df[self.df.t < ttrain]
        # Test data
        mask_test = (self.df.t > ttrain) & (self.df.t <= self.df.t.max())
        self.df2test = self.df[mask_test]
        # Scaler
        self._scaler = scaler
        self.dt = np.diff(self.df.t)[0]
        # values
        self.val_split = val_split
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.use_prev = use_prev
        

    def get_scaler(self):
        return self._scaler

    def scale_col(self, 
                  serie: pd.Series,
                  scaler=None):
        """ Scale a vector and return the scaler """
        vector = serie.values.reshape(-1, 1)
        if scaler:
            scld_data = scaler.fit_transform(vector)
            return scld_data.reshape(scld_data.shape[0])
        local_scl = self.get_scaler()
        local_scl = local_scl()
        scld_data = local_scl.fit_transform(vector)
        return scld_data.reshape(scld_data.shape[0]), local_scl

    def scld_data2train(self, 
                        df: pd.DataFrame,
                        cols2scl: list,
                        ):
        """ Routine to scale dataframe """
        d = df.copy()
        #d = d[d.G == g_value]
        # Scales fitted
        scalers = {}
        for col in cols2scl:
            serie = d[col]
            scld_data, scaler = self.scale_col(serie)
            d[col] = scld_data
            scalers[col] = scaler
        return d, scalers

    def scld_test(self, 
                  df: pd.DataFrame,
                  scalers: dict,
                  ):
        """ Appy scaled on the test dataframe """
        d = df.copy()
        #d = d[d.G == g]
        for col in scalers:
            scaler = scalers[col]
            d[col] = self.scale_col(serie=d[col],
                                    scaler=scaler)
        return d

    def data_prep(self,
                  df: pd.DataFrame,
                  input_cols: list,
                  output_col: str):
        """ Prepare data for feed keras model """
        data = {}
        d = df.copy()

        n = df.shape[0]
        for _, v in enumerate(input_cols + output_col):
            data[v] = d[v].values.reshape(n, 1)
        return data

    @staticmethod
    def create_inputs(vector, n_steps):
      """ Create data for cnn modeling """
      res = []

      for i in range(len(vector)):
        end_ix = i + n_steps
        if end_ix > len(vector):
          break
        seq = vector[i:end_ix]
        res.append(seq)

      return np.array(res)

    @staticmethod
    def create_output(vector, n_steps, is_input=False):
        """ Create output vector """
        res = []
        resy = []
        for i in range(len(vector)):
          end_ix = i + n_steps
          if end_ix >= len(vector):
            break
          if is_input:
            seq = vector[i:end_ix - 1]
            res.append(seq)
          seqy = vector[end_ix]
          resy.append(seqy) 

        return np.array(res), np.array(resy)

    def split_sequences(self, data, t_window):
        """ Split data to train the neural net model """
        n_steps = int(t_window / self.dt)
        X, y = list(), list()

        in_x = []
        for in_col in self.input_cols:
          to_stack = SmartDynamicPredictor.create_inputs(data[in_col], n_steps)
          in_x.append(to_stack)
        
        # inputs cols data preparation
        if len(self.input_cols) < 2:
            inputx = in_x[0]
            del in_x
        else:
            d = np.stack(in_x, axis=2)
            goal_to_cnn = (d.shape[0], d.shape[1], d.shape[2])
            input_x = d.reshape(goal_to_cnn)
            
        #output cols data preparation
        prev_res = []
        output_y = []
        for out_col in self.output_cols:
          prev, y = SmartDynamicPredictor.create_output(data[out_col], 
                                                        n_steps,
                                                        self.use_prev)
          prev_res.append(prev)
          output_y.append(y)
        
        # inputs cols data preparation
        out_d = np.stack(output_y, axis=2)
        out_to_cnn = (out_d.shape[0], out_d.shape[1])
        output = out_d.reshape(out_to_cnn)
        
        try:
        # Previus values
            prev_d = np.stack(prev_res, axis=2)
            prev_to_cnn = (prev_d.shape[0], prev_d.shape[1], prev_d.shape[2])
            previus = prev_d.reshape(prev_to_cnn)
        except np.AxisError:
            previus = None

    def agg_data_prep(self,
                      df: pd.DataFrame,
                      input_cols: list,
                      output_col: str,
                      t_window: float):
        """ Aggregate data_reparation """
        dataset = self.data_prep(df, input_cols, output_col)
        x, y = self.split_sequences(dataset, t_window)
        return x, y

    def main(self,
             input_cols: list,
             output_col: str,
             t_window: float,
             hs_values: list):
        """ Main function """
        cols2fit = input_cols + [output_col]

        aux_train = pd.DataFrame()
        aux_test = pd.DataFrame()
        self.scalers = {}
        df2tr, self.scalers = self.scld_data2train(self.df2train,
                                                   cols2fit,
                                                          )
        df2te = self.scld_test(self.df2test,
                               self.scalers,
                               )
        self.df2train = df2tr
        self.df2test = df2te
        
        x_train_aux = []
        x_test_aux = []
        x_val_aux = []
        y_train_aux = []
        y_test_aux = []
        y_val_aux = []
        all_data_by_hs = {}
        for hs in hs_values:
            
            df2tr = self.df2train[self.df2train.hs==hs]
            df2te = self.df2test[self.df2test.hs==hs]
            idx_val = int(df2te.shape[0] * self.val_split)
            x, y = self.agg_data_prep(df2tr, 
                                      input_cols, 
                                      output_col,
                                      t_window)
            x_train_aux.append(x)
            y_train_aux.append(y)
            # test
            x, y = self.agg_data_prep(df2te.iloc[idx_val:], 
                                      input_cols, 
                                      output_col,
                                      t_window)
            x_test_aux.append(x)
            y_test_aux.append(y)
            # validation
            x, y = self.agg_data_prep(df2te.iloc[:idx_val], 
                                      input_cols, 
                                      output_col,
                                      t_window)
            x_val_aux.append(x)
            y_val_aux.append(y)

            #all data
            df = pd.concat([df2tr, df2te])
            x, y = self.agg_data_prep(df, 
                                      input_cols, 
                                      output_col,
                                      t_window)
            all_data_by_hs[hs] = {"x": x,
                                "y": y}


        x_train_aux = tuple(x_train_aux)
        x_test_aux = tuple(x_test_aux)
        x_val_aux = tuple(x_val_aux)
        y_train_aux = tuple(y_train_aux)
        y_test_aux = tuple(y_test_aux)
        y_val_aux = tuple(y_val_aux)
        
        # storege_data
        data = {}
        data["x_train"] = np.vstack(x_train_aux)
        data["x_test"] = np.vstack(x_test_aux)
        data["x_val"] = np.vstack(x_val_aux)
        data["y_train"] = np.hstack(y_train_aux)
        data["y_test"] = np.hstack(y_test_aux)
        data["y_val"] = np.hstack(y_val_aux)
        data["n_steps"] = n_steps = int(t_window / self.dt)
        data["hs_values"] = hs_values
        data["x_train_aux"] = x_train_aux
        data["x_test_aux"] = x_test_aux
        data["x_val_aux"] = x_val_aux
        data["y_train_aux"] = y_train_aux
        data["y_test_aux"] = y_test_aux
        data["y_val_aux"] = y_val_aux
        data["all_data"] = all_data_by_hs

        return data