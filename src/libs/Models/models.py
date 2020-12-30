from .base import BaseModel
import lightgbm as lgb
import pickle
import numpy as np

class LGBM_Model(BaseModel):
    def get_model(self):
        return None

    def fit(self, train_X, train_y, valid_X, valid_y):
        train = lgb.Dataset(train_X, train_y)
        valid = lgb.Dataset(valid_X, valid_y)
        self.model = lgb.train(
            self.model_param,
            train, 
            valid_sets=valid, 
            #early_stopping_rounds=param["early_stopping_rounds"], 
            #verbose_eval=param["verbose_eval"],
        )

    def predict(self, X):
        preds = self.model.predict(X)
        return np.where(preds < 0, 0, preds)
    
    def save_weight(self, path):
        pickle.dump(self.model, open(path, 'wb'))

    def read_weight(self, fname):
        self.model = pickle.load(open(fname, 'rb'))


