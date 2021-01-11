from .base import BaseModel
from catboost import Pool, CatBoostRegressor, CatBoost
import lightgbm as lgb
import pickle
import numpy as np

class LGBM_Model(BaseModel):
    def get_model(self):
        return None

    def fit(self, train_X, train_y, valid_X, valid_y):
        train = lgb.Dataset(train_X.values, train_y)
        valid = lgb.Dataset(valid_X.values, valid_y)
        self.model = lgb.train(
            self.model_param,
            train, 
            valid_sets=valid, 
            #early_stopping_rounds=param["early_stopping_rounds"], 
            #verbose_eval=param["verbose_eval"],
        )

    def predict(self, X):
        preds = self.model.predict(X.values)
        return np.where(preds < 0, 0, preds)
    
    def save_weight(self, path):
        pickle.dump(self.model, open(path, 'wb'))

    def read_weight(self, fname):
        self.model = pickle.load(open(fname, 'rb'))



class CatBoost_Model(BaseModel):
    def get_model(self):
        if self.model_param is None:
            return None
        else:
            return CatBoostRegressor(**self.model_param)

    def fit(self, train_X, train_y, valid_X, valid_y):
        categorical_features_indices = np.where(train_X.dtypes == "object")[0]
        train_data = Pool(train_X, train_y, cat_features=categorical_features_indices)
        valid_data = Pool(valid_X, valid_y, cat_features=categorical_features_indices)
        self.model.fit(
            train_data,
            eval_set=valid_data, 
            early_stopping_rounds=100, 
            use_best_model=True,
        )

    def predict(self, X):
        preds = self.model.predict(X)
        return np.where(preds < 0, 0, preds)
    
    def save_weight(self, path):
        pickle.dump(self.model, open(path, 'wb'))

    def read_weight(self, fname):
        self.model = pickle.load(open(fname, 'rb'))


