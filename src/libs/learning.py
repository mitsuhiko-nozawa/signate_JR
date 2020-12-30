import pandas as pd
import os
import os.path as osp

from Models.models import *

class Learning():
    def __init__(self, param):
        self.param = param
        self.ROOT = param["ROOT"]
        self.WORK_DIR = param["WORK_DIR"]
        self.val_pred_path = osp.join(self.WORK_DIR, "val_preds")
        self.weight_path = osp.join(self.WORK_DIR, "weight")

        self.cv = param["cv"]
        self.seeds = param["seeds"]
        self.nfolds = param["nfolds"]

        self.model = param["model"]
        self.model_param = param["model_param"] 

        if "val_preds" not in os.listdir(self.WORK_DIR): os.mkdir(self.val_pred_path)
        if "weight" not in os.listdir(self.WORK_DIR): os.mkdir(self.weight_path)

    def __call__(self):
        for seed in self.seeds:
            self.train_by_seed(seed)
        
    
    def train_by_seed(self, seed):
        for fold in range(self.nfolds):
            self.train_by_fold(seed, fold)
        train_fold = pd.read_feather(osp.join(self.ROOT, "my_feature", "train", f"{self.cv}.feather"))

    def train_by_fold(self, seed, fold):
        train_X = pd.read_csv(osp.join(self.WORK_DIR, "train", f"train_X_{fold}.csv"))
        train_y = pd.read_csv(osp.join(self.WORK_DIR, "train", f"train_y_{fold}.csv"))
        valid_X = pd.read_csv(osp.join(self.WORK_DIR, "valid", f"valid_X_{fold}.csv"))
        valid_y = pd.read_csv(osp.join(self.WORK_DIR, "valid", f"valid_y_{fold}.csv"))

        self.model_param["seed"] = seed
        model = eval(self.model)(self.model_param)
        model.fit(train_X, train_y, valid_X, valid_y)

        val_pred = pd.DataFrame(model.predict(valid_y), columns=["pred"])
        val_pred_fname = osp.join(self.val_pred_path, f"preds_{seed}_{fold}.csv")
        val_pred.to_csv(val_pred_fname, index=False)

        weight_path = osp.join(self.weight_path, f"{seed}_{fold}.pkl")
        model.save_weight(weight_path)



