import pandas as pd
import os
import os.path as osp

from Models.models import *
from sklearn.metrics import mean_absolute_error

class Learning():
    def __init__(self, param):
        self.param = param
        self.ROOT = param["ROOT"]
        self.WORK_DIR = param["WORK_DIR"]
        self.val_pred_path = osp.join(self.WORK_DIR, "val_preds")
        self.weight_path = osp.join(self.WORK_DIR, "weight")

        self.train_flag = param["flag"]
        self.cv = param["cv"]
        self.seeds = param["seeds"]
        self.nfolds = param["nfolds"]
        self.y = param["y"]

        self.model = param["model"]
        self.model_param = param["model_param"] 

        if "val_preds" not in os.listdir(self.WORK_DIR): os.mkdir(self.val_pred_path)
        if "weight" not in os.listdir(self.WORK_DIR): os.mkdir(self.weight_path)

    def __call__(self):
        for seed in self.seeds:
            self.train_by_seed(seed)
        
    
    def train_by_seed(self, seed):
        cv_feat = f"{self.cv}_{seed}"
        if self.train_flag:
            for fold in range(self.nfolds):
                self.train_by_fold(seed, fold)

        train_y = pd.read_feather(osp.join(self.ROOT, "my_features", "train", f"{self.y}.feather"))
        oof_preds = pd.read_feather(osp.join(self.ROOT, "my_features", "train", f"{self.cv}.feather"))
        print(f"data size : {oof_preds.shape}")
        oof_preds["pred"] = 0
        cnt = 0

        for fold in range(self.nfolds):
            val_preds = pd.read_csv(osp.join(self.val_pred_path, f"preds_{seed}_{fold}.csv"))
            oof_preds["pred"][oof_preds[cv_feat] == fold] = val_preds["pred"].values
            cnt += val_preds.shape[0]
        print(f"valid sum : {cnt}")
        oof_preds = oof_preds[["pred"]]
        oof_preds.to_csv(osp.join(self.val_pred_path, f"oof_preds_{seed}.csv"), index=False)
        cv_score = mean_absolute_error(train_y[self.y], oof_preds["pred"])
        print(f"cv : {cv_score}")

    def train_by_fold(self, seed, fold):
        train_X = pd.read_csv(osp.join(self.WORK_DIR, "train", f"train_X_{seed}_{fold}.csv"))
        train_y = pd.read_csv(osp.join(self.WORK_DIR, "train", f"train_y_{seed}_{fold}.csv"))
        valid_X = pd.read_csv(osp.join(self.WORK_DIR, "valid", f"valid_X_{seed}_{fold}.csv"))
        valid_y = pd.read_csv(osp.join(self.WORK_DIR, "valid", f"valid_y_{seed}_{fold}.csv"))

        self.model_param["seed"] = seed
        model = eval(self.model)(self.model_param)
        model.fit(train_X, train_y, valid_X, valid_y)

        val_pred = pd.DataFrame(model.predict(valid_y), columns=["pred"])
        val_pred_fname = osp.join(self.val_pred_path, f"preds_{seed}_{fold}.csv")
        val_pred.to_csv(val_pred_fname, index=False)

        weight_path = osp.join(self.weight_path, f"{seed}_{fold}.pkl")
        model.save_weight(weight_path)



