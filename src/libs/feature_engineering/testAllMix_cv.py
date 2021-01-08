import os.path as osp
import numpy as np
import pandas as pd
from .base import Feature
import datetime
from sklearn.model_selection import KFold

class testAllMix_timeSeries_cv(Feature):
    def create_features(self):
        train_df, test_df = self.testAllMix_read_input()
        for df in [train_df]:
            df["ampm"] = df["planArrival"].map(lambda x : "am" if int(x[:2]) <= 14 else "pm")
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2]))
            df["planArrival_int"] = df["planArrival"].map(lambda x : int(x.replace(":", "")))
            df["date_ampm"] = df["date"].astype("str") + " * " + df["ampm"].astype("str")    
        valid_df = train_df[:1488885]
        valid_df = valid_df.query("801 <= planArrival_int <= 1400 or 1801 <= planArrival_int")
        valid_df = valid_df.groupby(["date_ampm"]).count().reset_index()    

        use_cols = []
        for seed in self.seeds:
            feat_name = f"{self.name}_{seed}"
            use_cols.append(feat_name)
            train_df[feat_name] = -1
            kf = KFold(n_splits=self.nfolds, random_state=seed, shuffle=True)
            for fold, (tr_ind, val_ind) in enumerate(kf.split(valid_df)):
                val_date_ampm = valid_df.iloc[val_ind]["date_ampm"].tolist()
                train_index = train_df[train_df["date_ampm"].isin(val_date_ampm)].query("801 <= planArrival_int <= 1400 or 1801 <= planArrival_int").index
                train_df.loc[train_index, feat_name] = fold
        train_df = train_df
        return train_df[use_cols], None