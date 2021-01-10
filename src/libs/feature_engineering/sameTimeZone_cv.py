import os.path as osp
import numpy as np
import pandas as pd
from .base import Feature
import datetime
from sklearn.model_selection import KFold

class sameTimeZone_8_cv(Feature):
    def create_features(self):
        train_df, test_df = self.testAllMix_read_input()
        hour = 8
        for df in [train_df]:
            df["ampm"] = df["planArrival"].map(lambda x : "am" if int(x[:2]) <= 14 else "pm")
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2]))
            df["planArrival_int"] = df["planArrival"].map(lambda x : int(x.replace(":", "")))
            df["date_ampm"] = df["date"].astype("str") + " * " + df["ampm"].astype("str")    
        dates = train_df[train_df["hour"] == hour]["date"].unique()
        use_cols = []
        for seed in self.seeds:
            feat_name = f"{self.name}_{seed}"
            use_cols.append(feat_name)
            train_df[feat_name] = -1
            kf = KFold(n_splits=self.nfolds, random_state=seed, shuffle=True)
            for fold, (tr_ind, val_ind) in enumerate(kf.split(dates)):
                val_date = dates[val_ind]
                train_df.loc[train_df["date"].isin(val_date), feat_name] = fold

        return train_df[use_cols], None

class sameTimeZone_9_cv(Feature):
    def create_features(self):
        train_df, test_df = self.testAllMix_read_input()
        hour = 9
        for df in [train_df]:
            df["ampm"] = df["planArrival"].map(lambda x : "am" if int(x[:2]) <= 14 else "pm")
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2]))
            df["planArrival_int"] = df["planArrival"].map(lambda x : int(x.replace(":", "")))
            df["date_ampm"] = df["date"].astype("str") + " * " + df["ampm"].astype("str")    
        dates = train_df[train_df["hour"] == hour]["date"].unique()
        use_cols = []
        for seed in self.seeds:
            feat_name = f"{self.name}_{seed}"
            use_cols.append(feat_name)
            train_df[feat_name] = -1
            kf = KFold(n_splits=self.nfolds, random_state=seed, shuffle=True)
            for fold, (tr_ind, val_ind) in enumerate(kf.split(dates)):
                val_date = dates[val_ind]
                train_df.loc[train_df["date"].isin(val_date), feat_name] = fold

        return train_df[use_cols], None

class sameTimeZone_18_cv(Feature):
    def create_features(self):
        train_df, test_df = self.testAllMix_read_input()
        hour = 18
        for df in [train_df]:
            df["ampm"] = df["planArrival"].map(lambda x : "am" if int(x[:2]) <= 14 else "pm")
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2]))
            df["planArrival_int"] = df["planArrival"].map(lambda x : int(x.replace(":", "")))
            df["date_ampm"] = df["date"].astype("str") + " * " + df["ampm"].astype("str")    
        dates = train_df[train_df["hour"] == hour]["date"].unique()
        use_cols = []
        for seed in self.seeds:
            feat_name = f"{self.name}_{seed}"
            use_cols.append(feat_name)
            train_df[feat_name] = -1
            kf = KFold(n_splits=self.nfolds, random_state=seed, shuffle=True)
            for fold, (tr_ind, val_ind) in enumerate(kf.split(dates)):
                val_date = dates[val_ind]
                train_df.loc[train_df["date"].isin(val_date), feat_name] = fold

        return train_df[use_cols], None

class sameTimeZone_19_cv(Feature):
    def create_features(self):
        train_df, test_df = self.testAllMix_read_input()
        hour = 19
        for df in [train_df]:
            df["ampm"] = df["planArrival"].map(lambda x : "am" if int(x[:2]) <= 14 else "pm")
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2]))
            df["planArrival_int"] = df["planArrival"].map(lambda x : int(x.replace(":", "")))
            df["date_ampm"] = df["date"].astype("str") + " * " + df["ampm"].astype("str")    
        dates = train_df[train_df["hour"] == hour]["date"].unique()
        use_cols = []
        for seed in self.seeds:
            feat_name = f"{self.name}_{seed}"
            use_cols.append(feat_name)
            train_df[feat_name] = -1
            kf = KFold(n_splits=self.nfolds, random_state=seed, shuffle=True)
            for fold, (tr_ind, val_ind) in enumerate(kf.split(dates)):
                val_date = dates[val_ind]
                train_df.loc[train_df["date"].isin(val_date), feat_name] = fold

        return train_df[use_cols], None