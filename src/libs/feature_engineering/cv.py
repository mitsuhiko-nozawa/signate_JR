import os.path as osp
import numpy as np
import pandas as pd
from .base import Feature
import datetime
from sklearn.model_selection import KFold

class date_cv(Feature):
    def create_features(self):
        # test_featsはNoneで返す
        # trainにしかないdateはランダムに5分割
        # trainにもtestにもあるdateは、その日付内でtrainNoで5分割
        train_df, test_df = self.read_input()
        use_cols = []
        for seed in self.seeds:
            feat_name = f"{self.name}_{seed}"
            use_cols.append(feat_name)
            train_df[feat_name] = -1

            train_date = train_df.groupby("date").count()[["id"]].sort_index().reset_index()
            test_date = test_df.groupby("date").count()[["id"]].sort_index().reset_index()
            date_df = train_date.merge(test_date, on="date", how="outer" , suffixes=["_train", "_test"]).sort_values("date")

            def func(x):
                is_train = x["id_train"] == x["id_train"]
                is_test = x["id_test"] == x["id_test"]
                is_train_test = is_train and is_test
                if is_train_test:
                    return "train_test"
                elif is_train:
                    return "train"
                elif is_test:
                    return "test"

            date_df["appearance"] = date_df.apply(func, axis=1)
            date_df_tr = date_df[date_df["appearance"] == "train"].set_index("date")

            kf = KFold(n_splits=self.nfolds, random_state=seed, shuffle=True)
            for fold, (tr_ind, val_ind) in enumerate(kf.split(date_df_tr)):
                val_date = date_df_tr.iloc[val_ind].index.to_list()
                train_index = train_df[train_df["date"].isin(val_date)].index
                train_df.loc[train_index, feat_name] = fold


            date_df_tr_te = date_df[date_df["appearance"] == "train_test"]
            for date in date_df_tr_te["date"]:
                trainNo_df = train_df[train_df["date"] == date].groupby("trainNo").count()[["id"]]
                kf = KFold(n_splits=self.nfolds, random_state=seed, shuffle=True)
                for fold, (tr_ind, val_ind) in enumerate(kf.split(trainNo_df)):
                    val_trainNo = trainNo_df.iloc[val_ind].index.to_list()
                    train_index = train_df[(train_df["date"] == date)&(train_df["trainNo"].isin(val_trainNo))].index
                    if train_df.loc[train_index][feat_name].nunique() != 1:
                        raise ValueError("error!")
                    train_df.loc[train_index, feat_name] = fold

        return train_df[use_cols], None

class date_ampm_cv(Feature):
    def create_features(self):
        # test_featsはNoneで返す
        # trainにしかないdateはランダムに5分割
        # trainにもtestにもあるdateは、その日付内でtrainNoで5分割
        # このcvはリークを全く許さない形式のcv
        train_df, test_df = self.read_input()
        train_df["ampm"] = train_df["planArrival"].map(lambda x : "am" if int(x[:2]) <= 14 else "pm")
        train_df["date_ampm"] = train_df["date"].astype("str") + " * " + train_df["ampm"]
        date_df = train_df.groupby("date_ampm")[["id"]].count().reset_index()
        use_cols = []
        for seed in self.seeds:
            feat_name = f"{self.name}_{seed}"
            use_cols.append(feat_name)
            train_df[feat_name] = -1
            kf = KFold(n_splits=self.nfolds, random_state=seed, shuffle=True)
            for fold, (tr_ind, val_ind) in enumerate(kf.split(date_df)):
                val_date = date_df.iloc[val_ind]["date_ampm"].to_list()
                train_df[feat_name][train_df["date_ampm"].isin(val_date)] = fold

        return train_df[use_cols], None

class timeSeries_cv(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        for df in [train_df]:
            df["ampm"] = df["planArrival"].map(lambda x : "am" if int(x[:2]) <= 14 else "pm")
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2]))
            df["planArrival_int"] = df["planArrival"].map(lambda x : int(x.replace(":", "")))
            df["date_ampm"] = df["date"].astype("str") + " * " + df["ampm"].astype("str")    
        valid_df = train_df.query("801 <= planArrival_int <= 1400 or 1801 <= planArrival_int")
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
        return train_df[use_cols], None