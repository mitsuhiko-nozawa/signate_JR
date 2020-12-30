import os.path as osp
import numpy as np
import pandas as pd
from .base import Feature

from sklearn.model_selection import KFold

class id(Feature):
    def create_features(self):
        return self.create_default_features(dtypes={self.fname : "int64"})

class date(Feature):
    def create_features(self):
        return self.create_default_features(dtypes={self.fname : "int64"})

class lineName(Feature):
    def create_features(self):
        return self.create_default_features(dtypes={self.fname : "object"})

class directionCode(Feature):
    def create_features(self):
        return self.create_default_features(dtypes={self.fname : "int64"})

class trainNo(Feature):
    def create_features(self):
        return self.create_default_features(dtypes={self.fname : "object"})

class stopStation(Feature):
    def create_features(self):
        return self.create_default_features(dtypes={self.fname : "object"})

class planArrival(Feature):
    def create_features(self):
        return self.create_default_features(dtypes={self.fname : "object"})

class delayTime(Feature):
    def create_features(self):
        return self.create_default_features(dtypes={self.fname : "int64"})


class continuedDelayTime(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        train_df["planArrivalNumeric"] = train_df["planArrival"].map(lambda x : int(x.replace(":", "")))

        train_index_am = train_df.query("901 <= planArrivalNumeric <= 1200").index
        train_index_pm = train_df.query("1901 <= planArrivalNumeric").index
        train_df.loc[train_index_am, "delayTime"] = np.nan
        train_df.loc[train_index_pm, "delayTime"] = np.nan

        date_am = test_df[test_df["planArrival"] == "08:01"].groupby("date").agg({"delayTime" : "count"}).query("delayTime != 0").index.to_list()
        train_df_am = train_df.query("801 <= planArrivalNumeric <= 900")
        train_index_am = train_df_am[~train_df_am["date"].isin(date_am)].index

        date_pm = test_df[test_df["planArrival"] == "18:01"].groupby("date").agg({"delayTime" : "count"}).query("delayTime != 0").index.to_list()
        train_df_pm = train_df.query("1801 <= planArrivalNumeric <= 1900")
        train_index_pm = train_df_pm[~train_df_pm["date"].isin(date_pm)].index

        train_df.loc[train_index_am, "delayTime"] = np.nan
        train_df.loc[train_index_pm, "delayTime"] = np.nan 

        train_df[self.name] = train_df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x: x.fillna(method="ffill").fillna(0))
        test_df[self.name] = test_df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x: x.fillna(method="ffill").fillna(0))       

        return train_df[[self.name]], test_df[[self.name]]


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

