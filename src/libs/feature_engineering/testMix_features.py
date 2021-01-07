import os.path as osp
import datetime
import numpy as np
import pandas as pd
from .base import Feature

from sklearn.model_selection import KFold

class testMix_id(Feature):
    def create_features(self):
        return self.testMix_create_default_features(dtypes={self.name : "int64"})

class testMix_date(Feature):
    def create_features(self):
        return self.testMix_create_default_features(dtypes={self.name : "int64"})

class testMix_lineName(Feature):
    def create_features(self):
        return self.testMix_create_default_features(dtypes={self.name : "object"})

class testMix_directionCode(Feature):
    def create_features(self):
        return self.testMix_create_default_features(dtypes={self.name : "int64"})

class testMix_trainNo(Feature):
    def create_features(self):
        return self.testMix_create_default_features(dtypes={self.name : "object"})

class testMix_stopStation(Feature):
    def create_features(self):
        return self.testMix_create_default_features(dtypes={self.name : "object"})

class testMix_planArrival(Feature):
    def create_features(self):
        return self.testMix_create_default_features(dtypes={self.name : "object"})

class testMix_delayTime(Feature):
    def create_features(self):
        return self.testMix_create_default_features(dtypes={self.name : "int64"})


class testMix_continuedDelayTime(Feature):
    def create_features(self):
        train_df, test_df = self.testMix_read_input()
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
        train_df["isnanDelayTime"] = train_df["delayTime"].map(lambda x : 1 if x != x else 0)
        test_df["isnanDelayTime"] = test_df["delayTime"].map(lambda x : 1 if x != x else 0)
        print(train_df["isnanDelayTime"].sum())

        train_df[self.name] = train_df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x: x.fillna(method="ffill").fillna(-999))
        test_df[self.name] = test_df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x: x.fillna(method="ffill").fillna(-999))       

        return train_df[[self.name, "isnanDelayTime"]], test_df[[self.name, "isnanDelayTime"]]


class testMix_date_cv(Feature):
    def create_features(self):
        # test_featsはNoneで返す
        # trainにしかないdateはランダムに5分割
        # trainにもtestにもあるdateは、その日付内でtrainNoで5分割
        train_df, test_df = self.testMix_read_input()
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
                        raise ValueError(f"Value error, feature {feat_name} has {train_df.loc[train_index][feat_name].nunique()} unique values.")
                    train_df.loc[train_index, feat_name] = fold

        return train_df[use_cols], None

class testMix_date_TrainNo_count(Feature):
    def create_features(self):
        train_df, test_df = self.testMix_read_input()
        train_df[self.name] = train_df.groupby(["date", "trainNo"])["id"].transform("count")
        test_df[self.name] = test_df.groupby(["date", "trainNo"])["id"].transform("count")
        return train_df[[self.name]], test_df[[self.name]]

class testMix_desc_continuedDelayTime(Feature):
    def create_features(self):
        train_df, test_df = self.testMix_read_input()
        cDT_tr = pd.read_feather(osp.join(self.ROOT, "my_features", "train", "continuedDelayTime.feather"))
        cDT_te = pd.read_feather(osp.join(self.ROOT, "my_features", "test", "continuedDelayTime.feather"))
        train_df = pd.concat([train_df, cDT_tr], axis=1)
        test_df = pd.concat([test_df, cDT_te], axis=1)
        aggs = ["mean", "median", "max", "min", "var"]
        feats = []
        for agg in aggs:
            feat_name = f"{agg}_continuedDelayTime"
            feats.append(feat_name)
            train_df[feat_name] = train_df.groupby(["date", "trainNo"])["continuedDelayTime"].transform(agg)
            test_df[feat_name] = test_df.groupby(["date", "trainNo"])["continuedDelayTime"].transform(agg)
        return train_df[feats], test_df[feats]

class testMix_info(Feature):
    def create_features(self):
        train_df, test_df = self.testMix_read_input()
        info_df = pd.read_csv(osp.join(self.ROOT, "input", "info.csv"))
        info_df = info_df.groupby(["date", "lineName"])["cse"].unique().reset_index()
        info_df["cse"] = info_df["cse"].map(lambda x : "".join(sorted(x)))
        train_df = train_df.merge(info_df, on=["date", "lineName"], how="left")
        test_df = test_df.merge(info_df, on=["date", "lineName"], how="left")
        train_df["cse"].fillna("None", inplace=True)
        test_df["cse"].fillna("None", inplace=True)
        return train_df[["cse"]], test_df[["cse"]]

class testMix_dateTransformed(Feature):
    def create_features(self):
        train_df, test_df = self.testMix_read_input()
        cols = ["hour", "minute", "ampm", "dayofWeek"]
        for df in [train_df, test_df]:
            df["hour"] = df["planArrival"].map(lambda x : x[:2]).astype(int)
            df["minute"] = df["planArrival"].map(lambda x : x[3:]).astype(int)
            df["ampm"] = df["hour"].map(lambda x : 1 if x < 15 else 0)
            df["dayofWeek"] = df["date"].map(lambda x : datetime.datetime(x//10000, (x%10000)//100, (x%100)).strftime('%A'))

        return train_df[cols], test_df[cols]