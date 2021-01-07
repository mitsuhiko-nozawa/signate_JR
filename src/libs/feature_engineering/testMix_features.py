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