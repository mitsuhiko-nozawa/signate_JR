import os.path as osp
import datetime
import numpy as np
import pandas as pd
from .base import Feature

from sklearn.model_selection import KFold

class testAllMix_id(Feature):
    def create_features(self):
        return self.testAllMix_create_default_features(dtypes={self.name : "int64"})

class testAllMix_date(Feature):
    def create_features(self):
        return self.testAllMix_create_default_features(dtypes={self.name : "int64"})

class testAllMix_lineName(Feature):
    def create_features(self):
        return self.testAllMix_create_default_features(dtypes={self.name : "object"})

class testAllMix_directionCode(Feature):
    def create_features(self):
        return self.testAllMix_create_default_features(dtypes={self.name : "int64"})

class testAllMix_trainNo(Feature):
    def create_features(self):
        return self.testAllMix_create_default_features(dtypes={self.name : "object"})

class testAllMix_stopStation(Feature):
    def create_features(self):
        return self.testAllMix_create_default_features(dtypes={self.name : "object"})

class testAllMix_planArrival(Feature):
    def create_features(self):
        return self.testAllMix_create_default_features(dtypes={self.name : "object"})

class testAllMix_delayTime(Feature):
    def create_features(self):
        return self.testAllMix_create_default_features(dtypes={self.name : "int64"})


class testAllMix_continuedDelayTime(Feature):
    def create_features(self):
        train_df, test_df = self.testAllMix_read_input()
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




class testAllMix_date_TrainNo_count(Feature):
    def create_features(self):
        train_df, test_df = self.testAllMix_read_input()
        train_df[self.name] = train_df.groupby(["date", "trainNo"])["id"].transform("count")
        test_df[self.name] = test_df.groupby(["date", "trainNo"])["id"].transform("count")
        return train_df[[self.name]], test_df[[self.name]]

class testAllMix_desc_continuedDelayTime(Feature):
    def create_features(self):
        train_df, test_df = self.testAllMix_read_input()
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

class testAllMix_info(Feature):
    def create_features(self):
        train_df, test_df = self.testAllMix_read_input()
        for df in [train_df, test_df]:
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")))
            df["planArrival_int"] = df["planArrival"].map(lambda x : int(x.replace(":", "")))
            df["ampm"] = df["hour"].map(lambda x : "am" if x <= 14 else "pm")

        info_df = pd.read_csv(osp.join(self.ROOT, "input", "info.csv"))
        info_df["hour"] = info_df["time"].map(lambda x : int(x[:2].replace(":", "")))
        info_df["time_int"] = info_df["time"].map(lambda x : int(x.replace(":", "")))
        info_df["ampm"] = info_df["hour"].map(lambda x : "am" if x <= 14 else "pm")
        cses = info_df["cse"].unique()
        info_df = pd.concat([info_df.drop(columns=["cse"]), pd.get_dummies(info_df["cse"])], axis=1)
        info_g = info_df.groupby(["date", "ampm", "lineName"])

        info_t = info_g.sum()[cses]
        info_t["min_time"] = info_g["time_int"].min()
        info_t["min_time-30"] = info_t["min_time"] -30
        info_t["max_time"] = info_g["time_int"].max()
        info_t[cses] = info_t[cses].clip(upper=1)
        train_df = pd.merge(train_df, info_t.reset_index(), on=["date", "ampm", "lineName"], how="left", suffixes=["", "_info"])
        test_df = pd.merge(test_df, info_t.reset_index(), on=["date", "ampm", "lineName"], how="left", suffixes=["", "_info"])

        for df in [train_df, test_df]:
            df["is_cse"] = df.apply(lambda x : 1 if x["min_time-30"] <= x["planArrival_int"] <= x["max_time"] else 0, axis=1)
            for cse in cses:
                df[cse] = (df["is_cse"] * df[cse]).fillna(0)
            df["cseElapsedTime"] = (df["planArrival_int"] - df["min_time"]).fillna(-999)
        cols = list(cses) + ["cseElapsedTime"]

        return train_df[cols].astype("int32"), test_df[cols].astype("int32")

class testAllMix_dateTransformed(Feature):
    def create_features(self):
        train_df, test_df = self.testAllMix_read_input()
        cols = ["hour", "minute", "ampm", "dayofWeek"]
        for df in [train_df, test_df]:
            df["hour"] = df["planArrival"].map(lambda x : x[:2]).astype(int)
            #df["minute"] = df["planArrival"].map(lambda x : x[3:]).astype(int)
            df["ampm"] = df["hour"].map(lambda x : 1 if x < 15 else 0)
            df["dayofWeek"] = df["date"].map(lambda x : datetime.datetime(x//10000, (x%10000)//100, (x%100)).strftime('%A'))

        return train_df[cols], test_df[cols]

class testAllMix_targetAgg(Feature):
    def create_features(self):
        train_df, test_df = self.testAllMix_read_input()
        for df in [train_df, test_df]:
            df["hour"] = df["planArrival"].map(lambda x : x[:2]).astype(int)
            df["dayofWeek"] = df["date"].map(lambda x : datetime.datetime(x//10000, (x%10000)//100, (x%100)).strftime('%A'))
            df["planArrival_int"] = df["planArrival"].map(lambda x : int(x.replace(":", "")))
            df["planArrival_30binning"] = 0
            col = "planArrival_int"
            for t in np.arange(0, 2400, 100):
                df.loc[(t <= df[col]) & (df[col] < t+30), "planArrival_30binning"] = t
                df.loc[(t+30 <= df[col]) & (df[col] < t+60), "planArrival_30binning"] = t+30
        colss = [
            ["dayofWeek", "lineName", "directionCode", "planArrival_30binning"],
            ["dayofWeek", "lineName", "directionCode", "hour"],
            ["dayofWeek", "lineName", "directionCode"],
            ["lineName", "directionCode", "planArrival_30binning"],
            ["dayofWeek", "lineName", "planArrival_30binning"],
        ]
        res_cols = []
        for cols in colss:
            for method in ["mean", "median", "sum", "count", "var", "skew"]:
                feat_name = "_".join(cols) + "_" + method
                res_cols.append(feat_name)
                for df in [train_df, test_df]:
                    df[feat_name] = pd.concat([df[col].astype("str") + " * " for col in cols], axis=1).sum(axis=1)
                agg_df = train_df.groupby(feat_name)["delayTime"].agg(method)
                for df in [train_df, test_df]:
                    df[feat_name] = df[feat_name].map(agg_df)

        return train_df[res_cols], test_df[res_cols]

class testAllMix_targetMean(Feature):
    def create_features(self):
        train_df, test_df = self.testAllMix_read_input()
        for df in [train_df, test_df]:
            df["hour"] = df["planArrival"].map(lambda x : x[:2]).astype(int)
            df["dayofWeek"] = df["date"].map(lambda x : datetime.datetime(x//10000, (x%10000)//100, (x%100)).strftime('%A'))
            df["planArrival_int"] = df["planArrival"].map(lambda x : int(x.replace(":", "")))
            df["planArrival_30binning"] = 0
            col = "planArrival_int"
            for t in np.arange(0, 2400, 100):
                df.loc[(t <= df[col]) & (df[col] < t+30), "planArrival_30binning"] = t
                df.loc[(t+30 <= df[col]) & (df[col] < t+60), "planArrival_30binning"] = t+30
        colss = [
            ["dayofWeek", "lineName", "directionCode", "planArrival_30binning"],
            ["dayofWeek", "lineName", "directionCode", "hour"],
            ["dayofWeek", "lineName", "directionCode"],
            ["lineName", "directionCode", "planArrival_30binning"],
            ["dayofWeek", "lineName", "planArrival_30binning"],
        ]
        res_cols = []
        for cols in colss:
            for method in ["mean"]:
                feat_name = "_".join(cols) + "_" + method
                res_cols.append(feat_name)
                for df in [train_df, test_df]:
                    df[feat_name] = pd.concat([df[col].astype("str") + " * " for col in cols], axis=1).sum(axis=1)
                agg_df = train_df.groupby(feat_name)["delayTime"].agg(method)
                for df in [train_df, test_df]:
                    df[feat_name] = df[feat_name].map(agg_df)

        return train_df[res_cols], test_df[res_cols]