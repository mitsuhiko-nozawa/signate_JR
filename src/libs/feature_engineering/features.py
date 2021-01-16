import os.path as osp
import numpy as np
import pandas as pd
from .base import Feature
import datetime
from sklearn.model_selection import KFold

class id(Feature):
    def create_features(self):
        return self.create_default_features(dtypes={self.name : "int64"})

class date(Feature):
    def create_features(self):
        return self.create_default_features(dtypes={self.name : "int64"})

class lineName(Feature):
    def create_features(self):
        return self.create_default_features(dtypes={self.name : "object"})

class directionCode(Feature):
    def create_features(self):
        return self.create_default_features(dtypes={self.name : "int64"})

class trainNo(Feature):
    def create_features(self):
        return self.create_default_features(dtypes={self.name : "object"})

class stopStation(Feature):
    def create_features(self):
        return self.create_default_features(dtypes={self.name : "object"})

class planArrival(Feature):
    def create_features(self):
        return self.create_default_features(dtypes={self.name : "object"})

class delayTime(Feature):
    def create_features(self):
        return self.create_default_features(dtypes={self.name : "float64"})

class date_trainNo(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        for df in [train_df, test_df]:
            df[self.name] = df["date"].astype("str") + " * " + df["trainNo"]
        return train_df[[self.name]], test_df[[self.name]]

class continuedDelayTime_8(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        hour = 8
        for df in [train_df]:
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
            df[self.name] = df["delayTime"].values.copy()
            df.loc[df["hour"] >= hour, self.name] = np.nan
            df[self.name] = df.groupby(["date", "trainNo"])[self.name].transform(lambda x: x.fillna(method="ffill"))
            df["isnanDelayTime"] = df[self.name].isnull().astype("int")
            df["continuedDelayTime_Mean"] = df.groupby(["date", "lineName", "directionCode", "hour"])[self.name].transform("mean")
            df["continuedDelayTime_Mean2"] = df.groupby(["date", "lineName", "directionCode", "hour", "stopStation"])[self.name].transform("mean")
            df[self.name] = df[self.name].fillna(-999)
        test_df["hour"] = test_df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
        
        test_df[self.name] = test_df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x: x.fillna(method="ffill"))
        test_df["isnanDelayTime"] = test_df[self.name].isnull().astype("int")
        test_df["continuedDelayTime_Mean"] = test_df.groupby(["date", "lineName", "directionCode", "hour"])[self.name].transform("mean")
        test_df["continuedDelayTime_Mean2"] = test_df.groupby(["date", "lineName", "directionCode", "hour", "stopStation"])[self.name].transform("mean")
        test_df[self.name] = test_df[self.name].fillna(-999)
        return train_df[[self.name, "continuedDelayTime_Mean", "continuedDelayTime_Mean2", "isnanDelayTime"]], test_df[[self.name, "continuedDelayTime_Mean", "continuedDelayTime_Mean2", "isnanDelayTime"]]

class continuedDelayTime_9(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        hour = 9
        for df in [train_df]:
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
            df[self.name] = df["delayTime"].values.copy()
            df.loc[df["hour"] >= hour, self.name] = np.nan
            df[self.name] = df.groupby(["date", "trainNo"])[self.name].transform(lambda x: x.fillna(method="ffill"))
            df["isnanDelayTime"] = df[self.name].isnull().astype("int")
            df["continuedDelayTime_Mean"] = df.groupby(["date", "lineName", "directionCode", "hour"])[self.name].transform("mean")
            df["continuedDelayTime_Mean2"] = df.groupby(["date", "lineName", "directionCode", "hour", "stopStation"])[self.name].transform("mean")
            df[self.name] = df[self.name].fillna(-999)
        test_df["hour"] = test_df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
        test_df["isnanDelayTime"] = test_df["delayTime"].isnull().astype("int")
        test_df[self.name] = test_df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x: x.fillna(method="ffill"))
        test_df["continuedDelayTime_Mean"] = test_df.groupby(["date", "lineName", "directionCode", "hour"])[self.name].transform("mean")
        test_df["continuedDelayTime_Mean2"] = test_df.groupby(["date", "lineName", "directionCode", "hour", "stopStation"])[self.name].transform("mean")
        test_df[self.name] = test_df[self.name].fillna(-999)
        return train_df[[self.name, "continuedDelayTime_Mean", "continuedDelayTime_Mean2", "isnanDelayTime"]], test_df[[self.name, "continuedDelayTime_Mean", "continuedDelayTime_Mean2", "isnanDelayTime"]]

class continuedDelayTime_18(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        hour = 18
        for df in [train_df]:
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
            df[self.name] = df["delayTime"].values.copy()
            df.loc[df["hour"] >= hour, self.name] = np.nan
            df[self.name] = df.groupby(["date", "trainNo"])[self.name].transform(lambda x: x.fillna(method="ffill"))
            df["isnanDelayTime"] = df[self.name].isnull().astype("int")
            df["continuedDelayTime_Mean"] = df.groupby(["date", "lineName", "directionCode", "hour"])[self.name].transform("mean")
            df["continuedDelayTime_Mean2"] = df.groupby(["date", "lineName", "directionCode", "hour", "stopStation"])[self.name].transform("mean")
            df[self.name] = df[self.name].fillna(-999)
        test_df["hour"] = test_df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
        test_df["isnanDelayTime"] = test_df["delayTime"].isnull().astype("int")
        test_df[self.name] = test_df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x: x.fillna(method="ffill"))
        test_df["continuedDelayTime_Mean"] = test_df.groupby(["date", "lineName", "directionCode", "hour"])[self.name].transform("mean")
        test_df["continuedDelayTime_Mean2"] = test_df.groupby(["date", "lineName", "directionCode", "hour", "stopStation"])[self.name].transform("mean")
        test_df[self.name] = test_df[self.name].fillna(-999)
        return train_df[[self.name, "continuedDelayTime_Mean", "continuedDelayTime_Mean2", "isnanDelayTime"]], test_df[[self.name, "continuedDelayTime_Mean", "continuedDelayTime_Mean2", "isnanDelayTime"]]

class continuedDelayTime_19(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        hour = 19
        for df in [train_df]:
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
            df[self.name] = df["delayTime"].values.copy()
            df.loc[df["hour"] >= hour, self.name] = np.nan
            df[self.name] = df.groupby(["date", "trainNo"])[self.name].transform(lambda x: x.fillna(method="ffill"))
            df["isnanDelayTime"] = df[self.name].isnull().astype("int")
            df["continuedDelayTime_Mean"] = df.groupby(["date", "lineName", "directionCode", "hour"])[self.name].transform("mean")
            df["continuedDelayTime_Mean2"] = df.groupby(["date", "lineName", "directionCode", "hour", "stopStation"])[self.name].transform("mean")
            df[self.name] = df[self.name].fillna(-999)
        test_df["hour"] = test_df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
        test_df["isnanDelayTime"] = test_df["delayTime"].isnull().astype("int")
        test_df[self.name] = test_df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x: x.fillna(method="ffill"))
        test_df["continuedDelayTime_Mean"] = test_df.groupby(["date", "lineName", "directionCode", "hour"])[self.name].transform("mean")
        test_df["continuedDelayTime_Mean2"] = test_df.groupby(["date", "lineName", "directionCode", "hour", "stopStation"])[self.name].transform("mean")
        test_df[self.name] = test_df[self.name].fillna(-999)
        return train_df[[self.name, "continuedDelayTime_Mean", "continuedDelayTime_Mean2", "isnanDelayTime"]], test_df[[self.name, "continuedDelayTime_Mean", "continuedDelayTime_Mean2", "isnanDelayTime"]]


class date_TrainNo_count(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        cols = ["dateAmpmTrainNo_Count", "trainSeq_Order", "trainSeq_Ratio", "is_firstTrain", "is_lastTrain"]
        for df in [train_df, test_df]:
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
            df["ampm"] = df["hour"].map(lambda x : "am" if x < 15 else "pm")
            df["dateAmpmTrainNo_Count"] = df.groupby(["date", "ampm", "trainNo"])["id"].transform("count")
            df["trainSeq_Order"] = df.groupby(["date", "ampm", "trainNo"])["id"].transform(lambda x : x - x.iloc[0] + 1)
            df["trainSeq_Ratio"] = df["trainSeq_Order"] / df["dateAmpmTrainNo_Count"]
            df["is_firstTrain"] = (df["trainSeq_Order"] == 1).astype("int")
            df["is_lastTrain"] = (df["trainSeq_Order"] == df["dateAmpmTrainNo_Count"]).astype("int")

        return train_df[cols], test_df[cols]


class info(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        info_df = pd.read_csv(osp.join(self.ROOT, "input", "info.csv"))
        info_df = info_df.groupby(["date", "lineName"])["cse"].unique().reset_index()
        info_df["cse"] = info_df["cse"].map(lambda x : "".join(sorted(x)))
        train_df = train_df.merge(info_df, on=["date", "lineName"], how="left")
        test_df = test_df.merge(info_df, on=["date", "lineName"], how="left")
        train_df["cse"].fillna("None", inplace=True)
        test_df["cse"].fillna("None", inplace=True)
        return train_df[["cse"]], test_df[["cse"]]

class dateTransformed(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        cols = ["planArrival_int", "hour", "ampm", "dayofWeek", "isWeekDay"]
        for df in [train_df, test_df]:
            df["planArrival_int"] = df["planArrival"].map(lambda x : int(x.replace(":", "")))
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
            df["ampm"] = df["hour"].map(lambda x : 1 if x < 15 else 0)
            df["dayofWeek"] = df["date"].map(lambda x : datetime.datetime(x//10000, (x%10000)//100, (x%100)).strftime('%A'))
            df["isWeekDay"] = df["dayofWeek"].map(lambda x : 0 if x in ["Saturday", "Sunday"] else 1)
        return train_df[cols], test_df[cols]

class hour_1_targetMean1(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        cols = ["date"]
        for df in [train_df, test_df]:
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
            df["hour-1"] = df["hour"] - 1
            agg_df = df.groupby(cols+["hour"])["delayTime"].mean().reset_index().rename(columns={"delayTime" : self.name, "hour" : "hour-1"})
            temp = pd.merge(df, agg_df, on=cols+["hour-1"], how="left")
            df[self.name] = temp[self.name]

        return train_df[[self.name]], test_df[[self.name]]

class hour_1_targetMean2(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        cols = ["date", "lineName"]
        for df in [train_df, test_df]:
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
            df["hour-1"] = df["hour"] - 1
            agg_df = df.groupby(cols+["hour"])["delayTime"].mean().reset_index().rename(columns={"delayTime" : self.name, "hour" : "hour-1"})
            temp = pd.merge(df, agg_df, on=cols+["hour-1"], how="left")
            df[self.name] = temp[self.name]

        return train_df[[self.name]], test_df[[self.name]]

class hour_1_targetMean3(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        cols = ["date", "lineName", "directionCode"]
        for df in [train_df, test_df]:
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
            df["hour-1"] = df["hour"] - 1
            agg_df = df.groupby(cols+["hour"])["delayTime"].mean().reset_index().rename(columns={"delayTime" : self.name, "hour" : "hour-1"})
            temp = pd.merge(df, agg_df, on=cols+["hour-1"], how="left")
            df[self.name] = temp[self.name]

        return train_df[[self.name]], test_df[[self.name]]

class hour_1_targetMean4(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        cols = ["date", "lineName", "stopStation", "directionCode"]
        for df in [train_df, test_df]:
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
            df["hour-1"] = df["hour"] - 1
            agg_df = df.groupby(cols+["hour"])["delayTime"].mean().reset_index().rename(columns={"delayTime" : self.name, "hour" : "hour-1"})
            temp = pd.merge(df, agg_df, on=cols+["hour-1"], how="left")
            df[self.name] = temp[self.name]

        return train_df[[self.name]], test_df[[self.name]]

class minute_10_targetMean(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        cols = ["date", "lineName", "directionCode"]
        for df in [train_df, test_df]:
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
            df["hour-1"] = df["hour"] - 1
            df["minute"] = df["planArrival"].map(lambda x : int(x[3:]))
            agg_df = df[(df["minute"] >= 51) | (df["minute"] == 0)].groupby(cols+["hour"])["delayTime"].mean().reset_index().rename(columns={"delayTime" : self.name, "hour" : "hour-1"})
            temp = pd.merge(df, agg_df, on=cols+["hour-1"], how="left")
            df[self.name] = temp[self.name]
        return train_df[[self.name]], test_df[[self.name]]


class zinshin(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        info_df = pd.read_csv(osp.join(self.ROOT, "input", "info.csv"))
        info_df["time_int"] = info_df["time"].map(lambda x : int(x.replace(":", "")))
        zinshin = info_df[info_df["cse"] == "人身事故"].groupby(["date", "lineName"]).agg({"time_int" : ["min", "max"]})
        zinshin.columns=["time_min", "time_max"]
        zinshin = zinshin.reset_index()
        for df in [train_df, test_df]:
            df["planArrival_int"] = df["planArrival"].map(lambda x : int(x.replace(":", "")))
            df[["time_min", "time_max"]] = pd.merge(df, zinshin, on=["date", "lineName"], how="left")[["time_min", "time_max"]]
            df["is_zinshin"] = np.abs(df["time_min"].isnull().astype("int") - 1)
            df["is_zinshinPeriod"] = df.apply(lambda x : 1 if x["time_min"] <= x["planArrival_int"] <= x["time_max"] else 0, axis=1)
            def func(x):
                t_min = x["time_min"]
                t_max = x["time_max"]
                t = x["planArrival_int"]
                t_min_30 = t_min - 30 if t_min % 100 >= 30 else t_min - 70
                return int(t_min_30 <= t <= t_max)
            df["is_zinshinPeriod_30"] = df.apply(func, axis=1)
            df["elapsedTime_zhinshin"] = df["planArrival_int"] - train_df["time_min"]
        cols = ["is_zinshin", "is_zinshinPeriod", "is_zinshinPeriod_30", "elapsedTime_zhinshin"]
        return train_df[cols], test_df[cols]

class prod_feature(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        use_cols = []
        colss = [
            ["date", "ampm", "lineName", "directionCode"],
        ]
        for df in [train_df, test_df]:
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
            df["ampm"] = df["hour"].map(lambda x : "am" if x < 15 else "pm")

        for i, cols in enumerate(colss):
            feat_name = f"prod_feat_{i}"
            use_cols.append(feat_name)
            for df in [train_df, test_df]:
                df[feat_name] = ""
                df[feat_name] = df[feat_name].str.cat(df[col].astype(str) for col in cols)


        return train_df[use_cols], test_df[use_cols]

class stationDetail(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        use_cols = ["lastStation", "prevStation", "nextStation", "lastStation_prevMean_from30", "nextStation_prevMean_from30"]
        for df in [train_df, test_df]:
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
            df["hour-1"] = df["hour"] - 1
            df["minute"] = df["planArrival"].map(lambda x : (int(x[3:])+59)%60) # 01 -> 00分
            df["lastStation"] = df.groupby(["date", "trainNo"])["stopStation"].transform(lambda x : x.iloc[x.shape[0]-1])
            df["nextStation"] = df.groupby(["date", "trainNo"])["stopStation"].transform(lambda x : x.shift(-1).fillna("None"))
            df["prevStation"] = df.groupby(["date", "trainNo"])["stopStation"].transform(lambda x : x.shift(1).fillna("None"))

            cols = ["date", "lineName", "directionCode"]
            col = "lastStation_prevMean_from30"
            agg_df = df[df["minute"] >= 30].groupby(cols+["stopStation", "hour"])["delayTime"].mean().reset_index().rename(columns={"delayTime" : col, "hour" : "hour-1", "stopStation" : "lastStation"})
            temp = pd.merge(df, agg_df, on=cols+["lastStation", "hour-1"], how="left")
            df[col] = temp[col].values.copy()

            col2 = "nextStation_prevMean_from30"
            agg_df = agg_df.rename(columns={col : col2, "lastStation" : "nextStation"})
            temp = pd.merge(df, agg_df, on=cols+["nextStation", "hour-1"], how="left")
            df[col2] = temp[col].values.copy()




        return train_df[use_cols], test_df[use_cols]

class continuedFeatures_8(Feature):
    def create_features(self):
        # 自分の先にいる電車、後にいる電車
        # 自分の5個前までの遅延時間
        h = 8
        train_df, test_df = self.read_input()
        use_cols = ["delayTime_11", "delayTime_ahead_10", "delayTime_2ahead_9", "delayTime_3ahead_8", "delayTime_4ahead_7"]
        for df in [train_df, test_df]:
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
            ###
            df.sort_values("planArrival", inplace=True)
            cols = ["date", "lineName", "directionCode", "stopStation"]
            temp_g = df.groupby(cols)
            df["trainNo_ahead"] = temp_g["trainNo"].transform(lambda x : x.shift(1).fillna("None")).copy()
            df["planArrival_ahead"] = temp_g["planArrival"].transform(lambda x : x.shift(1).fillna("None")).copy()
            df["trainNo_2ahead"] = temp_g["trainNo"].transform(lambda x : x.shift(2).fillna("None")).copy()
            df["planArrival_2ahead"] = temp_g["planArrival"].transform(lambda x : x.shift(2).fillna("None")).copy()
            df["trainNo_3ahead"] = temp_g["trainNo"].transform(lambda x : x.shift(3).fillna("None")).copy()
            df["planArrival_3ahead"] = temp_g["planArrival"].transform(lambda x : x.shift(3).fillna("None")).copy()
            df["trainNo_4ahead"] = temp_g["trainNo"].transform(lambda x : x.shift(4).fillna("None")).copy()
            df["planArrival_4ahead"] = temp_g["planArrival"].transform(lambda x : x.shift(4).fillna("None")).copy()

            #df["trainNo_behind"] = temp_g["trainNo"].transform(lambda x : x.shift(-1).fillna("None")).copy()
            #df["planArrival_behind"] = temp_g["planArrival"].transform(lambda x : x.shift(-1).fillna("None")).copy()
            df.sort_values("id", inplace=True)

            df.loc[df["hour"] == h, "delayTime"] = -9999
            #df["delayTime_1"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(1).fillna(-999))
            #df["delayTime_2"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(2).fillna(-999))
            #df["delayTime_3"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(3).fillna(-999))
            #df["delayTime_4"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(4).fillna(-999))
            #df["delayTime_5"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(5).fillna(-999))
            #df["delayTime_6"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(6).fillna(-999))
            df["delayTime_7"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(7).fillna(-999))
            df["delayTime_8"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(8).fillna(-999))
            df["delayTime_9"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(9).fillna(-999))
            df["delayTime_10"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(10).fillna(-999))
            df["delayTime_11"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(11).fillna(-999))
            #df["delayTime_12"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(12).fillna(-999))
            ###
            col = "delayTime_ahead_10"
            cols = ["date", "lineName", "directionCode"]
            temp = df[cols+["planArrival", "trainNo", "delayTime_10"]].copy().rename(columns={"trainNo":"trainNo_ahead", "planArrival" : "planArrival_ahead", "delayTime_10":col})
            temp = temp.groupby(cols+["trainNo_ahead", "planArrival_ahead"]).mean().reset_index()
            df[col] = pd.merge(df, temp, on=cols+["trainNo_ahead", "planArrival_ahead"], how="left")[col].values.copy()
            ###
            col = "delayTime_2ahead_9"
            cols = ["date", "lineName", "directionCode"]
            temp = df[cols+["planArrival", "trainNo", "delayTime_9"]].copy().rename(columns={"trainNo":"trainNo_2ahead", "planArrival" : "planArrival_2ahead", "delayTime_9":col})
            temp = temp.groupby(cols+["trainNo_2ahead", "planArrival_2ahead"]).mean().reset_index()
            df[col] = pd.merge(df, temp, on=cols+["trainNo_2ahead", "planArrival_2ahead"], how="left")[col].values.copy()
            ###
            col = "delayTime_3ahead_8"
            cols = ["date", "lineName", "directionCode"]
            temp = df[cols+["planArrival", "trainNo", "delayTime_8"]].copy().rename(columns={"trainNo":"trainNo_3ahead", "planArrival" : "planArrival_3ahead", "delayTime_8":col})
            temp = temp.groupby(cols+["trainNo_3ahead", "planArrival_3ahead"]).mean().reset_index()
            df[col] = pd.merge(df, temp, on=cols+["trainNo_3ahead", "planArrival_3ahead"], how="left")[col].values.copy()
            ###
            col = "delayTime_4ahead_7"
            cols = ["date", "lineName", "directionCode"]
            temp = df[cols+["planArrival", "trainNo", "delayTime_7"]].copy().rename(columns={"trainNo":"trainNo_4ahead", "planArrival" : "planArrival_4ahead", "delayTime_7":col})
            temp = temp.groupby(cols+["trainNo_4ahead", "planArrival_4ahead"]).mean().reset_index()
            df[col] = pd.merge(df, temp, on=cols+["trainNo_4ahead", "planArrival_4ahead"], how="left")[col].values.copy()
            ###
        return train_df[use_cols], test_df[use_cols]

class continuedFeatures_9(Feature):
    def create_features(self):
        # 自分の先にいる電車、後にいる電車
        # 自分の5個前までの遅延時間
        h = 9
        train_df, test_df = self.read_input()
        use_cols = ["delayTime_10", "delayTime_11", "delayTime_12", "delayTime_ahead_10", "delayTime_2ahead_9", "delayTime_3ahead_8", "delayTime_4ahead_7"]
        for df in [train_df, test_df]:
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
            ###
            df.sort_values("planArrival", inplace=True)
            cols = ["date", "lineName", "directionCode", "stopStation"]
            temp_g = df.groupby(cols)
            df["trainNo_ahead"] = temp_g["trainNo"].transform(lambda x : x.shift(1).fillna("None")).copy()
            df["planArrival_ahead"] = temp_g["planArrival"].transform(lambda x : x.shift(1).fillna("None")).copy()
            df["trainNo_2ahead"] = temp_g["trainNo"].transform(lambda x : x.shift(2).fillna("None")).copy()
            df["planArrival_2ahead"] = temp_g["planArrival"].transform(lambda x : x.shift(2).fillna("None")).copy()
            df["trainNo_3ahead"] = temp_g["trainNo"].transform(lambda x : x.shift(3).fillna("None")).copy()
            df["planArrival_3ahead"] = temp_g["planArrival"].transform(lambda x : x.shift(3).fillna("None")).copy()
            df["trainNo_4ahead"] = temp_g["trainNo"].transform(lambda x : x.shift(4).fillna("None")).copy()
            df["planArrival_4ahead"] = temp_g["planArrival"].transform(lambda x : x.shift(4).fillna("None")).copy()

            #df["trainNo_behind"] = temp_g["trainNo"].transform(lambda x : x.shift(-1).fillna("None")).copy()
            #df["planArrival_behind"] = temp_g["planArrival"].transform(lambda x : x.shift(-1).fillna("None")).copy()
            df.sort_values("id", inplace=True)

            df.loc[df["hour"] == h, "delayTime"] = -9999
            #df["delayTime_1"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(1).fillna(-999))
            #df["delayTime_2"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(2).fillna(-999))
            #df["delayTime_3"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(3).fillna(-999))
            #df["delayTime_4"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(4).fillna(-999))
            #df["delayTime_5"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(5).fillna(-999))
            #df["delayTime_6"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(6).fillna(-999))
            df["delayTime_7"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(7).fillna(-999))
            df["delayTime_8"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(8).fillna(-999))
            df["delayTime_9"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(9).fillna(-999))
            df["delayTime_10"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(10).fillna(-999))
            df["delayTime_11"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(11).fillna(-999))
            df["delayTime_12"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(12).fillna(-999))
            ###
            col = "delayTime_ahead_10"
            cols = ["date", "lineName", "directionCode"]
            temp = df[cols+["planArrival", "trainNo", "delayTime_10"]].copy().rename(columns={"trainNo":"trainNo_ahead", "planArrival" : "planArrival_ahead", "delayTime_10":col})
            temp = temp.groupby(cols+["trainNo_ahead", "planArrival_ahead"]).mean().reset_index()
            df[col] = pd.merge(df, temp, on=cols+["trainNo_ahead", "planArrival_ahead"], how="left")[col].values.copy()
            ###
            col = "delayTime_2ahead_9"
            cols = ["date", "lineName", "directionCode"]
            temp = df[cols+["planArrival", "trainNo", "delayTime_9"]].copy().rename(columns={"trainNo":"trainNo_2ahead", "planArrival" : "planArrival_2ahead", "delayTime_9":col})
            temp = temp.groupby(cols+["trainNo_2ahead", "planArrival_2ahead"]).mean().reset_index()
            df[col] = pd.merge(df, temp, on=cols+["trainNo_2ahead", "planArrival_2ahead"], how="left")[col].values.copy()
            ###
            col = "delayTime_3ahead_8"
            cols = ["date", "lineName", "directionCode"]
            temp = df[cols+["planArrival", "trainNo", "delayTime_8"]].copy().rename(columns={"trainNo":"trainNo_3ahead", "planArrival" : "planArrival_3ahead", "delayTime_8":col})
            temp = temp.groupby(cols+["trainNo_3ahead", "planArrival_3ahead"]).mean().reset_index()
            df[col] = pd.merge(df, temp, on=cols+["trainNo_3ahead", "planArrival_3ahead"], how="left")[col].values.copy()
            ###
            col = "delayTime_4ahead_7"
            cols = ["date", "lineName", "directionCode"]
            temp = df[cols+["planArrival", "trainNo", "delayTime_7"]].copy().rename(columns={"trainNo":"trainNo_4ahead", "planArrival" : "planArrival_4ahead", "delayTime_7":col})
            temp = temp.groupby(cols+["trainNo_4ahead", "planArrival_4ahead"]).mean().reset_index()
            df[col] = pd.merge(df, temp, on=cols+["trainNo_4ahead", "planArrival_4ahead"], how="left")[col].values.copy()
            ###
        return train_df[use_cols], test_df[use_cols]

class continuedFeatures_18(Feature):
    def create_features(self):
        # 自分の先にいる電車、後にいる電車
        # 自分の5個前までの遅延時間
        h = 18
        train_df, test_df = self.read_input()
        use_cols = ["delayTime_11", "delayTime_ahead_10", "delayTime_2ahead_9", "delayTime_3ahead_8", "delayTime_4ahead_7"]
        for df in [train_df, test_df]:
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
            ###
            df.sort_values("planArrival", inplace=True)
            cols = ["date", "lineName", "directionCode", "stopStation"]
            temp_g = df.groupby(cols)
            df["trainNo_ahead"] = temp_g["trainNo"].transform(lambda x : x.shift(1).fillna("None")).copy()
            df["planArrival_ahead"] = temp_g["planArrival"].transform(lambda x : x.shift(1).fillna("None")).copy()
            df["trainNo_2ahead"] = temp_g["trainNo"].transform(lambda x : x.shift(2).fillna("None")).copy()
            df["planArrival_2ahead"] = temp_g["planArrival"].transform(lambda x : x.shift(2).fillna("None")).copy()
            df["trainNo_3ahead"] = temp_g["trainNo"].transform(lambda x : x.shift(3).fillna("None")).copy()
            df["planArrival_3ahead"] = temp_g["planArrival"].transform(lambda x : x.shift(3).fillna("None")).copy()
            df["trainNo_4ahead"] = temp_g["trainNo"].transform(lambda x : x.shift(4).fillna("None")).copy()
            df["planArrival_4ahead"] = temp_g["planArrival"].transform(lambda x : x.shift(4).fillna("None")).copy()

            #df["trainNo_behind"] = temp_g["trainNo"].transform(lambda x : x.shift(-1).fillna("None")).copy()
            #df["planArrival_behind"] = temp_g["planArrival"].transform(lambda x : x.shift(-1).fillna("None")).copy()
            df.sort_values("id", inplace=True)

            df.loc[df["hour"] == h, "delayTime"] = -9999
            #df["delayTime_1"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(1).fillna(-999))
            #df["delayTime_2"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(2).fillna(-999))
            #df["delayTime_3"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(3).fillna(-999))
            #df["delayTime_4"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(4).fillna(-999))
            #df["delayTime_5"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(5).fillna(-999))
            #df["delayTime_6"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(6).fillna(-999))
            df["delayTime_7"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(7).fillna(-999))
            df["delayTime_8"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(8).fillna(-999))
            df["delayTime_9"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(9).fillna(-999))
            df["delayTime_10"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(10).fillna(-999))
            df["delayTime_11"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(11).fillna(-999))
            df["delayTime_12"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(12).fillna(-999))
            ###
            col = "delayTime_ahead_10"
            cols = ["date", "lineName", "directionCode"]
            temp = df[cols+["planArrival", "trainNo", "delayTime_10"]].copy().rename(columns={"trainNo":"trainNo_ahead", "planArrival" : "planArrival_ahead", "delayTime_10":col})
            temp = temp.groupby(cols+["trainNo_ahead", "planArrival_ahead"]).mean().reset_index()
            df[col] = pd.merge(df, temp, on=cols+["trainNo_ahead", "planArrival_ahead"], how="left")[col].values.copy()
            ###
            col = "delayTime_2ahead_9"
            cols = ["date", "lineName", "directionCode"]
            temp = df[cols+["planArrival", "trainNo", "delayTime_9"]].copy().rename(columns={"trainNo":"trainNo_2ahead", "planArrival" : "planArrival_2ahead", "delayTime_9":col})
            temp = temp.groupby(cols+["trainNo_2ahead", "planArrival_2ahead"]).mean().reset_index()
            df[col] = pd.merge(df, temp, on=cols+["trainNo_2ahead", "planArrival_2ahead"], how="left")[col].values.copy()
            ###
            col = "delayTime_3ahead_8"
            cols = ["date", "lineName", "directionCode"]
            temp = df[cols+["planArrival", "trainNo", "delayTime_8"]].copy().rename(columns={"trainNo":"trainNo_3ahead", "planArrival" : "planArrival_3ahead", "delayTime_8":col})
            temp = temp.groupby(cols+["trainNo_3ahead", "planArrival_3ahead"]).mean().reset_index()
            df[col] = pd.merge(df, temp, on=cols+["trainNo_3ahead", "planArrival_3ahead"], how="left")[col].values.copy()
            ###
            col = "delayTime_4ahead_7"
            cols = ["date", "lineName", "directionCode"]
            temp = df[cols+["planArrival", "trainNo", "delayTime_7"]].copy().rename(columns={"trainNo":"trainNo_4ahead", "planArrival" : "planArrival_4ahead", "delayTime_7":col})
            temp = temp.groupby(cols+["trainNo_4ahead", "planArrival_4ahead"]).mean().reset_index()
            df[col] = pd.merge(df, temp, on=cols+["trainNo_4ahead", "planArrival_4ahead"], how="left")[col].values.copy()
            ###
        return train_df[use_cols], test_df[use_cols]

class continuedFeatures_19(Feature):
    def create_features(self):
        # 自分の先にいる電車、後にいる電車
        # 自分の5個前までの遅延時間
        h = 19
        train_df, test_df = self.read_input()
        use_cols = ["delayTime_11", "delayTime_ahead_10", "delayTime_2ahead_9", "delayTime_3ahead_8", "delayTime_4ahead_7"]
        for df in [train_df, test_df]:
            df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
            ###
            df.sort_values("planArrival", inplace=True)
            cols = ["date", "lineName", "directionCode", "stopStation"]
            temp_g = df.groupby(cols)
            df["trainNo_ahead"] = temp_g["trainNo"].transform(lambda x : x.shift(1).fillna("None")).copy()
            df["planArrival_ahead"] = temp_g["planArrival"].transform(lambda x : x.shift(1).fillna("None")).copy()
            df["trainNo_2ahead"] = temp_g["trainNo"].transform(lambda x : x.shift(2).fillna("None")).copy()
            df["planArrival_2ahead"] = temp_g["planArrival"].transform(lambda x : x.shift(2).fillna("None")).copy()
            df["trainNo_3ahead"] = temp_g["trainNo"].transform(lambda x : x.shift(3).fillna("None")).copy()
            df["planArrival_3ahead"] = temp_g["planArrival"].transform(lambda x : x.shift(3).fillna("None")).copy()
            df["trainNo_4ahead"] = temp_g["trainNo"].transform(lambda x : x.shift(4).fillna("None")).copy()
            df["planArrival_4ahead"] = temp_g["planArrival"].transform(lambda x : x.shift(4).fillna("None")).copy()

            #df["trainNo_behind"] = temp_g["trainNo"].transform(lambda x : x.shift(-1).fillna("None")).copy()
            #df["planArrival_behind"] = temp_g["planArrival"].transform(lambda x : x.shift(-1).fillna("None")).copy()
            df.sort_values("id", inplace=True)

            df.loc[df["hour"] == h, "delayTime"] = -9999
            #df["delayTime_1"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(1).fillna(-999))
            #df["delayTime_2"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(2).fillna(-999))
            #df["delayTime_3"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(3).fillna(-999))
            #df["delayTime_4"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(4).fillna(-999))
            #df["delayTime_5"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(5).fillna(-999))
            #df["delayTime_6"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(6).fillna(-999))
            df["delayTime_7"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(7).fillna(-999))
            df["delayTime_8"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(8).fillna(-999))
            df["delayTime_9"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(9).fillna(-999))
            df["delayTime_10"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(10).fillna(-999))
            df["delayTime_11"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(11).fillna(-999))
            df["delayTime_12"] = df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x : x.shift(12).fillna(-999))
            ###
            col = "delayTime_ahead_10"
            cols = ["date", "lineName", "directionCode"]
            temp = df[cols+["planArrival", "trainNo", "delayTime_10"]].copy().rename(columns={"trainNo":"trainNo_ahead", "planArrival" : "planArrival_ahead", "delayTime_10":col})
            temp = temp.groupby(cols+["trainNo_ahead", "planArrival_ahead"]).mean().reset_index()
            df[col] = pd.merge(df, temp, on=cols+["trainNo_ahead", "planArrival_ahead"], how="left")[col].values.copy()
            ###
            col = "delayTime_2ahead_9"
            cols = ["date", "lineName", "directionCode"]
            temp = df[cols+["planArrival", "trainNo", "delayTime_9"]].copy().rename(columns={"trainNo":"trainNo_2ahead", "planArrival" : "planArrival_2ahead", "delayTime_9":col})
            temp = temp.groupby(cols+["trainNo_2ahead", "planArrival_2ahead"]).mean().reset_index()
            df[col] = pd.merge(df, temp, on=cols+["trainNo_2ahead", "planArrival_2ahead"], how="left")[col].values.copy()
            ###
            col = "delayTime_3ahead_8"
            cols = ["date", "lineName", "directionCode"]
            temp = df[cols+["planArrival", "trainNo", "delayTime_8"]].copy().rename(columns={"trainNo":"trainNo_3ahead", "planArrival" : "planArrival_3ahead", "delayTime_8":col})
            temp = temp.groupby(cols+["trainNo_3ahead", "planArrival_3ahead"]).mean().reset_index()
            df[col] = pd.merge(df, temp, on=cols+["trainNo_3ahead", "planArrival_3ahead"], how="left")[col].values.copy()
            ###
            col = "delayTime_4ahead_7"
            cols = ["date", "lineName", "directionCode"]
            temp = df[cols+["planArrival", "trainNo", "delayTime_7"]].copy().rename(columns={"trainNo":"trainNo_4ahead", "planArrival" : "planArrival_4ahead", "delayTime_7":col})
            temp = temp.groupby(cols+["trainNo_4ahead", "planArrival_4ahead"]).mean().reset_index()
            df[col] = pd.merge(df, temp, on=cols+["trainNo_4ahead", "planArrival_4ahead"], how="left")[col].values.copy()
            ###
        return train_df[use_cols], test_df[use_cols]



class zinshin2(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        info_df = pd.read_csv(osp.join(self.ROOT, "input", "info.csv"))
        info_df["time_int"] = info_df["time"].map(lambda x : int(x.replace(":", "")))
        zinshin = info_df[info_df["cse"] == "人身事故"].groupby(["date", "lineName"]).agg({"time_int" : ["min", "max"]})
        zinshin.columns=["time_min", "time_max"]
        zinshin = zinshin.reset_index()
        for df in [train_df, test_df]:
            df["planArrival_int"] = df["planArrival"].map(lambda x : int(x.replace(":", "")))
            df[["time_min", "time_max"]] = pd.merge(df, zinshin, on=["date", "lineName"], how="left")[["time_min", "time_max"]]
            df["is_zinshin"] = np.abs(df["time_min"].isnull().astype("int") - 1)
            df["is_zinshinPeriod"] = df.apply(lambda x : 1 if x["time_min"] <= x["planArrival_int"] <= x["time_max"] else 0, axis=1)
            def func(x):
                t_min = x["time_min"]
                t_max = x["time_max"]
                t = x["planArrival_int"]
                t_min_30 = t_min - 30 if t_min % 100 >= 30 else t_min - 70
                return int(t_min_30 <= t <= t_max)
            df["is_zinshinPeriod_30"] = df.apply(func, axis=1)
            df["elapsedTime_zhinshin"] = df["planArrival_int"] - train_df["time_min"]
        cols = ["is_zinshin", "is_zinshinPeriod", "is_zinshinPeriod_30", "elapsedTime_zhinshin"]
        return train_df[cols], test_df[cols]