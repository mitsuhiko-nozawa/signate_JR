import numpy as np
import pandas as pd
import sys, os
import warnings
warnings.filterwarnings("ignore")


def main():
    train_df = pd.read_csv("../../input/train.csv")
    test_df = pd.read_csv("../../input/test.csv")
    pred_zone = []

    train_df[["isnanDelayTime_8", "continuedDelayTime_8"]] = pd.read_feather("../../my_features/train/continuedDelayTime_8.feather")[["isnanDelayTime", "continuedDelayTime_8"]]
    train_df[["isnanDelayTime_9", "continuedDelayTime_9"]] = pd.read_feather("../../my_features/train/continuedDelayTime_9.feather")[["isnanDelayTime", "continuedDelayTime_9"]]
    train_df[["isnanDelayTime_18", "continuedDelayTime_18"]] = pd.read_feather("../../my_features/train/continuedDelayTime_18.feather")[["isnanDelayTime", "continuedDelayTime_18"]]
    train_df[["isnanDelayTime_19", "continuedDelayTime_19"]] = pd.read_feather("../../my_features/train/continuedDelayTime_19.feather")[["isnanDelayTime", "continuedDelayTime_19"]]
    for df in [train_df]:
        df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
    train_df["pred_2"] = np.nan
    mask_8 = (train_df["hour"] == 8) & (train_df["isnanDelayTime_8"] == 0)
    mask_9 = (train_df["hour"] == 9) & (train_df["isnanDelayTime_9"] == 0)
    mask_18 = (train_df["hour"] == 18) & (train_df["isnanDelayTime_18"] == 0)
    mask_19 = (train_df["hour"] == 19) & (train_df["isnanDelayTime_19"] == 0)
    train_df.loc[mask_8, "pred_2"] = train_df[mask_8]["continuedDelayTime_8"].values
    train_df.loc[mask_9, "pred_2"] = train_df[mask_9]["continuedDelayTime_9"].values
    train_df.loc[mask_18, "pred_2"] = train_df[mask_18]["continuedDelayTime_18"].values
    train_df.loc[mask_19, "pred_2"] = train_df[mask_19]["continuedDelayTime_19"].values
    train_df.loc[train_df["pred_2"] == -999, "pred_2"] = np.nan
    train_df["isnan_pred"] = train_df["pred_2"].isnull().astype("int")

    # =========================== calculate cv ===========================
    df_8_c = pd.read_csv("../../experiments/exp_032/val_preds/oof_preds.csv") 
    df_9_c = pd.read_csv("../../experiments/exp_033/val_preds/oof_preds.csv") 
    df_18_c = pd.read_csv("../../experiments/exp_034/val_preds/oof_preds.csv") 
    df_19_c = pd.read_csv("../../experiments/exp_035/val_preds/oof_preds.csv") 
    df_8_i = pd.read_csv("../../experiments/exp_036/val_preds/oof_preds.csv") 
    df_9_i = pd.read_csv("../../experiments/exp_037/val_preds/oof_preds.csv") 
    df_18_i = pd.read_csv("../../experiments/exp_038/val_preds/oof_preds.csv") 
    df_19_i = pd.read_csv("../../experiments/exp_039/val_preds/oof_preds.csv") 
    train_df["pred"] = np.nan  # 予測値
    train_df.loc[train_df["hour"] == 8, "pred"] = df_8_i["pred"].values.copy()
    train_df.loc[train_df["hour"] == 9, "pred"] = df_9_i["pred"].values.copy()
    train_df.loc[train_df["hour"] == 18, "pred"] = df_18_i["pred"].values.copy()
    train_df.loc[train_df["hour"] == 19, "pred"] = df_19_i["pred"].values.copy()

    train_df.loc[mask_8, "pred"] = df_8_c["pred"].values
    train_df.loc[mask_9, "pred"] = df_9_c["pred"].values
    train_df.loc[mask_18, "pred"] = df_18_c["pred"].values
    train_df.loc[mask_19, "pred"] = df_19_c["pred"].values

    train_df["err1"] = np.abs(train_df["delayTime"] - train_df["pred"])
    train_df["err2"] = np.abs(train_df["delayTime"] - train_df["pred_2"])
    for h in [8, 9, 18, 19]:
        temp_df = train_df[train_df["hour"] == h].copy()
        continued_cv = temp_df[temp_df["isnan_pred"] == 0]["err2"].mean()
        continued_pred_cv = temp_df[temp_df["isnan_pred"] == 0]["err1"].mean()
        pred_cv = temp_df[temp_df["isnan_pred"] == 1]["err1"].mean()
        print(f"hour {h}, continued cv: {continued_cv}, continued pred cv: {continued_pred_cv}, pred cv: {pred_cv}")
        if continued_cv > continued_pred_cv:
            train_df.loc[train_df["hour"] == h, "pred_2"] = train_df[train_df["hour"] == h]["pred"]  # 書き換え
            pred_zone.append(h)
            print(f"time zone {h}'s pred is better than continued")
    train_df["err2"] = np.abs(train_df["delayTime"] - train_df["pred_2"])
    cv = train_df[train_df["hour"].isin([8, 9, 18, 19])]["err2"].mean()
    train_df[train_df["hour"].isin([8, 9, 18, 19])].to_csv("train.csv", index=False)
    print(f"overall cv: {cv}")



    # =========================== make submission ===========================
    for df in [test_df]:
        df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )
    test_df["pred_2"] = test_df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x: x.fillna(method="ffill"))
    test_df["isnanDelayTime"] = test_df["pred_2"].isnull().astype("int")
    print("complete!!!")

    mask_8 = (test_df["hour"] == 8) & (test_df["isnanDelayTime"] == 0) #引き継いだ部分
    mask_9 = (test_df["hour"] == 9) & (test_df["isnanDelayTime"] == 0)
    mask_18 = (test_df["hour"] == 18) & (test_df["isnanDelayTime"] == 0)
    mask_19 = (test_df["hour"] == 19) & (test_df["isnanDelayTime"] == 0)


    df_8_c = pd.read_csv("../../experiments/exp_032/preds/pred.csv") 
    df_9_c = pd.read_csv("../../experiments/exp_033/preds/pred.csv") 
    df_18_c = pd.read_csv("../../experiments/exp_034/preds/pred.csv") 
    df_19_c = pd.read_csv("../../experiments/exp_035/preds/pred.csv") 
    df_8_i = pd.read_csv("../../experiments/exp_036/preds/pred.csv") 
    df_9_i = pd.read_csv("../../experiments/exp_037/preds/pred.csv") 
    df_18_i = pd.read_csv("../../experiments/exp_038/preds/pred.csv") 
    df_19_i = pd.read_csv("../../experiments/exp_039/preds/pred.csv") 

    test_df["pred"] = np.nan  # 予測値
    test_df.loc[test_df["hour"] == 8, "pred"] = df_8_i["pred"].values.copy()
    test_df.loc[test_df["hour"] == 9, "pred"] = df_9_i["pred"].values.copy()
    test_df.loc[test_df["hour"] == 18, "pred"] = df_18_i["pred"].values.copy()
    test_df.loc[test_df["hour"] == 19, "pred"] = df_19_i["pred"].values.copy()



    test_df.loc[test_df["hour"] == 8, "pred_3"] = df_8_c["pred"].values.copy()
    test_df.loc[test_df["hour"] == 9, "pred_3"] = df_9_c["pred"].values.copy()
    test_df.loc[test_df["hour"] == 18, "pred_3"] = df_18_c["pred"].values.copy()
    test_df.loc[test_df["hour"] == 19, "pred_3"] = df_19_c["pred"].values.copy()

    test_df.loc[mask_8, "pred"] = test_df.loc[mask_8, "pred_3"].values.copy()
    test_df.loc[mask_9, "pred"] = test_df.loc[mask_9, "pred_3"].values.copy()
    test_df.loc[mask_18, "pred"] = test_df.loc[mask_18, "pred_3"].values.copy()
    test_df.loc[mask_19, "pred"] = test_df.loc[mask_19, "pred_3"].values.copy()

    for h in pred_zone:
        print(h)
        test_df.loc[(test_df["hour"] == h) & (test_df["isnanDelayTime"] == 0), "pred_2"] = test_df[(test_df["hour"] == h) & (test_df["isnanDelayTime"] == 0)]["pred"].values.copy()
    test_df["pred_2"] = test_df["pred_2"].fillna(test_df["pred"]).values.copy()
    sub_df = test_df[test_df["target"] == 1]
    sub_df[["id", "pred_2"]].to_csv("sub_catboost_1seed.csv", index=False, header=False)
    print("complete!!!")

if __name__ == "__main__":
    main()