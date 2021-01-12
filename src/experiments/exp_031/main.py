import numpy as np
import pandas as pd
import sys, os
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    train_df = pd.read_csv("../../input/train.csv")
    test_df = pd.read_csv("../../input/test.csv")
    pred_zone = []

    for df in [train_df, test_df]:
        df["hour"] = df["planArrival"].map(lambda x : int(x[:2].replace(":", "")) if int(x[3:]) != 0 else int(x[:2].replace(":", ""))-1  )


    # =========================== calculate cv ===========================
    df_8 = pd.read_csv("../../experiments/exp_026/val_preds/oof_preds.csv") 
    df_9 = pd.read_csv("../../experiments/exp_027/val_preds/oof_preds.csv") 
    df_18 = pd.read_csv("../../experiments/exp_028/val_preds/oof_preds.csv") 
    df_19 = pd.read_csv("../../experiments/exp_029/val_preds/oof_preds.csv") 

    train_df["pred"] = np.nan  # 予測値
    train_df.loc[train_df["hour"] == 8, "pred"] = df_8["pred"].values
    train_df.loc[train_df["hour"] == 9, "pred"] = df_9["pred"].values
    train_df.loc[train_df["hour"] == 18, "pred"] = df_18["pred"].values
    train_df.loc[train_df["hour"] == 19, "pred"] = df_19["pred"].values

    train_df["pred_2"] = train_df["delayTime"].values.copy()  # 引き継ぎ+予測値
    for h in [19, 18, 9, 8]:
        train_df.loc[train_df["hour"] == h, "pred_2"] = np.nan
        train_df["pred_2"] = train_df.groupby(["date", "trainNo"])["pred_2"].transform(lambda x: x.fillna(method="ffill"))
    train_df["isnan_pred"] = train_df["pred_2"].isnull().astype("int")
    train_df["pred_2"] = train_df["pred_2"].fillna(train_df["pred"])
    train_df["err1"] = np.abs(train_df["delayTime"] - train_df["pred"])
    train_df["err2"] = np.abs(train_df["delayTime"] - train_df["pred_2"])
    for h in [8, 9, 18, 19]:
        temp_df = train_df[train_df["hour"] == h].copy()
        continued_cv = temp_df[temp_df["isnan_pred"] == 0]["err2"].mean()
        continued_pred_cv = temp_df[temp_df["isnan_pred"] == 0]["err1"].mean()
        pred_cv = temp_df[temp_df["isnan_pred"] == 1]["err1"].mean()
        print(f"hour {h}, continued cv: {continued_cv}, continued pred cv: {continued_pred_cv}, pred cv: {pred_cv}")
        if continued_cv > continued_pred_cv:
            train_df.loc[train_df["hour"] == h, "pred2"] = train_df[train_df["hour"] == h]["pred"]  # 書き換え
            pred_zone.append(h)
            print(f"time zone {h}'s pred is better than continued")
    train_df["err2"] = np.abs(train_df["delayTime"] - train_df["pred_2"])
    cv = train_df[train_df["hour"].isin([8, 9, 18, 19])]["err2"].mean()
    print(f"overall cv: {cv}")



    # =========================== make submission ===========================
    df_8 = pd.read_csv("../../experiments/exp_026/preds/pred.csv") 
    df_9 = pd.read_csv("../../experiments/exp_027/preds/pred.csv") 
    df_18 = pd.read_csv("../../experiments/exp_028/preds/pred.csv") 
    df_19 = pd.read_csv("../../experiments/exp_029/preds/pred.csv") 

    test_df["pred"] = np.nan
    test_df.loc[test_df["hour"] == 8, "pred"] = df_8["pred"].values
    test_df.loc[test_df["hour"] == 9, "pred"] = df_9["pred"].values
    test_df.loc[test_df["hour"] == 18, "pred"] = df_18["pred"].values
    test_df.loc[test_df["hour"] == 19, "pred"] = df_19["pred"].values

    test_df["pred_2"] = test_df.groupby(["date", "trainNo"])["delayTime"].transform(lambda x: x.fillna(method="ffill"))
    for h in pred_zone:
        test_df.loc[test_df["hour"] == h, "pred_2"] = test_df[test_df["hour"] == h]["pred"].values
    test_df["pred_2"] = test_df["pred_2"].fillna(test_df["pred"])
    sub_df = test_df[test_df["target"] == 1]
    sub_df[["id", "pred_2"]].to_csv("sub_catboost.csv", index=False, header=False)

if __name__ == "__main__":
    main()