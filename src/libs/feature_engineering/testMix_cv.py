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