from feature_engineering.features import *
import os
import os.path as osp
from sklearn.preprocessing import LabelEncoder

class Preprocessing():
    def __init__(self, param):
        # 先にcv/date/にわけたインデックスを保存しておく
        # 普通に特徴作る
        # train valid 分けて作るものは分けて作る
        self.param = param
        self.cv = param["cv"]
        self.nfolds = param["nfolds"]
        self.seeds = param["seeds"]
        self.feats = param["features"]
        self.drop_feats = param["drop_features"]
        self.label_encode = param["label_encode"]
        self.y = param["y"]
        self.prepro_flag = param["flag"]
        
        self.ROOT = param["ROOT"] # */src
        self.WORK_DIR = param["WORK_DIR"]
        self.outdir = param["output_dir"] # my_features
        self.out_train_path = osp.join(self.ROOT, self.outdir, "train") # */src/my_features/train
        self.out_test_path = osp.join(self.ROOT, self.outdir, "test") # */src/my_features/test

        if "train" not in os.listdir(self.WORK_DIR): os.mkdir(osp.join(self.WORK_DIR, "train")) 
        if "valid" not in os.listdir(self.WORK_DIR): os.mkdir(osp.join(self.WORK_DIR, "valid")) 
        if "test" not in os.listdir(self.WORK_DIR): os.mkdir(osp.join(self.WORK_DIR, "test")) 

    def __call__(self):
        if self.prepro_flag: # まだ作ってないなら
            feat_classes = [eval(feat)(self.param) for feat in self.feats]
            for f_class in feat_classes:
                f_class.run()

            train_df, test_df = self.read_feature()
            
            print("label encode")
            for feat in self.label_encode:
                lbl_enc = LabelEncoder().fit(pd.concat([train_df[feat], test_df[feat]]))
                train_df[feat] = lbl_enc.transform(train_df[feat])
                test_df[feat] = lbl_enc.transform(test_df[feat])

            print("drop cols")
            train_df.drop(columns=self.drop_feats, inplace=True)
            test_df.drop(columns=self.drop_feats, inplace=True)

            print("save data")
            cv_feats = [f"{self.cv}_{seed}" for seed in self.seeds]
            for seed, cv_feat in zip(self.seeds, cv_feats):
                for fold in range(self.nfolds):
                    train = train_df[~(train_df[cv_feat] == fold)]
                    valid = train_df[train_df[cv_feat] == fold]
                    train_X = train.drop(columns=[self.y]+cv_feats)
                    valid_X = valid.drop(columns=[self.y]+cv_feats)
                    train_y = train[[self.y]]
                    valid_y = valid[[self.y]]
                    train_X.to_csv(osp.join(self.WORK_DIR, "train", f"train_X_{seed}_{fold}.csv"), index=False)
                    train_y.to_csv(osp.join(self.WORK_DIR, "train", f"train_y_{seed}_{fold}.csv"), index=False)
                    valid_X.to_csv(osp.join(self.WORK_DIR, "valid", f"valid_X_{seed}_{fold}.csv"), index=False)
                    valid_y.to_csv(osp.join(self.WORK_DIR, "valid", f"valid_y_{seed}_{fold}.csv"), index=False)
            
            test_X = test_df.drop(columns=[self.y])
            test_X.to_csv(osp.join(self.WORK_DIR, "test", f"test_X.csv"), index=False)


    def read_feature(self):
        train_feat_fnames = [osp.join(self.out_train_path, f"{feat}.feather") for feat in self.feats]
        test_feat_fnames = [osp.join(self.out_test_path, f"{feat}.feather") for feat in self.feats if feat != self.cv]
        train_df = pd.concat([pd.read_feather(fname) for fname in train_feat_fnames], axis=1)
        test_df = pd.concat([pd.read_feather(fname) for fname in test_feat_fnames], axis=1)
        return train_df, test_df
        
        
