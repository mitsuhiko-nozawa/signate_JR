from cross_validation import create_cv
class Preprocessing():
    def __init__(self, param):
        # 先にcv/date/にわけたインデックスを保存しておく
        # 普通に特徴作る
        # train valid 分けて作るものは分けて作る
        self.param = param
        self.cv = param["cv"]
        self.feats = param["features"]
        self.feats_in_cv = param["features_in_cv"]
        self.drop_feats = param["drop_features"]
        
        self.ROOT = param["ROOT"]
        self.WORK_DIR = param["WORK_DIR"]
        self.feat_outdir = param["feature_outdir"]

    def __call__(self):
        #CV = create_cv(self.param)
        #CV()
        pass