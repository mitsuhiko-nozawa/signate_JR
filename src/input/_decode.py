import numpy as np
import pandas as pd

def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    # 路線のデコード
    line_di = {"A" : "中央・総武各停", "B" : "京浜東北線", "C" : "山手線", "D" : "中央線快速"}
    train["lineName"] = train["lineName"].map(line_di)
    test["lineName"] = test["lineName"].map(line_di)

    # 駅名のデコード
    A_stationCode = ["trdb","daXW","Rlfq","coZB","LMww","VNyR","jhlV","efzW","PcxI","ejfb","RDLf","cRgf","SvNu","GLXs","iHon","OJng","HOLt","qzrJ","TlTg","NqOG","KsBm","QCTc","fpRv","zUdG","APgF","RUhy","tPfo","jPbe","stJE","ufGe","tncu","wbkB","aCxM","AFTQ","CTnl","mxQg","PsTo","jebQ","dJlm","Femc","vpGT"]
    B_stationCode = ["qzWm","UUlH","lUUx","rGbD","OpgJ","FPFv","mjPK","uQdT","zNdV","KReX","aVeu","oNkU","vcLy","cvCM","EoNR","qdzg","lCym","DTSj","hwjA","PYtN","KpNq","ZcpD","QkqQ","TaOj","yuIQ","TlTg","KsBm","ReJL","yUYG","CbCQ","aMqH","bgbz","LKUe","LglK","cbJh","daBI","MQfL","jdBM","xKSN","qCwi","sLAH","ugLy","GsES","ntaA","Zlsb"]
    C_stationCode = ["yuIQ","TlTg","KsBm","ReJL","yUYG","CbCQ","aMqH","bgbz","LKUe","fmwC","NLJh","IXNJ","mKyE","tNrH","daSA","Gftx","RDLf","cRgf","djPS","fRXM","vzmD","fXMY","thmK","SAOS","hwjA","PYtN","KpNq","ZcpD","QkqQ","TaOj"]
    D_stationCode = ["vpGT","Femc","dJlm","jebQ","PsTo","mxQg","CTnl","AFTQ","aCxM","wbkB","tncu","ufGe","stJE","jPbe","tPfo","RUhy","APgF","zUdG","fpRv","QCTc","KsBm","TlTg","NqOG","qzrJ","HOLt","OJng","iHon","GLXs","SvNu","cRgf","RDLf","ejfb","PcxI","efzW","jhlV","VNyR","LMww","coZB","Rlfq","daXW","trdb","tCey","GgOD","Hzeg","fZfY","UMoa","GxuL","BCRD","AVjc","uYlv","wwYD","mkGW"]
    A_stationName = ["JBxx 武蔵境","JB01 三鷹","JB02 吉祥寺","JB03 西荻窪","JB04 荻窪","JB05 阿佐ヶ谷","JB06 高円寺","JB07 中野","JB08 東中野","JB09 大久保","JB10 新宿","JB11 代々木","JB12 千駄ヶ谷","JB13 信濃町","JB14 四ツ谷","JB15 市ヶ谷","JB16 飯田橋","JB17 水道橋","JBxx 神田","JB18 御茶ノ水","JB19 秋葉原","JB20 浅草橋","JB21 両国","JB22 錦糸町","JB23 亀戸","JB24 平井","JB25 新小岩","JB26 小岩","JB27 市川","JB28 本八幡","JB29 下総中山","JB30 西船橋","JB31 船橋","JB32 東船橋","JB33 津田沼","JB34 幕張本郷","JB35 幕張","JB36 新検見川","JB37 稲毛","JB38 西千葉","JB39 千葉"]
    B_stationName = ["JK02 本郷台","JK03 港南台","JK04 洋光台","JK05 新杉田","JK06 磯子","JK07 根岸","JK08 山手","JK09 石川町","JK10 関内","JK11 桜木町","JK12 横浜","JK13 東神奈川","JK14 新子安","JK15 鶴見","JK16 川崎","JK17 蒲田","JK18 大森","JK19 大井町","JK20 品川","JK21 高輪ゲートウェイ","JK22 田町","JK23 浜松町","JK24 新橋","JK25 有楽町","JK26 東京","JK27 神田","JK28 秋葉原","JK29 御徒町","JK30 上野","JK31 鶯谷","JK32 日暮里","JK33 西日暮里","JK34 田端","JK35 上中里","JK36 王子","JK37 東十条","JK38 赤羽","JK39 川口","JK40 西川口","JK41 蕨","JK42 南浦和","JK43 浦和","JK44 北浦和","JK45 与野","JK46 さいたま新都心"]
    C_stationName = ["JY01 東京","JY02 神田","JY03 秋葉原","JY04 御徒町","JY05 上野","JY06 鶯谷","JY07 日暮里","JY08 西日暮里","JY09 田端","JY10 駒込","JY11 巣鴨","JY12 大塚","JY13 池袋","JY14 目白","JY15 高田馬場","JY16 新大久保","JY17 新宿","JY18 代々木","JY19 原宿","JY20 渋谷","JY21 恵比寿","JY22 目黒","JY23 五反田","JY24 大崎","JY25 品川","JY26 高輪ゲートウェイ","JY27 田町","JY28 浜松町","JY29 新橋","JY30 有楽町"]
    D_stationName = ["JCxx 千葉","JCxx 西千葉","JCxx 稲毛","JCxx 新検見川","JCxx 幕張","JCxx 幕張本郷","JCxx 津田沼","JCxx 東船橋","JCxx 船橋","JCxx 西船橋","JCxx 下総中山","JCxx 本八幡","JCxx 市川","JCxx 小岩","JCxx 新小岩","JCxx 平井","JCxx 亀戸","JCxx 錦糸町","JCxx 両国","JCxx 浅草橋","JCxx 秋葉原","JC02 神田","JC03 御茶ノ水","JCxx 水道橋","JCxx 飯田橋","JCxx 市ヶ谷","JC04 四ツ谷","JCxx 信濃町","JCxx 千駄ヶ谷","JCxx 代々木","JC05 新宿","JCxx 大久保","JCxx 東中野","JC06 中野","JC07 高円寺","JC08 阿佐ヶ谷","JC09 荻窪","JC10 西荻窪","JC11 吉祥寺","JC12 三鷹","JC13 武蔵境","JC14 東小金井","JC15 武蔵小金井","JC16 国分寺","JC17 西国分寺","JC18 国立","JCxx 新小平","JC19 立川","JC20 日野","JC21 豊田","JC22 八王子","JC23 西八王子"]
    
    # テスト
    if len(A_stationCode) != len(A_stationName):
        print("A")
    if len(B_stationCode) != len(B_stationName):
        print("B")
    if len(C_stationCode) != len(C_stationName):
        print("C")
    if len(D_stationCode) != len(D_stationName):
        print("D")

    A_stationDict = {}
    B_stationDict = {}
    C_stationDict = {}
    D_stationDict = {}
    stationDict = {}
    for line in ["A", "B", "C", "D"]:
        for code, name in zip(eval(line+"_stationCode"), eval(line+"_stationName")):
            eval(line+"_stationDict")[code] = name
        stationDict.update(eval(line+"_stationDict"))
    
    train["stopStation"] = train["stopStation"].map(stationDict)
    test["stopStation"] = test["stopStation"].map(stationDict)

    train[:100000].to_csv("decoded_train_100000.csv", index=False, encoding = "cp932")
    test[:100000].to_csv("decoded_test_100000.csv", index=False, encoding = "cp932")




if __name__ == "__main__":
    main()