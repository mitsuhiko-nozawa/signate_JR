# signate_JR
実験を回す用のリポジトリです。  
異なるリソースで実行した実験の結果を統合できるように、GCS(Google Cloud Storage)にアップロードしています。  
数日のうちに実験結果をローカルにも引っ張ってこれるようなツールを作れたらと思います。  

## コード概要
src配下に入っています。使うコードは基本的に全てlibsに置いています。作成されたデータについては、特徴量はmy_features/配下、そのほかtrain_X, valid_Xなど実験ごとのデータは全てexperiments/exp_xxx/配下に作成されます。  
exp_xxxディレクトリに入ってrun.shを実行すると実験が回ります。その中でmain.pyを実行し、yamlファイルが読み込まれます。main.pyのなかのrunnerモジュールで、preprocessing、learning、predictig、loggingモジュールが実行されます。  
preprocessingでは特徴量の作成、trainとvalidの作成が行われます。  
learningでseed*foldごとにモデルを作り学習させます。  
predictingは予測をします。learningにくっつけたいですがコンペでそうしろと言われているので作っています。  
loggingは実験結果として残したいものを作ります。cv、feature_importancesなどをここで計算します。  
特徴量は１特徴量グループごとにclassを書いています。全て基底クラスFeatureを継承させるようにしていて、create_feature関数を実装させます。  
cvも特徴量と同じような管理の方法をさせていて、例えば5foldなら−1~4までの数が値として入るように作ります。(-1はどのフォールドにも属さない)  
モデルはLGBMモデルしか作っていませんが、基底クラスbase_modelを継承させて作ります。base_modelに実装されている5つの関数を統一インターフェイスとしています。
  
## アイデア・方針
ところどころにmemo.txtやdescription.txtが散見されますが、その時思った思考の断片を書いているのであてにしないでくださいm(__)m  
必要だと思う場合はrootのnote.pptxにも書いていきます。

### cv方針
testもtrainに混ぜて、date × am_pmでGroupKFoldし、予測対象区間だけvalidに回す。
ただしGroupKFold標準ではrandom_stateを引数に持たないので、groupbyしてKFoldを複数シードしたい。切り方を増やすことで汎化性能のが上がると嬉しい。






