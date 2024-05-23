import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
#下記は決定木可視化のためのツール
import pydotplus
from IPython.display import Image
from six import StringIO



#データセットのpath指定
data = os.path.join("/workspace","dataset","MalwareData.csv")



#pandasでデータセットをッ読込む
MalwareDataset = pd.read_csv(data,sep='|')
#説明変数にImageBaseとSubsystemを格納する
X = MalwareDataset.loc[:,["ImageBase"]]
#目的変数にlegitimate(マルウェアであれば0、クリーンウェアであれば1)を格納する
y = MalwareDataset["legitimate"]



X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=101
        )



#決定木のモデルを構築する。なお最大の深さは3とする
DecisionTree = DecisionTreeClassifier(max_depth=3)
#学習
DecisionTree.fit(X_train,y_train)



# テスト用のデータを使用して推論
pred = DecisionTree.predict(X_test)



# 予測結果とテスト用のデータを使って正解率と、混同行列を出力
print("Precision: {:.5f}".format(precision_score(y_test, pred)))
print("Recall: {:.5f}".format(recall_score(y_test, pred)))
print("F1 Score: {:.5f}".format(f1_score(y_test, pred)))



#決定木の可視化
dot_data = StringIO() #dotファイル情報の格納先
export_graphviz(DecisionTree, out_file=dot_data,  
                     feature_names=["ImageBase"],
                     class_names=["False","True"],
                     filled=True, rounded=True,  
                     special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png("decision_tree.png")
