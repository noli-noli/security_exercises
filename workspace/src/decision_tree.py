import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier,export_graphviz
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



#決定木のモデルを構築する。なお最大の深さは3とする
DecisionTree = DecisionTreeClassifier(max_depth=3)
#学習
DecisionTree.fit(X,y)



#決定木の可視化
dot_data = StringIO() #dotファイル情報の格納先
export_graphviz(DecisionTree, out_file=dot_data,  
                     feature_names=["ImageBase"],
                     class_names=["False","True"],
                     filled=True, rounded=True,  
                     special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png("decision_tree.png")
