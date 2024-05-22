import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier,export_graphviz
#下記は決定木可視化のためのツール
import pydotplus
from IPython.display import Image
from six import StringIO


data = os.path.join("/workspace","src","Decisiontree","MalwareData.csv")


MalwareDataset = pd.read_csv(data,sep='|')
#MalwareDatasetからname,md5,legitimateを除外してXに格納する
X = MalwareDataset.loc[:,["ImageBase","Subsystem"]]
#legitimateをyに格納する
y = MalwareDataset["legitimate"]


#決定木のモデルを構築する。
DecisionTree = DecisionTreeClassifier()
DecisionTree.fit(X,y)


dot_data = StringIO() #dotファイル情報の格納先
export_graphviz(DecisionTree, out_file=dot_data,  
                     feature_names=["ImageBase", "Subsystem"],
                     class_names=["False","True"],
                     filled=True, rounded=True,  
                     special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png("decision_tree.png")
