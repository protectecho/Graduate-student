# -*- coding: utf-8 -*-
import pydotplus
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO
from decisiontree_extracting import tree_to_code
from sklearn.datasets import load_iris

class DecisionTree(object):
    '''
    data:输入数据
    feature_names:特征名字
    y:类型标签
    treeNumber:决策树个数
    class_names:类型标签名字
    '''
    def __init__(self,data,feature_names,y,treeNumber,class_names):
        self.data=data
        self.feature_names=feature_names
        self.y=y
        self.treeNumber=treeNumber
        self.class_names=class_names

    def RandomForest(self):
        print 'Generating Random Forest Classifier...'
        clf = RandomForestClassifier(n_estimators=self.treeNumber)
        clf = clf.fit(self.data, self.y)
        for i in xrange(len(clf.estimators_)):
            dot_data = StringIO()
            result_filename = 'classifying_result//classifying-{}.txt'.format(i)
            tree_to_code(clf.estimators_[i], feature_names=self.feature_names, class_names=self.class_names,
                         result_filename=result_filename)
            # txt_value=tree_to_code_db_all(clf.estimators_[i], feature_names = feature_names, class_names = class_names)
            # tree_to_code_db(clf.estimators_[i], type, feature_names=feature_names, class_names=class_names,
            #                 package_name=package_name)
            print 'Generating %d plot...' % i
            tree.export_graphviz(clf.estimators_[i], out_file=dot_data, feature_names=self.feature_names,
                                 class_names=self.class_names, filled=True, rounded=True, special_characters=True,
                                 leaves_parallel=True)
            print 'Convering %d plot...' % i
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
            graph.write_png('classifying_result//classifying-%d.png' % i)

if __name__=='__main__':
    print 'ds'
    iris=load_iris()
    X=iris.data
    y=iris.target
    feature=['kk','kl','fg','k']
    class_names=['good','poor','usual']
    forest1=DecisionTree(data=iris.data,feature_names=['kk','kl','fg','k'],y=iris.target,treeNumber=1,class_names=['good','poor','usual'])
    forest1.RandomForest()


