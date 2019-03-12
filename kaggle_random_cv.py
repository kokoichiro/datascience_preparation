
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from itertools import product
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor,BaggingClassifier,VotingClassifier,AdaBoostClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.externals.six import StringIO
#from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
#from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,roc_auc_score,
                             f1_score,accuracy_score,confusion_matrix,mean_absolute_error,r2_score,mean_squared_error)
import datetime

import time
import os
import sys

from scipy import misc
from scipy.stats import randint

import yaml
import time
import os
import sys

import argparse
from random import randint

class kaggleexploration():

	def __init__(self):
		self.training_file=None
		self.test_file=None
		self.submission_file=None
		self.explore_test_rate=0
		self.target_name=None
		self.target_metric=None
		self.training_rate=0
		self.n_jobs=0
		self.record_txt=None
		self.id=None

	def set_parameters(self,
		config_loc,
		training_file=None,
		test_file=None,
		submission_file=None,
		explore_test_rate=0,
		target_name=None,
		target_metric=None,
		training_rate=0,
		n_jobs=0,
		record_txt=None,
		id=None):
		with open(config_loc,'r') as ymlfile:
			cfg = yaml.load(ymlfile)

		self.training_file=cfg['file']['training_file']
		self.test_file=cfg['file']['test_file']
		self.submission_file=cfg['file']['submission_file']
		self.explore_test_rate=cfg['ML_parameter']['explore_test_rate']
		self.target_name=cfg['ML_parameter']['target_name']
		self.target_metric=cfg['ML_parameter']['target_metric']
		self.training_rate=cfg['ML_parameter']['training_rate']
		self.n_jobs=cfg['ML_parameter']['n_jobs']
		self.record_txt=cfg['file']['record_txt']
		self.id=cfg['ML_parameter']['id']


	def get_score(clf, train_features, train_labels):
		X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.5, random_state=0)
		clf.fit(X_train, y_train)
		print (clf.score(X_test, y_test))

	def get_accuracy(clf, train_features, train_labels):
		scores = cross_validation.cross_val_score(clf, train_features, train_labels, cv=10)
		print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


	def random_searchSVM(self,train_features, train_labels, n_jobs,objective):
		param_dist = {
	#   {'C': [1, 10, 100], 'kernel': ['linear']},
						#'C': [1,10,50,100], 'gamma': [0.0001, 0.001], 'kernel': ['rbf','linear']}
	  	'C': [1], 'gamma': [0.001], 'kernel': ['rbf']}
	

		clf = GridSearchCV(svm.SVC(class_weight="balanced"), param_dist,n_jobs=n_jobs,scoring=objective)
		clf.fit(train_features, train_labels)
		return clf.best_estimator_,clf.best_score_,clf.best_params_

	def random_searchGB(self,train_features, train_labels, n_jobs,objective):
		param_dist = {'n_estimators': randint(5, 300),
						'min_samples_leaf': randint(5, 30),
						'max_depth': randint(5, 10),
						'learning_rate': [0.05,0.001]}

		clf = RandomizedSearchCV(GradientBoostingClassifier(),param_dist,n_jobs=n_jobs,scoring=objective)
		clf.fit(train_features, train_labels)
		return clf.best_estimator_,clf.best_score_,clf.best_params_

	def random_searchRF(self,train_features, train_labels, n_jobs,objective):
		param_dist = {
			'n_estimators'  : randint(5, 300),
			'max_features'  : randint(3, 20),
			'min_samples_split' : randint(3, 100),
			'max_depth' : randint(4, 30)}
		clf = RandomizedSearchCV(RandomForestClassifier(class_weight="balanced"), param_dist,n_jobs=n_jobs,scoring=objective)
		clf.fit(train_features, train_labels)
		return clf.best_estimator_,clf.best_score_,clf.best_params_

	def random_searchVC(self,train_features,train_labels,n_jobs,objective):
		clf1 = LogisticRegression(random_state=1,class_weight="balanced")
		clf2 = RandomForestClassifier(random_state=1,class_weight="balanced")
		clf3 = GaussianNB()
		eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
		params = {'lr__C': randint(1, 300), 'rf__n_estimators': randint(5, 300),'rf__max_features':randint(3, 20),'rf__min_samples_split' : randint(3, 100),'rf__max_depth':randint(3, 100)}
		clf = RandomizedSearchCV(estimator=eclf,param_distributions=params,n_jobs=n_jobs,scoring=objective)
		clf.fit(train_features, train_labels)
		return clf.best_estimator_,clf.best_score_,clf.best_params_

	def random_searchAB(self,train_features,train_labels,n_jobs,objective):
		param_dist = {
		'algorithm'  : ['SAMME', 'SAMME.R'],
		'n_estimators'  : randint(5, 300)}
		bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))
		clf = RandomizedSearchCV(estimator=bdt,param_distributions=param_dist,n_jobs=n_jobs,scoring=objective)
		clf.fit(train_features, train_labels)
		return clf.best_estimator_,clf.best_score_,clf.best_params_


	def random_searchNC(self,train_features,train_labels,n_jobs,objective):
		param_dist = {
		'activation'  : ['identity', 'logistic', 'tanh', 'relu'],
		'hidden_layer_sizes' : [(5,2), (10,5),(10,20,5),(5,10,10,5)],
		'alpha' : [1e-5,1e-4,1e-3],
		'solver' : ['lbfgs', 'sgd', 'adam']}
  
		clf = GridSearchCV(estimator= MLPClassifier(),param_grid=param_dist,n_jobs=n_jobs,scoring=objective)
		clf.fit(train_features, train_labels)
		return clf.best_estimator_,clf.best_score_,clf.best_params_


	def best_estimater(self,train_features,train_labels,test_features,test_labels,objective):
		start_time = time.time()
		n_jobs = 30

		#objective = 'f1' or 'accuracy' or 'recall' or 'roc_auc'
		a_RF,b_RF,c_RF=self.random_searchRF(train_features, train_labels,n_jobs,objective)
		  
		print("---RF finished %s seconds ---" % round(time.time() - start_time,2))
		duration_time = time.time()
		  
		a_GB,b_GB,c_GB=self.random_searchGB(train_features, train_labels,n_jobs,objective)
		  
		print("---GB finished %s seconds ---" % round(time.time() - duration_time,2))
		duration_time = time.time()
		  
		a_SVM,b_SVM,c_SVM=self.random_searchSVM(train_features, train_labels,n_jobs,objective)
		  
		print("---SVM finished %s seconds ---" % round(time.time() - duration_time,2))
		duration_time = time.time()
		  
		a_VC,b_VC,c_VC=self.random_searchVC(train_features, train_labels,n_jobs,objective)
		  
		print("---VotingClassification finished %s seconds ---" % round(time.time() - start_time,2))
		duration_time = time.time()
		  
		a_AB,b_AB,c_AB=self.random_searchAB(train_features, train_labels,n_jobs,objective)
		  
		print("---AdaBoost finished %s seconds ---" % round(time.time() - duration_time,2))
		duration_time = time.time()
		  
		a_NC,b_NC,c_NC=self.random_searchNC(train_features, train_labels,n_jobs,objective)
		  
		print("---NeuralNetwork finished %s seconds ---" % round(time.time() - duration_time,2))
		duration_time = time.time()
		  
		  
		  
		RF_ML = a_RF
		RF_ML.fit(train_features,train_labels)
		RF_ML.score(test_features,test_labels)
		RF_y_pred = RF_ML.predict(test_features)
		  
		  
		print("---RF fitting finished %s seconds ---" % round(time.time() - duration_time,2))
		duration_time = time.time()
		  
		GB_ML = a_GB
		GB_ML.fit(train_features,train_labels)
		GB_ML.score(test_features,test_labels)
		GB_y_pred = GB_ML.predict(test_features)
		  
		  
		print("---GB fitting finished %s seconds ---" % round(time.time() - duration_time,2))
		duration_time = time.time()
		  
		SVM_ML = a_SVM
		SVM_ML.fit(train_features,train_labels)
		SVM_ML.score(test_features,test_labels)
		SVM_y_pred = SVM_ML.predict(test_features)
		  
		  
		print("---SVM fitting finished %s seconds ---" % round(time.time() - duration_time,2))
		duration_time = time.time()
		  
		  
		VC_ML = a_VC
		VC_ML.fit(train_features,train_labels)
		VC_ML.score(test_features,test_labels)
		VC_y_pred = VC_ML.predict(test_features)
		  
		  
		print("---VotingClassification fitting finished %s seconds ---" % round(time.time() - duration_time,2))
		duration_time = time.time()
		  
		AB_ML = a_AB
		AB_ML.fit(train_features,train_labels)
		AB_ML.score(test_features,test_labels)
		AB_y_pred = AB_ML.predict(test_features)
		  
		  
		print("---AdaBoosting fitting finished %s seconds ---" % round(time.time() - duration_time,2))
		duration_time = time.time()
		  
		NC_ML = a_NC
		NC_ML.fit(train_features,train_labels)
		NC_ML.score(test_features,test_labels)
		NC_y_pred = NC_ML.predict(test_features)
		  
		  
		print("---Neural Network fitting finished %s seconds ---" % round(time.time() - duration_time,2))
		duration_time = time.time()
		  
		if objective == "f1":
			RF_score=f1_score(test_labels, RF_y_pred)
			GB_score=f1_score(test_labels, GB_y_pred)
			SVM_score=f1_score(test_labels, SVM_y_pred)
			VC_score=f1_score(test_labels, VC_y_pred)
			NC_score=f1_score(test_labels, NC_y_pred)
			AB_score=f1_score(test_labels, AB_y_pred)
		elif objective == "recall":
			RF_score=recall_score(test_labels, RF_y_pred)
			GB_score=recall_score(test_labels, GB_y_pred)
			SVM_score=recall_score(test_labels, SVM_y_pred)
			VC_score=recall_score(test_labels, VC_y_pred)
			NC_score=recall_score(test_labels, NC_y_pred)
			AB_score=recall_score(test_labels, AB_y_pred)
		elif objective == "accuracy":
			RF_score=accuracy_score(test_labels, RF_y_pred)
			GB_score=accuracy_score(test_labels, GB_y_pred)
			SVM_score=accuracy_score(test_labels, SVM_y_pred)
			VC_score=accuracy_score(test_labels, VC_y_pred)
			NC_score=accuracy_score(test_labels, NC_y_pred)
			AB_score=accuracy_score(test_labels, AB_y_pred)
		elif objective == "roc_auc":
			RF_score=roc_auc_score(test_labels, RF_y_pred)
			GB_score=roc_auc_score(test_labels, GB_y_pred)
			SVM_score=roc_auc_score(test_labels, SVM_y_pred)
			VC_score=roc_auc_score(test_labels, VC_y_pred)
			NC_score=roc_auc_score(test_labels, NC_y_pred)
			AB_score=roc_auc_score(test_labels, AB_y_pred)
		print ('RF:'+str(RF_score)+', GB:'+str(GB_score)+', SVM:'+str(SVM_score)+', VC:'+str(VC_score)+', AB:'+str(AB_score)+', NC:'+str(NC_score))
		score_list = [RF_score,GB_score,SVM_score,VC_score,NC_score,AB_score]
		if RF_score ==max(score_list):
			return a_RF,b_RF,c_RF
		elif GB_score ==max(score_list):
			return a_GB,b_GB,c_GB
		elif SVM_score ==max(score_list):
			return a_SVM,b_SVM,c_SVM
		elif VC_score ==max(score_list):
			return a_VC,b_VC,c_VC
		elif NC_score ==max(score_list):
			return a_NC,b_NC,c_NC
		elif AB_score ==max(score_list):
			return a_AB,b_AB,c_AB
		else:
			print('The error happens')







if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog='kaggle_random_cv.py',usage='Use this code for exploring the best methodology and parameter for ML.',description='this code needs 2 different args')
	parser.add_argument('configuration', help = 'The location of Configuration file')
	parser.add_argument('--mode',required=False,help = 'You can select one variable from 3 parameters you can skip this parameter.',default='all',choices=['all','random_search','simple_ml'])

	args=parser.parse_args()

	from kaggle_random_cv import kaggleexploration
	exp=kaggleexploration()
	exp.set_parameters(args.configuration)


	df_train=pd.read_csv(exp.training_file)
	train_features1=df_train[exp.target_name]
	train_labels1=df_train.drop(exp.id,axis=1)
	train_labels1=train_labels1.drop(exp.target_name,axis=1)
	if args.mode == 'all' or 'random_search':
		y_train, y_test, X_train, X_test = train_test_split(train_features1, train_labels1,test_size=exp.explore_test_rate, random_state=0)
		a,b,c=exp.best_estimater(X_train, y_train,X_test,y_test,exp.target_metric)
		a_txt = str(a)

		with  open(exp.record_txt,mode='w') as f:
			f.write(a_text)
			f.write(b)
			f.write(c)
	if args.mode == 'all' or 'simple_ml':

		if args.mode == 'all':
			ML=a
		elif args.mode == 'simple_ml':
			ML=VotingClassifier(estimators=[('lr', LogisticRegression(C=199, class_weight='balanced', dual=False,
			fit_intercept=True, intercept_scaling=1, max_iter=100,
			multi_class='ovr', n_jobs=1, penalty='l2', random_state=1,
			solver='liblinear', tol=0.0001, verbose=0, warm_start=False)), ('rf', Rand...=False, random_state=1,
			verbose=0, warm_start=False)), ('gnb', GaussianNB(priors=None))],
			flatten_transform=None, n_jobs=1, voting='soft', weights=None)
		ML.fit(train_labels1,train_features1)
		#ML.fit(X_train, y_train)
		df_test=pd.read_csv(exp.test_file)
		df_label_test=df_test.drop(exp.id,axis=1)
		y_pred=ML.predict(df_label_test)
		y_pred = ML.predict(X_test)
		df_answer=pd.concat([df_test[exp.id],y_pred],axis=1)
		df_answer.to_csv(exp.submission_file,index=False)
