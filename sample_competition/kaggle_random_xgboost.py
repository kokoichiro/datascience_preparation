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

from datetime import datetime 
from datetime import date

import time
import os
import sys

from scipy import misc
from scipy.stats import randint

import yaml

from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA

from xgboost import XGBClassifier


import argparse

class mlxgboost():
	def __init__(self):
		self.training_file=None
		self.test_file=None
		self.submission_file=None
		self.explore_test_rate=0
		self.target_name=None
		self.target_metric=None
		self.training_rate=0
		self.n_jobs=0
		self.record_pkl=None
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
		record_pkl=None,
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
		self.record_pkl=cfg['file']['record_pkl']
		self.id=cfg['ML_parameter']['id']

	def load_pickle(self,pickle_file):
		try:
			with open(pickle_file, 'rb') as f:
				pickle_data = pickle.load(f)
		except UnicodeDecodeError as e:
			with open(pickle_file, 'rb') as f:
				pickle_data = pickle.load(f, encoding='latin1')
		except Exception as e:
			print('Unable to load data ', pickle_file, ':', e)
			raise

		return pickle_data

	def timer(self,start_time=None):
		if not start_time:
			start_time = datetime.now()
			return start_time
		elif start_time:
			thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
			tmin, tsec = divmod(temp_sec, 60)
			print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))



if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog='kaggle_random_cv.py',usage='Use this code for exploring the best methodology and parameter for ML.',description='this code needs 2 different args')
	parser.add_argument('configuration', help = 'The location of Configuration file')
	parser.add_argument('--mode',required=False,help = 'You can select one variable from 3 parameters you can skip this parameter.',default='all',choices=['all','random_search','simple_ml'])

	args=parser.parse_args()
	now = datetime.today()
	today_str= now.strftime("%Y%m%d")


	from kaggle_random_xgboost import mlxgboost
	xgbexp=mlxgboost()
	xgbexp.set_parameters(args.configuration)
	df_train=pd.read_csv(xgbexp.training_file)
	train_features1=df_train[xgbexp.target_name]
	train_labels1=df_train.drop(xgbexp.id,axis=1)
	train_labels1=train_labels1.drop(xgbexp.target_name,axis=1)
	params = {
        'learning_rate': [0.01,0.02,0.3,0.5],'min_child_weight': [1, 5, 10],'gamma': [0.5, 1, 1.5, 2, 5],'subsample': [0.6, 0.8, 1.0],'colsample_bytree': [0.6, 0.8, 1.0],'max_depth': [3, 4, 5,10]
        }
	xgb = XGBClassifier( objective='binary:logistic')
	folds = 8
	param_comb = 5

	X=train_labels1
	Y=train_features1
	skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
	random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='f1', n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001 )
	start_time = xgbexp.timer(None) # timing starts from this point for "start_time" variable
	random_search.fit(X, Y)
	xgbexp.timer(start_time) # timing ends here for "start_time" variable

	print('\n All results:')
	print(random_search.cv_results_)
	print('\n Best estimator:')
	print(random_search.best_estimator_)
	print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
	print(random_search.best_score_ * 2 - 1)
	print('\n Best hyperparameters:')
	print(random_search.best_params_)

	model_name=today_str+'.model'
	random_search.best_estimator_.save_model(model_name)

	df_test=pd.read_csv('./santandar/test.csv')
	df_label_test=df_test.drop('ID_code',axis=1)
	y_test = random_search.predict(df_label_test)
	y_pred=pd.DataFrame(y_test)
	df_answer=pd.concat([df_test["ID_code"],y_pred],axis=1)
	output_name='./santandar/submission_xgbt'+today_str+'.csv'
	df_answer.to_csv(output_name,index=False)
