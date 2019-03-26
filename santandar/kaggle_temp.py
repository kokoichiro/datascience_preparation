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
from sklearn.externals import joblib
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
from catboost import CatBoostClassifier
from catboost import Pool
from lightgbm import LGBMClassifier


import argparse
import optuna

class multipleboosts():
	def __init__(self):
		self.home_folder=None
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
		self.nfolds=0
		self.X=None
		self.Y=None

	def set_parameters(self,
		config_loc,
		training_file=None,
		test_file=None,
		submission_file=None,
		home_folder=None,
		explore_test_rate=0,
		target_name=None,
		target_metric=None,
		training_rate=0,
		n_jobs=0,
		record_pkl=None,
		model_location=None,

		id=None,
		nfolds=0):
		with open(config_loc,'r') as ymlfile:
			cfg = yaml.load(ymlfile)
		now = datetime.today()
		today_str= now.strftime("%Y%m%d")

		self.home_folder=cfg['file']['home_folder']
		self.explore_test_rate=cfg['ML_parameter']['explore_test_rate']
		self.target_name=cfg['ML_parameter']['target_name']
		self.target_metric=cfg['ML_parameter']['target_metric']
		self.training_rate=cfg['ML_parameter']['training_rate']
		self.n_jobs=cfg['ML_parameter']['n_jobs']
		self.record_pkl=cfg['file']['record_pkl']
		record_model=str(self.record_pkl).split(".")
		self.model_location=self.home_folder+'model/'+record_model[0]+'_'+today_str+'.'+record_model[1]
		self.submission_file=self.home_folder+'submission/'+record_model[0]+'_'+today_str+'.csv'
		self.training_file=self.home_folder+'data/train.csv'
		self.test_file=self.home_folder+'data/test.csv'
		self.id=cfg['ML_parameter']['id']
		self.nfolds=cfg['ML_parameter']['nfolds']



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

	def catboost_randomized(self,skf,X,Y,param_comb):
		cat_params={
			#'max_depth': [2,5],
			#'learning_rate': [0.02,0.1],
			#'colsample_bylevel': [0.05, 0.3]
			'max_depth': [2,5],
			'learning_rate': [0.02,0.05],
			'colsample_bylevel': [0.05,0.3]
			}		
		cat = CatBoostClassifier(objective="Logloss")
		random_search_cat = RandomizedSearchCV(cat, param_distributions=cat_params, n_iter=param_comb, scoring=self.target_metric, n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001)
		return random_search_cat

	def xgb_randomized(self,skf,X,Y,param_comb):
		xgb_params = {
			'learning_rate': [0.02,0.1],
			'min_child_weight': [1,10],
			'gamma': [0.5,10],
			'subsample': [0.1,1.0],
			'colsample_bytree': [0.3, 1.0],
			'max_depth': [3, 5]
			}		
		xgb = XGBClassifier(objective='binary:logistic') #'multi:softmax'
		random_search_xgb = RandomizedSearchCV(xgb, param_distributions=xgb_params, n_iter=param_comb, scoring=self.target_metric, n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001)
		return random_search_xgb

	def lgb_randomized(self,skf,X,Y,param_comb):
		lgb_params={
			'max_depth': [2,5],
			'learning_rate': [0.02,0.1],
			'colsample_bytree': [0.05, 0.3],
			'num_leaves': [2, 5],
			'metric': [self.target_metric],
			'max_depth': [3, 5]
		}
		lgb = LGBMClassifier(objective="binary")
		random_search_lgb = RandomizedSearchCV(xlgb, param_distributions=lgb_params, n_iter=param_comb, scoring=self.target_metric, n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001)
		return random_search_lgb

	
	def objective(trial):
		#categorical_features_indices = np.where(X.dtypes != np.float)[0]
		train_x,test_x,train_y,test_y=train_test_split(multipleboosts.X,multipleboosts.Y,test_size=0.5)
		train_pool = Pool(train_x,train_y)
		test_pool = Pool(test_x,test_y)
		params = {
		'iterations' : trial.suggest_int('iterations', 50, 300), 
		'depth' : trial.suggest_int('depth', 4, 10),       
		'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3),       
		'random_strength' :trial.suggest_int('random_strength', 0, 100),       
		'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00), 
		'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
		'od_wait' :trial.suggest_int('od_wait', 10, 50)}
		model = CatBoostClassifier(**params)
		model.fit(train_pool)
		preds = model.predict(test_pool)
		pred_labels = np.rint(preds)
		roc_auc = roc_auc_score(test_y, pred_labels)
		return 1.0 - roc_auc

if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog='kaggle_random_cv.py',usage='Use this code for exploring the best methodology and parameter for ML.',description='this code needs 2 different args')
	parser.add_argument('configuration', help = 'The location of Configuration file')
	parser.add_argument('--mode',required=False,help = 'You can select one variable from 3 parameters you can skip this parameter.',default='all',choices=['all','random_search','simple_ml'])

	args=parser.parse_args()
	now = datetime.today()
	today_str= now.strftime("%Y%m%d")


	from kaggle_temp import multipleboosts
	mulb=multipleboosts()
	mulb.set_parameters(args.configuration)
	df_train=pd.read_csv(mulb.training_file)
	train_labels1=df_train[mulb.target_name]
	train_features1=df_train.drop(mulb.id,axis=1)
	train_features1=train_features1.drop(mulb.target_name,axis=1)
	multipleboosts.X=train_features1
	multipleboosts.Y=train_labels1
	study =optuna.create_study()
	study.optimize(multipleboosts.objective,n_trials=10)
	print(study.best_params)



	'''
	xgb_params = {
		'learning_rate': [0.02],
		'min_child_weight': [3,10],
		#'gamma': [0.5,10],
		#'max_leaf_nodes': [8,9]
		#'subsample': [0.1,1.0],
		#'colsample_bytree': [0.3, 1.0],
		#'max_depth': [3, 5]
		}

	cat_params={
		#'max_depth': [2,5],
		#'learning_rate': [0.02,0.1],
		#'colsample_bylevel': [0.05, 0.3]
		#'lambda': [1,2,3],
		'max_depth': randint(2,10),
		'bagging_temperature': [0.01,0.1,1.0,10.0,100.0],
		'random_strength': randint(0,100),
		'learning_rate': [0.02,0.05],
		'colsample_bylevel': [0.05,0.3]
		}

	lgb_params={
		'max_depth': randint(-1,10),
		'learning_rate': [0.02,0.05,0.1,0.3],
		'colsample_bytree': [0.05, 0.3],
		'num_leaves': randint(2, 16),
		'metric': [mulb.target_metric],
		'max_depth': randint(3, 10)
		}

	xgb = XGBClassifier(objective='binary:logistic',eval_metric='auc')
	'''
	
	'''
	cat = CatBoostClassifier(objective="Logloss",max_depth=7, bagging_temperature=0.01, colsample_bylevel=0.05, random_strength=62, learning_rate=0.04)
	#cat_features=[0,1]
	cat.fit(train_features1,train_labels1)

	'''
	'''
	lgb = LGBMClassifier(objective="binary")


	param_comb = 3

	X=train_labels1
	Y=train_features1

	skf = StratifiedKFold(n_splits=mulb.nfolds, shuffle = True, random_state = 1001)
	random_search_cat = RandomizedSearchCV(cat, param_distributions=cat_params, n_iter=param_comb, scoring=mulb.target_metric, n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001)
	random_search_xgb = RandomizedSearchCV(xgb, param_distributions=xgb_params, n_iter=param_comb, scoring=mulb.target_metric, n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001)
	random_search_lgb = RandomizedSearchCV(lgb, param_distributions=lgb_params, n_iter=param_comb, scoring=mulb.target_metric, n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001)
	
	start_time = mulb.timer(None) # timing starts from this point for "start_time" variable
	random_search_xgb.fit(X, Y)
	mulb.timer(start_time) # timing ends here for "start_time" variable
	
	start_time = mulb.timer(None) # timing starts from this point for "start_time" variable
	random_search_cat.fit(X, Y)
	mulb.timer(start_time) # timing ends here for "start_time" variable
	

	start_time = mulb.timer(None) # timing starts from this point for "start_time" variable
	random_search_lgb.fit(X, Y)
	mulb.timer(start_time) # timing ends here for "start_time" variable
	
	if random_search_xgb.best_score_ >= random_search_cat.best_score_ and random_search_xgb.best_score_ >= random_search_lgb.best_score_:
		random_search=random_search_xgb
	elif random_search_cat.best_score_ >= random_search_xgb.best_score_ and random_search_cat.best_score_ >= random_search_lgb.best_score_:
		random_search=random_search_cat
	elif random_search_lgb.best_score_ >= random_search_cat.best_score_ and random_search_lgb.best_score_ >= random_search_xgb.best_score_:
	
		random_search=random_search_lgb
		



	print('\n All results:')
	print(random_search.cv_results_)
	print('\n Best estimator:')
	print(random_search.best_estimator_)
	print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (mulb.nfolds, param_comb))
	print(random_search.best_score_ * 2 - 1)
	print('\n Best hyperparameters:')
	print(random_search.best_params_)

	try:
		random_search.best_estimator_.save_model(mulb.model_location)
	except:
		try:
			joblib.dump(random_search.best_estimator_, mulb.model_location)
		except TypeError:
			print(e)
	'''
	'''
	cat.fit(train_features1,train_labels1)
	df_test=pd.read_csv(mulb.test_file)
	df_label_test=df_test.drop(mulb.id,axis=1)
	y_test = cat.predict_proba(df_label_test)
	y_pred=pd.DataFrame(y_test)
	#y_pred=y_pred.drop('0',axis=1)
	df_answer=pd.concat([df_test[mulb.id],y_pred],axis=1)
	#df_answer.columns=[mulb.id,mulb.target_name]
	output_name=mulb.submission_file
	df_answer.to_csv(output_name,index=False)
	'''