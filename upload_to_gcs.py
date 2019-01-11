import argparse
from google.cloud import storage
from google.cloud import bigquery
from oauth2client.service_account import ServiceAccountCredentials
import dask.dataframe as dd
import yaml
import gcsfs
import os
import pandas as pd
from pathlib import Path
import pandas_gbq



def from_local_to_gcs(bucket_name, source_file_name, destination_blob_name):
	path_list=source_file_name.split('/')
	filename=path_list[-1]
	destination=destination_blob_name+'/'+filename
	storage_client = storage.Client()
	bucket = storage_client.get_bucket(bucket_name) #I tried to add my folder here
	blob = bucket.blob(destination)

	try:
		blob.upload_from_filename(source_file_name)
	except MemoryError: 
		i=1
		chunksize=256 * 1024
		for chunk in pd.read_csv(source_file_name, chunksize=chunksize):
			destination_url='gs://'+bucket_name+destination[:-4]+'-'+i+'-*.csv'
			dask_df=dd.from_pandas(df,npartitions=1)
			dask_df.to_csv(destination_url)
			i+=1

	print('File {} uploaded to {}.'.format(
	source_file_name,
	destination_blob_name))

def get_schema(source_file_name):
	df_schema=pd.read_csv(source_file_name,nrows=10)
	schema=gbq.generate_bq_schema(df_schema, default_type='STRING')
	return schema


def from_local_to_bq(project_id,dataset_id,source_file_name):
	path_list=source_file_name.split('/')
	table_id=path_list[-1]
	chunksize=256 * 1024
	client=bigquery.Client()
	datasets = list(client.list_datasets())
	if dataset_id not in datasets:
		dataset_ref=client.dataset(dataset_id)
		dataset=bigquery.Dataset(dataset_ref)
		dataset.location='US'
		dataset=client.create_dataset(dataset)

	destination_bq=dataset_id+'.'+table_id
	try:
		df=pd.read_csv(source_file_name)
		df.to_gbq(destination_bq, project_id=project_id,if_exists='replace')
	except MemoryError:
		for chunk in pd.read_csv(source_file_name, chunksize=chunksize):
			chunk.to_gbq(destination_bq, project_id=project_id,chunksize=chunksize,if_exists='append')


def read_configuration(configuration_file):
	with open(configuration_file, 'r') as ymlfile:
		cfg = yaml.load(ymlfile)
	upload_folder=cfg['local']['upload_folder']
	credential_file=cfg['local']['credential_file']
	project=cfg['cloud']['project']
	bucket=cfg['cloud']['bucket']
	folder=cfg['cloud']['folder']
	BQ_project=cfg['cloud']['BQ_project']
	BQ_datasource=cfg['cloud']['BQ_datasource']
	BQ_table=cfg['cloud']['BQ_table']

	return upload_folder,credential_file,project,bucket,folder,BQ_project,BQ_datasource,BQ_table

def file_search(upload_folder):
	p = Path(upload_folder)
	file_list=list(p.glob("*.csv"))
	f_list=[]
	for f in file_list:
		f_list.append(f.as_posix())
	return f_list


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('configuration', help = 'the location of configuration file')
	args=parser.parse_args()
	upload_folder,credential_file,project,bucket,folder,BQ_project,BQ_datasource,BQ_table=read_configuration(args.configuration)
	file_list=file_search(upload_folder)
	for f in file_list:
		from_local_to_gcs(bucket,f,folder)
		print('Uploaded '+f)
		from_local_to_bq(BQ_project,BQ_datasource,f)