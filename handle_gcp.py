import argparse
import json
import yaml
import os
import re
import datetime

import pandas as pd
from pathlib import Path
import pandas_gbq.gbq as gbq
import pandas_gbq


from google.cloud import storage
from google.cloud import bigquery
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient import discovery
import dask.dataframe as dd
import gcsfs


class gcptransfer:

	def __init__(self):
		self.gcs_to_bq=True
		self.local_to_gcs=True
		self.credential_file=''

	def from_gcs_to_bq(gcs_url,bq_project_id,bq_dataset_id,bq_client):
		#bq_client=bigquery.Client(project=bq_project_id
		#	,credentials=credentials
		#	)

		#bq_client=bigquery.Client.from_service_account_json(credential_file)

		table_parts=gcs_url.split('/')

		dataset_id=bq_dataset_id
		dataset_ref = bq_client.dataset(dataset_id)
		try:
			dataset=bq_client.get_dataset(dataset_ref)
		except Exception as e:
			dataset = bigquery.Dataset(dataset_ref)
			dataset = bq_client.create_dataset(dataset)


		job_config = bigquery.LoadJobConfig()
		#job_config.autodetect = True

		if re.compile('-0').search(gcs_url):
			job_config.write_disposition = 'WRITE_TRUNCATE'
			tablename=table_parts[-1][:-6]

		elif re.compile('-[1-9]').search(gcs_url):
			job_config.write_disposition = 'WRITE_APPEND'
			tablename=table_parts[-1][:-6]

		else:
			job_config.write_disposition = 'WRITE_TRUNCATE'
			tablename=table_parts[-1][:-4]
			#job_config.write_disposition = 'WRITE_EMPTY'
		bq_table_location=bq_project_id+'.'+bq_dataset_id+'.'+tablename
		#job_config.time_partitioning = bigquery.table.TimePartitioning(field='datetime_local')
		job_config.skip_leading_rows = 1
		job_config.source_format = bigquery.SourceFormat.CSV
		df=dd.read_csv(gcs_url,encoding='utf-8',dtype='object')

		df=df.get_partition(1).compute()
		df=gcptransfer.change_dtype(df)
		df_schema=gbq.generate_bq_schema(df, default_type='STRING')


		job_config.schema=gcptransfer.schema_generator(df_schema)


		job_config.source_format = bigquery.SourceFormat.CSV
		
		load_job = bq_client.load_table_from_uri(gcs_url,dataset_ref.table(tablename),job_config=job_config)

		job_done= False
		while job_done == False:
			try:
				load_job.result()
				job_done=True
				print('Job finished.  {}  is uploaded to {}'.format(gcs_url,bq_table_location))
			except Exception as e:
				print(e)


	def change_dtype(df):
		columns=df.columns
		for column in columns:
			if str(df.loc[0,column]).isdigit():
				df[column]=pd.to_numeric(df[column],errors='coerce')
		return df



	def schema_generator(df_schema):
		bq_schema=[]
		for column_info in df_schema["fields"]:
			bq_column=bigquery.SchemaField(column_info["name"],column_info["type"])
			bq_schema.append(bq_column)
		return bq_schema


	def from_local_to_gcs(bucket_name, source_file_name,destination_blob_name,storage_client):
		path_list=source_file_name.split('/')
		filename=path_list[-1]
		destination=destination_blob_name+'/'+filename
		#storage_client = storage.Client()
		#storage_client=storage.Client().from_service_account_json(credential_file)
		bucket = storage_client.get_bucket(bucket_name) #I tried to add my folder here
		blob = bucket.blob(destination)
		destination_urls=[]
		try:
			blob.upload_from_filename(source_file_name)
			print('File {} uploaded to {}.'.format(source_file_name,destination_blob_name))
			destination_url='gs://'+bucket_name+'/'+destination
			destination_urls.append(destination_url)
		except MemoryError: 
			i=1
			chunksize=256 * 1024
			for chunk in pd.read_csv(source_file_name, chunksize=chunksize):
				destination_url='gs://'+bucket_name+'/'+destination[:-4]+'-'+i+'-*.csv'
				dask_df=dd.from_pandas(df,npartitions=1)
				dask_df.to_csv(destination_url)
				print('{}: File {} uploaded to {}.'.format(i,source_file_name,destination_url))
				destination_urls.append(destination_url)
				i+=1
		return destination_urls


	def search_gcs(bucket_name,destination_blob_name):
		client = discovery.build('storage', 'v1beta2')
		request = client.objects().list(bucket=bucket_name,prefix=destination_blob_name)
		while request is not None:
			response = request.execute()
			print(json.dumps(response, indent=2))
			request = request.list_next(request, response)



	def get_schema_local(source_file_name):
		df_schema=pd.read_csv(source_file_name,nrows=10)
		schema=gbq.generate_bq_schema(df_schema, default_type='STRING')
		return schema

	def get_schema_gcs(gcs_url):
		df_schema=dd.read_csv(gcs_url,nrows=10)
		df_schema=df_schema.compute()
		schema=gbq.generate_bq_schema(df_schema, default_type='STRING')
		return schema


	def from_local_to_bq(project_id,dataset_id,source_file_name,bq_client):
		path_list=source_file_name.split('/')
		table_id=path_list[-1]
		chunksize=256 * 1024

		dataset_id=bq_dataset_id
		dataset_ref = bq_client.dataset(dataset_id)
		try:
			dataset=bq_client.get_dataset(dataset_ref)
		except Exception as e:
			dataset = bigquery.Dataset(dataset_ref)
			dataset = bq_client.create_dataset(dataset)
		

		destination_bq=dataset_id+'.'+table_id
		print(destination_bq)
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
		return upload_folder,credential_file,project,bucket,folder


	def file_search(upload_folder):
		p = Path(upload_folder)
		file_list=list(p.glob("*.csv"))
		f_list=[]
		for f in file_list:
			f_list.append(f.as_posix())
		return f_list

	def gcp_client(credential_file):
		storage_client = storage.Client().from_service_account_json(credential_file)
		bq_client=bigquery.Client().from_service_account_json(credential_file)
		return storage_client,bq_client


if __name__ == "__main__":
	from handle_gcp import gcptransfer
	print(datetime.datetime.now())
	parser = argparse.ArgumentParser(description='You need to specify the location of config file and file option')
	parser.add_argument('configuration', help = 'the location of configuration file')
	args=parser.parse_args()
	upload_folder,credential_file,project,bucket,folder=gcptransfer.read_configuration(args.configuration)

	storage_client,bq_client=gcptransfer.gcp_client(credential_file)
	file_list=gcptransfer.file_search(upload_folder)


	
	for f in file_list:
		print(f)
		gcs_urls=gcptransfer.from_local_to_gcs(bucket,f,folder,storage_client)
		'''
		for index, gcs_url in enumerate(gcs_urls):
			print("gcs file, " + gcs_url + ", is in position " + str(index) + ".")
		'''

		for gcs_f in gcs_urls:
			print(gcs_f)
			gcptransfer.from_gcs_to_bq(gcs_f,project,folder,bq_client)

		
		#from_local_to_bq(project,folder,f,bq_client)