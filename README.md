# Data handling on GCP

This is a documentation to describe data handling codeset for GCP.
***

## Overview
This repository aims to help data scientists to minimize and simplify preparation process of data environment for big data.
You can easily upload huge dataset to GCS and handle them on BigQuery.
The most frequent usecase is the data science competition like kaggle.

***
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.


### Prerequisites
Using pipenv as version and environment management tool.
If you don't have pipenv, you need to install pipenv.
```
pip install pipenv
```

Download pipfile and locate your working folder.
Put the following commands.
```
pipenv --python 3
pipenv install
pipenv shell
```

You need to fill yaml file as below before executing the code.
```
local:
  upload_folder: './'

  credential_file: ''xxxxxxxx'
  
cloud:
  project: 'xxxxxxxx'
  bucket: 'xxxxxxxx'
  folder: 'xxxxxxxx'
  BQ_project: 'xxxxxxxx'
  BQ_datasource: 'xxxxxxxx'
  BQ_table: ''
```

*** 

## Code Structure
tbd


### IQO_executor.py
This code pulls all parameters from yaml file and upload all files in the folder to GCS.
The actual example is below.
```
pipenv run python upload_to_gcs.py './config.yml'

```



## Output
TBD


## Data Definition



### Consideration


[jupyter notebook]: https://github.com/dan-global/iqo-google/blob/master/IQO%20data%20extraction-dataintegration-20181123.ipynb

