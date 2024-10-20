# Store Sales Forecasting with Time Series Models and CI/CD Pipeline

## Project Overview

This project demonstrates the use of advanced time series forecasting techniques to predict store sales for Favorita stores. It showcases the implementation of both Amazon SageMaker's DeepAR algorithm and a custom deep learning time series model, along with the setup of a CI/CD pipeline in AWS for model deployment and monitoring.

## Data Source

The data used in this project is the [Store Sales Time Series Forecasting](https://www.kaggle.com/code/ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide#11.-Exponential-Moving-Average) dataset from Kaggle. The data was stored in S3 and contained the following files:

#### train.csv

The training data, comprising time series of features store_nbr and onpromotion as well as the target sales.

store_nbr identifies the store at which the products are sold.

sales gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips).

onpromotion gives the total number of items in a product family that were being promoted at a store at a given date.

#### stores.csv

Store metadata, including city, state, type, and cluster (cluster is a grouping of similar stores).

#### oil.csv

Daily oil price. Includes values during both the train and test data timeframes. (Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.)

#### holidays_events.csv

Holidays and Events, with metadata.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Loading](#data-loading)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Model Development](#model-development)
   - [Baseline Linear Regression](#baseline-linear-regression)
   - [Custom Time Series Model](#custom-time-series-model)
   - [DeepAR Model](#deepar-model)
7. [Model Deployment](#model-deployment)
8. [CI/CD Pipeline Setup](#cicd-pipeline-setup)
9. [Tools and Technologies](#tools-and-technologies)

## Environment Setup

To set up the project environment:

1. Open the `0-Environment_Setup.ipynb` notebook.
2. Run all cells to install necessary libraries and set up AWS credentials.
3. This notebook will also define global variables and functions used throughout the project.

## Data Loading

To load the project data:

1. Open the `1 - Load Data.ipynb` notebook.
2. Run all cells to load data from the local environment into the AWS S3 datalake.
3. The notebook uses the AWS CLI to copy CSV files to the specified S3 bucket.

## Exploratory Data Analysis

The `2 - Exploratory_Data_Analysis_w_expanded_EDA.ipynb` notebook contains a comprehensive analysis of the dataset, including:

- Sales trends by store characteristics
- Impact of holidays on sales
- Effect of promotions
- Relationship between transactions and sales
- Influence of oil prices on sales

## Data Preprocessing

The `3 - Data Preprocessing.ipynb` notebook covers:

- Data cleaning
- Handling missing values
- Date/time feature extraction

## Feature Engineering

The `5 - Feature Engineering.ipynb` notebook details the creation of:

- Time-based features
- Store characteristic features
- Economic indicators
- Holiday and promotion features

## Model Development

### Baseline Linear Regression

The `7.0 - Baseline Model Linear Regression Development.ipynb` notebook establishes a baseline model for comparison.

### Custom Time Series Model

The `7.1 - Custom Time Series Model Development.ipynb` notebook covers the development of a custom deep learning model for time series forecasting.

### DeepAR Model

The `7.2 - Model Training DeepAR.ipynb` notebook demonstrates the use of Amazon SageMaker's DeepAR algorithm for forecasting.

## Model Deployment

- Custom Model: `8.1 - Deploy Custom Time Series Model.ipynb`
- DeepAR Model: `8.2 - Deploy model DeepAR.ipynb`

These notebooks cover the process of deploying the trained models to Amazon SageMaker endpoints.

## CI/CD Pipeline Setup

- Custom Model: `10.1 - CICD Pipeline Custom Time Series Model.ipynb`
- DeepAR Model: `10.2 - CICD Pipeline DeepAR.ipynb`

These notebooks detail the setup of CI/CD pipelines for automated model training, evaluation, and deployment using AWS services.

## Tools and Technologies

This project utilizes a range of AWS services and machine learning tools:

- Amazon SageMaker: For model training, deployment, and pipeline orchestration
- Amazon S3: Data storage and model artifacts
- Amazon Athena: For querying data in S3
- AWS Glue: Data catalog and ETL jobs
- Amazon CloudWatch: For monitoring and logging
- Pandas, NumPy, Scikit-learn: Data manipulation and preprocessing
- TensorFlow: Custom model development
- Matplotlib, Seaborn: Data visualization

## Getting Started

1. Clone this repository.
2. Ensure you have the necessary AWS permissions and credentials set up.
3. Follow the notebooks in order, starting with `0-Environment_Setup.ipynb`.
4. Each notebook contains detailed instructions and explanations for each step of the process.

## Contributors

@t4ai / Tyler Foreman

@julietlawton / Juliet Lawton (commits from "root" are also Juliet)

@Yoha02 / Eyoha Gir

## License

Apache License Version 2.0, January 2004
http://www.apache.org/licenses/
