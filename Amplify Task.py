#!/usr/bin/env python
# coding: utf-8

# ## Preliminaries

# In[ ]:


# import packages
import numpy as np
import pandas as pd
import xgboost as xgb

import sys
import os
from copy import deepcopy

import datetime
import time

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sklearn.metrics as skmetrics

import shap
from pdpbox import pdp, info_plots
import pdpbox


# In[ ]:


main_path = r"C:\Users\c54827a\OneDrive - EXPERIAN SERVICES CORP\Python\Amplify"


# ## Data Prep

# ### Load Data

# In[ ]:


dest_path = os.path.join(main_path, r"train_small.csv")
df = pd.read_csv(dest_path)
print(df.shape)
df.head()


# In[ ]:


# Load Data Dict
dest_path = os.path.join(main_path, r"Data Dictionary.xlsx")
data_dict = pd.read_excel(dest_path, sheet_name = 'DataDict', index_col=1)


# ### Calculate summary stats

# In[ ]:


summary_stats = df.describe().transpose()
summary_stats['description'] = data_dict['Description']
summary_stats['data type'] = df.dtypes
summary_stats['# unique'] = df.nunique()
summary_stats['# missing'] = df.isna().sum()
summary_stats['% missing'] = round(((summary_stats['# missing']/(summary_stats['count'] + summary_stats['# missing'])) * 100), 2)

# export
dest_path = os.path.join(main_path, r"summary_statistics.xlsx")
summary_stats.to_excel(dest_path, sheet_name = "Summary")


# ### Go through some examples

# In[ ]:


dest_path = os.path.join(main_path, r"one_listing.xlsx")
df[df['listing_id'] == 27525].to_excel(dest_path, sheet_name = "27525")


# In[ ]:


df.user_country_id.value_counts(normalize=True, dropna = False)


# The data is heavily skewed towards one country (id = 219); it takes up 58.4% of the population so any model tested on a different country might not perform so well

# In[ ]:


df.site_id.value_counts(normalize=True, dropna = False)


# The data is heavily skewed towards one site (id = 5); it takes up 62.2% of the population so any model tested on a different site might not perform so well

# ### Check the timestamp frequency of the data

# In[ ]:


df.timestamp.value_counts(dropna = False, normalize = True)


# In[ ]:


df.timestamp.isna().sum()


# In[ ]:


# Get the date into quarters
df['timestamp_DT'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
df['timestamp_Q'] = df['timestamp_DT'].dt.to_period('Q')
df[['timestamp', 'timestamp_Q']].head()


# In[ ]:


df.timestamp_Q.value_counts(dropna = False, normalize = True)


# ### Check for uniqueness

# In[ ]:


# sorting
df.sort_values(["search_id", "listing_id"], inplace = True)
  
# making a bool series
df['duplicate_flag'] = df.duplicated(subset = ["search_id", "listing_id"], 
                                                              keep = 'first') 
df['duplicate_flag'].value_counts()


# The data is unique by the combination of "search_id" and "listing_id".

# ### Check the correlations between features

# In[ ]:


corr_matrix = df.corr(method = 'pearson')
dest_path = os.path.join(main_path, r"Correlation Matrix.xlsx")
corr_matrix.to_excel(dest_path, sheet_name = "Matrix")


# In[ ]:


pd.crosstab(df['booked'], df['booking_value'].isna(), margins = True)


# ### Imbalanced Data

# In[ ]:


df.clicked.value_counts(dropna = False, normalize = True)


# In[ ]:


df.booked.value_counts(dropna = False, normalize = True)


# ### Privacy Issues

# In[ ]:


dest_path = os.path.join(main_path, r"Site Country Crosstab.xlsx")
pd.crosstab(df['site_id'], df['user_country_id'], margins = True).to_excel(dest_path, sheet_name = "Crosstab")


# ## CTR and Conversion Rate

# In[ ]:


# CTR
df['clicked_count'] = df.groupby('listing_id')['clicked'].transform('sum')
df['impressions_count'] = df.groupby('listing_id')['clicked'].transform('count')
df['CTR'] = round(((df['clicked_count']/df['impressions_count']) * 100), 2)


# In[ ]:


# Conversion Rate
df['booked_count'] = df.groupby('listing_id')['booked'].transform('sum')
df['conversion_rate'] = round(((df['booked_count']/df['impressions_count']) * 100), 2)


# ### Correlation

# In[ ]:


corr_rates_matrix = df[['listing_stars', 'listing_review_score','CTR','conversion_rate']].corr(method = 'pearson')
corr_rates_matrix


# ### Linear Regression

# In[ ]:


x = df[['listing_stars', 'listing_review_score']].copy()


# In[ ]:


print(np.any(np.isnan(x['listing_stars'])))
print(np.any(np.isnan(x['listing_review_score'])))


# In[ ]:


x['listing_review_score'].fillna(value = 0, inplace=True)
x['listing_review_score'].value_counts(dropna= False)


# In[ ]:


x['listing_stars'].value_counts(dropna= False)


# In[ ]:


def QD_LinReg (x, y):

    # importing train_test_split from sklearn
    from sklearn.model_selection import train_test_split
    # splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    
    # importing module
    from sklearn.linear_model import LinearRegression
    # creating an object of LinearRegression class
    LR = LinearRegression()
    # fitting the training data
    LR.fit(x_train, y_train)
    
    y_prediction =  LR.predict(x_test)
    
    # importing r2_score module
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    # predicting the accuracy score
    score = r2_score(y_test, y_prediction)
    print('r2 score is ', score)
    print('mean_sqrd_error is ', mean_squared_error(y_test, y_prediction))
    print('root_mean_squared error of is ', np.sqrt(mean_squared_error(y_test, y_prediction)))
    
#     return x_train, x_test, y_train, y_test, y_prediction


# In[ ]:


QD_LinReg(x, df[['conversion_rate']].copy())


# In[ ]:


QD_LinReg(x, df[['CTR']].copy())


# In[ ]:


QD_LinReg(x[['listing_stars']], df[['CTR']].copy())


# In[ ]:


QD_LinReg(x[['listing_review_score']], df[['CTR']].copy())


# In[ ]:


QD_LinReg(x[['listing_stars']], df[['conversion_rate']].copy())


# In[ ]:


QD_LinReg(x[['listing_review_score']], df[['conversion_rate']].copy())


# ## Modelling

# ### Stratified Random Sampling

# In[ ]:


# randomizing by criteria 
from sklearn.model_selection import train_test_split

start = time.time()
df_use, df_unused = train_test_split(df, train_size=100000, random_state=1234, 
                                     stratify=df[['timestamp_Q', 'site_id', 'booked']])
end = time.time()
print(end - start)

print(df_use.shape)
print(df_unused.shape)


# In[ ]:


dest_path = os.path.join(main_path, r"df_use.pkl")
df_use.to_pickle(dest_path)


# In[ ]:


### Load data
dest_path = os.path.join(main_path, r"df_use.pkl")
# load the dataframe from pickle
df_use = pd.read_pickle(dest_path)
df_use.shape


# In[ ]:


df_use.booked.value_counts(normalize = True)


# ### XGBoost

# ### Prepare sample

# In[ ]:


from sklearn.model_selection import train_test_split

X = df_use[[
'listing_stars',
'listing_review_score',
'is_brand',
'location_score1',
'log_historical_price',
'length_of_stay',
'has_promotion',
'booking_window',
'num_rooms',
'stay_on_saturday',
'distance_to_dest',
'random_sort']].copy()

y = df_use[['booked']].copy()

# Create 80/20 split using sklearn package

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


# Set the scale_pos_weight
scale_pos_weight = y_train.value_counts().max() / y_train.value_counts().min()
print(scale_pos_weight)


# ### CV Approach

# In[ ]:


import json
from scipy.sparse import load_npz, vstack
from sklearn.model_selection import PredefinedSplit, StratifiedKFold, RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from scipy.stats import distributions


# In[ ]:


# Set the parameters and let the optimization task find the optimum:

cv_params =     {
    "max_depth"        : distributions.randint(low=3, high=10),
    "n_estimators"     : distributions.randint(low=2, high=50),
    "subsample"        : distributions.uniform(loc=0.5, scale=0.5), # uniform [0.5 to 1]
    "colsample_bytree" : distributions.uniform(loc=0.5, scale=0.5), # uniform [0.5 to 1]
    "colsample_bylevel": distributions.uniform(loc=0.5, scale=0.5), # uniform [0.5 to 1]
    "gamma"            : [0, 0.1, 0.3, 0.6, 1, 2, 10, 20],
    "reg_alpha"        : [0, 0.001, 0.005, 0.01, 0.05, 0.1, 1, 10],
    "reg_lambda"       : [0, 0.001, 0.005, 0.01, 0.05, 0.1, 1, 10],
    "learning_rate"    : distributions.uniform(loc=0.0, scale=0.5),
    "min_child_weight" : distributions.randint(low=5, high=30)
}

ind_params = {'objective': 'binary:logistic',
              'seed': 42,
              'random_state': 42,
              'scale_pos_weight': scale_pos_weight,
              'tree_method': 'exact'
             }

optimized_model = RandomizedSearchCV(xgb.XGBClassifier(**ind_params),
                                     cv_params,
                                     scoring = 'roc_auc',
                                     cv = StratifiedKFold(n_splits=5, random_state=None),
                                     n_jobs = -1,
                                     return_train_score = True,
                                     random_state = 42,
                                     n_iter = 100,
                                     verbose = 1
                                     )


# In[ ]:


# Fit the parameters with the data available
optimized_model.fit(X_train, y_train)


# In[ ]:


# Create a dataframe from the matrix with the cross-validation results
cv_results = pd.DataFrame(optimized_model.cv_results_).sort_values('mean_test_score', ascending=False)

# Calculate Development/Validation Gini coefficients from AUC metrics
cv_results['dev_gini'] = cv_results['mean_train_score'] * 2 - 1
cv_results['val_gini'] = cv_results['mean_test_score'] * 2 - 1
cv_results['diff_gini'] = abs(cv_results['dev_gini'] - cv_results['val_gini'])

# Save the results in Excel file
dest_path = os.path.join(main_path, r"XGB CV.xlsx")
cv_results.to_excel(dest_path)


# ### Model & Performance

# In[ ]:


# Get the best estimator from the matrix
optimized_model.best_estimator_


# In[ ]:


print("AUC for the CV test samples: ", optimized_model.best_score_)


# In[ ]:


print("Gini for the CV test samples: ", optimized_model.best_score_ * 2 - 1)


# In[ ]:


# Load this XGBoost version
opt_params = {'colsample_bylevel': 0.6797455756098776, 'colsample_bytree': 0.6467959221322467, 'gamma': 10, 
              'learning_rate': 0.20989042822313825, 'max_depth': 4, 'min_child_weight': 29, 'n_estimators': 25, 
              'reg_alpha': 0.05, 'reg_lambda': 0.01, 'subsample': 0.7556711994304689}

optimized_model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', max_delta_step=0,
                            missing=np.NaN, n_jobs=-1, nthread=-1, objective='binary:logistic', random_state=42,
                           scale_pos_weight=scale_pos_weight, seed=42, silent=True,
                           **opt_params)

#Fit the object
optimized_model.fit(X_train, y_train)


# In[ ]:


# save
dest_path = os.path.join(main_path, r"optimized_model.pkl")
optimized_model.save_model(dest_path)


# #### Test Performance

# In[ ]:


AUC = skmetrics.roc_auc_score(y_true=y_test, y_score=optimized_model.predict_proba(X_test)[:, 1])
Gini = 2 * AUC - 1
print("AUC for the holdout sample: ", AUC)
print("Gini for the holdout sample: ", Gini)


# ### Confusion Matrices

# In[ ]:


pd.crosstab(y_test['booked'], optimized_model.predict(X_test))


# In[ ]:


pd.crosstab(y_train['booked'], optimized_model.predict(X_train))


# In[ ]:


# ROC curves for the two models
fpr_full, tpr_full, thresholds_full = skmetrics.roc_curve(y_true=y_test, y_score=optimized_model.predict_proba(X_test)[:, 1])
roc_auc_full = skmetrics.auc(fpr_full, tpr_full)

plt.figure()
lw = 1

plt.plot(fpr_full, tpr_full, color='darkred', lw=lw,
         label='AUC = {:0.3f}; Gini = {:0.3f}'.format(
             roc_auc_full, (2 * roc_auc_full - 1))
)


# Slightly above the diagonal line

# ### Feature Importance

# In[ ]:


xgb.plot_importance(
  optimized_model, 
  max_num_features=20, 
  importance_type = 'weight', 
  grid=False,
  title='Feature importance (weight)'
)

xgb.plot_importance(
  optimized_model, 
  max_num_features=20, 
  importance_type = 'gain', 
  grid=False,
  title='Feature importance (gain)'
)


# ### SHAP Values

# In[ ]:


explainer = shap.TreeExplainer(optimized_model)
shap_values = explainer.shap_values(X_test)


# In[ ]:


shap.summary_plot(shap_values, X_test, plot_type="bar")


# In[ ]:


shap.summary_plot(shap_values, X_test)


# In[ ]:


df_use.random_sort.value_counts(dropna = False)


# In[ ]:




