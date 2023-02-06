#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd


# In[42]:


df=pd.read_csv(r"D:\Downloads\Telecom_customer churn.csv")


# In[43]:


df1=df.copy();
#getting 80% of the data for the training set.
training=df1.sample(frac=.8)
#just reviewing the effect of using the sample function on the dataframe
training


# In[95]:


corr1 = df.copy()

Correlation = corr1.corr()['churn']
print(Correlation)


#In[61]:


#getting 20% of the data for the test set.
test=df1.drop(training.index);
test
test_num = test.select_dtypes(exclude=["object_","bool_"])

for column in test_num :
        test_num[column].fillna(test_num[column].min()-0.5,inplace=True);

test_cat = test.select_dtypes(exclude=["number","float_"])

for column in test_cat :
    test_cat[column].fillna("null",inplace=True)


# In[45]:


#getting the numerical values for the training dataset  
training_num=training.select_dtypes(exclude=["object_","bool_"]);


# In[46]:


for column in training_num :
        training_num[column].fillna(training_num[column].min()-0.5,inplace=True);


# In[47]:


training_num.isnull().sum()


# In[48]:


get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from scipy.stats._continuous_distns import _distn_names
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range','gumbel_r','gumbel_l','kstwo','burr12','gompertz','invgamma','norminvgauss','maxwell','kappa3','ncf','t','pareto','powernorm','semicircular','uniform']]):

        print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

        distribution = getattr(st, distribution)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                
                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                
                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                best_distributions.append((distribution, params, sse))
        
        except Exception:
            pass

    
    return sorted(best_distributions, key=lambda x:x[2])


# In[49]:


import numpy as np
def get_pdf_prob(a, dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg) 
    pdf = pd.Series(y, x)
    arrx = pdf.index
    ##### we will find the closest number to the input (the value of the column of this client)
    absolute_val_array = np.abs(arrx - a)
    smallest_difference_index = absolute_val_array.argmin()
    closest_element = arrx[smallest_difference_index]
    
    index = np.where(arrx == closest_element)
    yval = y[index[0][0]]
    
    return  yval


# In[50]:


def get_pmf_prob(find,clientNumber,j):
    column = training_cat.iloc[clientNumber].index[j]
    pmf = training_cat[column].value_counts().sort_index() / len(training_cat)
    arrx = pmf.index
    index = np.where(arrx == find) # find is the string to find
    yval = pmf.iloc[index[0][0]]
    
    return yval
 


# In[60]:


def get_pmf_prob_churn(find,clientNumber,j):
    column = churn_cat.iloc[clientNumber].index[j]
    pmf = churn_cat[column].value_counts().sort_index() / len(churn_cat)
    arrx = pmf.index
    index = np.where(arrx == find) # find is the string to find
    yval = pmf.iloc[index[0][0]]
    
    return yval


# In[51]:


churn_d = training.copy()
churn_d.drop(churn_d[churn_d.churn == 0].index, inplace = True)
prChurn = len(churn_d)/len(training)
print("prChurn = ",+ prChurn)


# In[52]:


# FINDING MISSING CAT DATA AND SUBSTITUTING THE NULLS WITH NULL AND CREATING A NEW CATEGORY  

training_cat = training.select_dtypes(exclude=["number","float_"])

for column in training_cat :
    training_cat[column].fillna("null",inplace=True)   


# In[53]:


churn_num_data=churn_d.select_dtypes(exclude=["object_","bool_"])

for column in churn_num_data :
        churn_num_data[column].fillna(churn_num_data[column].min()-0.5,inplace=True);


# In[54]:


churn_cat = churn_d.select_dtypes(exclude=["number","float_"])

for column in churn_cat :
    churn_cat[column].fillna("null",inplace=True)


# In[97]:


# numerator pdf

churnlength = len(churn_num_data)
churndist = []
churnparams = []


#for i in range (0, churnlength):
# for i in range(0, 4):
for i in ['change_mou','peak_vce_Mean','mou_peav_Mean','mou_Mean']:
    data = pd.Series(churn_num_data[i])


    # # Find best fit distribution
    best_distibutions = best_fit_distribution(data, 200 , ax);
    best_dist = best_distibutions[0]
    
    churndist.append(best_dist[0])
    churnparams.append(best_dist[1])
    print(churndist)
    print(churnparams)
    
    print("Column is done")


# In[98]:


# denominator pdf

generallength = len(churn_num_data)
generaldist = []
generalparams = []


#for i in range (0, generallength):
for i in ['change_mou','peak_vce_Mean','mou_peav_Mean','mou_Mean']:
    data =pd.Series(training_num[i])

    # # Find best fit distribution
    best_distibutions = best_fit_distribution(data, 200 , ax);
    best_dist = best_distibutions[0]
    
    generaldist.append(best_dist[0])
    generalparams.append(best_dist[1])
    print(generaldist)
    print(generalparams)
    
    print("Column is done")


# In[99]:


def willChurn(c_num,c_cat,cNumber): 
    
    
    #numerator given churn
    
    churnResult = 1
    
    # pdf multiplication given churn
#    for i in ['change_mou','peak_vce_Mean','mou_peav_Mean','mou_Mean']:
#     for i in [1,9,35,37]:
#         cpar = c_num[i]
#         generalResult = generalResult * get_pdf_prob(cpar, churndist[i], churnparams[i])
    
    for i in [1,9,35,37]:
        cpar = c_num[i]
        if i == 1:
            churnResult = churnResult * get_pdf_prob(cpar, churndist[0], churnparams[0])
        if i == 9:
            churnResult = churnResult * get_pdf_prob(cpar, churndist[1], churnparams[1])
        if i == 35:
            churnResult = churnResult * get_pdf_prob(cpar, churndist[2], churnparams[2])
        if i == 37:
            churnResult = churnResult * get_pdf_prob(cpar, churndist[3], churnparams[3])
    
    # pmf multiplication given churn
    
    for j in range(0,4):
        cpar = c_cat[j]
        churnResult = churnResult * get_pmf_prob_churn(cpar, cNumber, j)
    
    
    
    ##############################################
    
    #denominator
    
    generalResult = 1
    
    # pdf multiplication general
    
#    for i in ['change_mou','peak_vce_Mean','mou_peav_Mean','mou_Mean']:
#     for i in [1,9,35,37]:
#         cpar = c_num[i]
#         generalResult = generalResult * get_pdf_prob(cpar, generaldist[i], generalparams[i])
    
    for i in [1,9,35,37]:
        cpar = c_num[i]
        if i == 1:
            churnResult = churnResult * get_pdf_prob(cpar, generaldist[0], generalparams[0])
        if i == 9:
            churnResult = churnResult * get_pdf_prob(cpar, generaldist[1], generalparams[1])
        if i == 35:
            churnResult = churnResult * get_pdf_prob(cpar, generaldist[2], generalparams[2])
        if i == 37:
            churnResult = churnResult * get_pdf_prob(cpar, generaldist[3], generalparams[3])
    
    # pmf multiplication general
    
    for j in range(0,4):
        cpar = c_cat[j]
        churnResult = churnResult * get_pmf_prob(cpar, cNumber, j)
        
    ##############################################
    
    prchurnrow = (churnResult/generalResult)*prChurn
    
    print("Client will churn: " + str(prchurnrow*100) + " %")
    
    return prchurnrow*100


# In[100]:


willChurn(test_num.iloc[2005],test_cat.iloc[2005], 2005)


# In[101]:


print(test_num.iloc[2005][48])


# In[102]:


print(test.iloc[134])


# In[88]:


import math

testlen = len(test_num)
correctCounter = 0
#for i in range(0,testlen):
for i in range(0, 2000):
    check = willChurn(test_num.iloc[i],test_cat.iloc[i],i)
    if check >= 0.5:
        churndecision = 1.0
    else: 
        churndecision = 0.0
    
    if int(churndecision) == int(test_num.iloc[i][48]):
        correctCounter = correctCounter + 1
    
    print(i)    

accuracy = correctCounter/2000
print(str(math.ceil(accuracy*100)) + "%")
        


# In[178]:


import math
print(str(math.ceil(accuracy*100)) + "%")


# In[22]:


print(test_num.iloc[134].index[2])


# In[25]:


for column in training_cat:
    pmf = training_cat[column].value_counts().sort_index() / len(training_cat)
    print(pmf)
    print()


# In[37]:


column = training_cat.iloc[134].index[5]
pmf = training_cat[column].value_counts().sort_index() / len(training_cat)
print(pmf)
arrx = pmf.index
print(arrx)
index = np.where(arrx == 'T')
print(index[0][0])
yval = pmf.iloc[index[0][0]]
print(yval)


# In[ ]:




