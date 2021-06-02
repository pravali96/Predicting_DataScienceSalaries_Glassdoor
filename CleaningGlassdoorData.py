#!/usr/bin/env python
# coding: utf-8

# In[142]:


import pandas as pd
df=pd.read_csv('glassdoor_jobs.csv')


# In[143]:


df.head()


# In[144]:


# Need to work on
# salary parsing
# company name text only
# extract state from location
# age of a company
# parsing of job desc


# In[145]:


#remove rows where salary estimate is null
df=df[df['Salary Estimate']!='-1'] 


# In[146]:


# taking out 'glassdoor est.' text from salary estimate column
salary=df['Salary Estimate'].apply(lambda x:x.split('(')[0])
salary


# In[147]:


#Replacing K's and $'s in salary estimate with blank
minus_Kd= salary.apply(lambda x: x.replace('K', '').replace('$',''))
minus_Kd


# In[148]:


#There are 2 kinds of salary estimates- employer provided, hourly


# In[149]:


df['hourly']= df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)


# In[150]:


df['employer_provided']= df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)


# In[151]:


# Replacing 'per hour', 'employer provided salary:' with blank
min_hr=minus_Kd.apply(lambda x: x.lower().replace('per hour', ''). replace('employer provided salary:', ''))
min_hr


# In[152]:


# Splitting min_hr on '-' to get min, max salary
df['min_salary']=min_hr.apply(lambda x: x.split('-')[0])
#df['min_salary'].astype(str).astype(int)
df["min_salary"] = pd.to_numeric(df["min_salary"])


# In[154]:


df['max_salary']=min_hr.apply(lambda x: x.split('-')[1])
#df['max_salary'].astype(str).astype(int)
df["max_salary"] = pd.to_numeric(df["max_salary"])


# In[155]:


df['avg_salary']=(df.min_salary+df.max_salary)/2


# In[157]:


# remove 4 chars from the end of each record in Company Name to remove rating and ''\n'
df['company_txt']=df.apply(lambda x: x['Company Name'] if x['Rating']<0 else  x['Company Name'][:-4], axis=1)


# In[159]:


# Seperating State from Job Location
df['job_state']=df['Location'].apply(lambda x:x.split(',')[1])


# In[94]:


df.job_state.value_counts()


# In[161]:


# Is job location same as headquarters?
df['same_state']=df.apply(lambda x : 1 if x.Location==x.Headquarters else 0, axis=1)


# In[163]:


# If founded = is not given then leave it as it if not replace with (current year - year founded )
df['age']=df.Founded.apply(lambda x: x if x<0 else 2021-x )


# In[164]:


#parsing job desc
df['Job Description'][0]


# In[ ]:


# Look for words such as python, r studio, spark, aws, excel


# In[165]:


df['python_yn']=df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)


# In[166]:


df['spark']=df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() or 'apache spark' in x.lower() else 0)


# In[167]:


df.spark.value_counts()


# In[168]:


df['aws']=df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)


# In[169]:


df.aws.value_counts()


# In[170]:


df['excel']=df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)


# In[171]:


df.excel.value_counts()


# In[172]:


df.columns


# In[120]:


# Drop unnecessary columns
df_out= df.drop(['Unnamed: 0'], axis=1)


# In[122]:


df_out.to_csv('salary_cleaned_data.csv', index=False)


# In[ ]:




