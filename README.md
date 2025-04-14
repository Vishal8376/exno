# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
```
import pandas as pd
df=pd.read_csv("/content/SAMPLEIDS (1).csv")
df
```
![alt text](<Screenshot 2025-04-15 030948.png>)
```
df.info()
```
![alt text](<Screenshot 2025-04-15 031056.png>)

```
df.describe()
```
![alt text](<Screenshot 2025-04-15 031222.png>)
```
df.shape #(rows,columns)
```
![alt text](<Screenshot 2025-04-15 031318.png>)
```
df.isnull()
```
![alt text](<Screenshot 2025-04-15 031359.png>)
```
df.notnull()
```
![alt text](<Screenshot 2025-04-15 031436.png>)
```
df.dropna(axis=0) #deletes the row if it has a null value
```
![alt text](<Screenshot 2025-04-15 031516.png>)
```
df.dropna(axis=1) #deletes the column if it has a nul value
```
![alt text](<Screenshot 2025-04-15 031558.png>)
```
dfs=df[df['TOTAL']>270]
dfs
```
![alt text](<Screenshot 2025-04-15 031636.png>)

```
dfs=df[df['NAME'].str.startswith(('A','C'))&(df['TOTAL']>250)]
dfs
```
![alt text](<Screenshot 2025-04-15 031725.png>)
```
df.iloc[:4]
```
![alt text](<Screenshot 2025-04-15 031725-1.png>)
```
df.iloc[0:4,1:4]
```
![alt text](<Screenshot 2025-04-15 031935.png>)
```
df.iloc[[1,3,5],[1,3]]
```
![alt text](<Screenshot 2025-04-15 032000.png>)
```
dff=df.fillna(0)
dff
```
![alt text](<Screenshot 2025-04-15 032048.png>)
```
df['TOTAL'].fillna(value=df['TOTAL'].mean(),inplace=True)
df
```
![alt text](<Screenshot 2025-04-15 032126.png>)
```
df.fillna(method='ffill')
```
![alt text](<Screenshot 2025-04-15 032202.png>)
```
df.fillna(method='bfill')
```
![alt text](<Screenshot 2025-04-15 032239.png>)

```
df['TOTAL'].fillna(value=df['TOTAL'].mean(),inplace=True)
df
```
![alt text](<Screenshot 2025-04-15 032349.png>)
```
import seaborn as sns
sns.heatmap(df.isnull(),yticklabels=False,annot=True)
```
![alt text](download.png)
```
df.dropna(inplace=True)
sns.heatmap(df.isnull(),yticklabels=False,annot=True)
```
![alt text](<download (1).png>)
```
page=[1,3,28,27,25,92,30,39,40,50,26,24,29,94]
af=pd.DataFrame(age)
af
```
![alt text](<Screenshot 2025-04-15 032604.png>) ![alt text](<Screenshot 2025-04-15 032349-1.png>)
```
sns.boxplot(data=af)
```
![alt text](<download (2).png>)
```
sns.scatterplot(data=af)
```
![alt text](<download (3).png>)
```
q1=af.quantile(0.25)
q2=af.quantile(0.50)
q3=af.quantile(0.75)
iqr=q3-q1
iqr
```
![alt text](<Screenshot 2025-04-15 032805.png>)
```
import numpy as np
Q1=np.percentile(af,25)
Q3=np.percentile(af,75)
IQR=Q3-Q1
IQR
```
![alt text](<Screenshot 2025-04-15 032836.png>)
```
lower_bound=Q1-1.5*IQR
lower_bound
```
![alt text](<Screenshot 2025-04-15 032946.png>)
```
upper_bound=Q3+1.5*IQR
upper_bound
```
![alt text](<Screenshot 2025-04-15 033036.png>)
```
outliers=[x for x in age if x<lower_bound or x>upper_bound]
print("Q1:",Q1)
print("Q3:",Q3)
print("IQR:",IQR)
print("Lower Bound:",lower_bound)
print("Upper Bound:",upper_bound)
print("Outliers:",outliers)
```
![alt text](<Screenshot 2025-04-15 033113.png>)
```
af=af[((af>=lower_bound)&(af<=upper_bound))]
af
```
![alt text](<Screenshot 2025-04-15 033148.png>)
```
af=af.dropna()
af
```
![alt text](<Screenshot 2025-04-15 033220.png>)
```
sns.boxplot(data=af)
```
![alt text](<download (4).png>)
```
sns.scatterplot(data=af)
```
![alt text](<download (5).png>)
```
data=[1,2,2,2,3,1,1,15,2,2,2,3,1,1,2]
mean=np.mean(data)
std=np.std(data)
print('mean of the dataset is',mean)
print('std.deviation is',std)
```
![alt text](<Screenshot 2025-04-15 033425.png>)

```
threshold=3
outlier=[]
for i in data:
  z=(i-mean)/std
  if z>threshold:
    outlier.append(i)
print('outlier in dataset is',outlier)
```
![alt text](<Screenshot 2025-04-15 033455.png>)
```
from scipy import stats
data={'weight':[12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,
66,69,202,72,75,78,81,84,232,87,90,93,96,99,258]}
wf=pd.DataFrame(data)
wf
```
![alt text](<Screenshot 2025-04-15 033933.png>)
```
z=np.abs(stats.zscore(wf))
print(wf[z['weight']>3])
```
![alt text](<Screenshot 2025-04-15 033647.png>)
# Result
          <<include your Result here>>
          