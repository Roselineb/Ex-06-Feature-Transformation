# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
## STEP 1
Read the given Data

## STEP 2
Clean the Data Set using Data Cleaning Process

## STEP 3
Apply Feature Transformation techniques to all the features of the data set

## STEP 4
Save the data to the file

# CODE
```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.head()

df.isnull().sum()

df.info()

df.describe()

df1 = df.copy()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

```
# OUPUT
## Dataset:
# ![image](https://user-images.githubusercontent.com/128909895/232391650-61436757-4a49-4a33-b976-41acc97b1345.png)
## Head:
# ![image](https://user-images.githubusercontent.com/128909895/232391958-75c71234-552b-4973-963e-914a80730ddd.png)
## Null data:
# ![image](https://user-images.githubusercontent.com/128909895/232392505-287b00fc-8718-42bb-8df8-e8be6d30ae1e.png)
## Information:
# ![image](https://user-images.githubusercontent.com/128909895/232392780-651fa4bb-23b5-4a3a-89f0-94ca42460eda.png)
## Description:
# ![image](https://user-images.githubusercontent.com/128909895/232392856-f800c697-6685-47a0-9312-819ed850a641.png)
## Highly Positive Skew:
# ![image](https://user-images.githubusercontent.com/128909895/232394035-60dd203d-d742-428a-bf34-1e249acfbac5.png)
## Highly Negative Skew:
# ![image](https://user-images.githubusercontent.com/128909895/232394285-3cd50a7f-2b52-445b-b9ed-08f960d4f769.png)
## Moderate Positive Skew:
# ![image](https://user-images.githubusercontent.com/128909895/232394518-46bef34e-561f-4d91-8632-23f114dce33c.png)
## Moderate Negative Skew:
# ![image](https://user-images.githubusercontent.com/128909895/232394642-c576445a-8416-4e2e-b95c-da0fa9b5571c.png)
## Log of Highly Positive Skew:
# ![image](https://user-images.githubusercontent.com/128909895/232394805-dbd768b7-9151-4890-a33f-9bb9a805b1ce.png)
## Log of Moderate Positive Skew:
# ![image](https://user-images.githubusercontent.com/128909895/232395019-f67218a1-ada7-410c-9397-67c3147838bc.png)
## Square root tranformation:
# ![image](https://user-images.githubusercontent.com/128909895/232395276-76bd7391-4976-48f1-8f0a-34cfca4e7232.png)
## Power transformation of Moderate Positive Skew:
# ![image](https://user-images.githubusercontent.com/128909895/232395433-be040a19-a5d7-4d4d-afee-84488bb27a25.png)
## Power transformation of Moderate Negative Skew:
# ![image](https://user-images.githubusercontent.com/128909895/232395765-81169396-a36d-43d1-970a-c6f920b8581a.png)
## Quantile transformation:
# ![image](https://user-images.githubusercontent.com/128909895/232395916-62a0602c-2a0e-4f7c-acf3-2167b646e63d.png)
# Result:
Thus, Feature transformation is performed and executed successfully for the given dataset
