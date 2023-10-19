# ODD2023 Datascience Ex06
# EX-06 FEATURE TRANSFORMATION
### AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

### EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

### ALGORITHM
### Step1:
Read the given Data.
### Step2:
Clean the Data Set using Data Cleaning Process.
### Step3:
Apply Feature Transformation techniques to all the features of the data set.
### Step4:
Print the transformed features.
### PROGRAM:
```
DEVELOPED BY: DIVYA K
REGISTER NUMBER:212222230035
```
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
df.skew
np.log(df["Highly Positive Skew"])
np.reciprocal(df["Moderate Positive Skew"])
np.sqrt(df["Highly Positive Skew"])
np.square(df["Highly Positive Skew"])
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
df["Moderate Positive Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
sm.qqplot(df["Moderate Negative Skew_1"],line='45')
plt.show()
df["Highly Negative skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
sm.qqplot(df["Highly Negative skew_1"],line='45')
plt.show()
```
### OUTPUT:
![275131699-c87c186e-aa56-4ada-bd0b-f206e28da12c](https://github.com/divyakumars/ODD2023-Datascience-Ex06/assets/119393621/72cbf8f5-2bdc-4dc0-8076-d52d0b074b30)


![275131606-79e6bbba-c80b-4e4f-936e-90548808b4fe](https://github.com/divyakumars/ODD2023-Datascience-Ex06/assets/119393621/1bc20687-145a-4af1-934c-cb282ce9c20f)

![275131957-981ffa46-f0e2-4064-8300-cabc49f1fe29](https://github.com/divyakumars/ODD2023-Datascience-Ex06/assets/119393621/70f4c989-88c6-497d-a33e-0d170410f589)


![275132230-e13a50a8-3720-4fbb-ad5a-f53e8f073f62](https://github.com/divyakumars/ODD2023-Datascience-Ex06/assets/119393621/07f93f12-4ef7-4952-8d59-1d2615c64aa2)


![275132256-82d25927-e0a0-40da-a568-29fcf65fb1cd](https://github.com/divyakumars/ODD2023-Datascience-Ex06/assets/119393621/af492e91-24e7-4855-a471-59affbb3f7db)

![275132632-974244ba-bf85-49a2-9c5a-9efc6917f089](https://github.com/divyakumars/ODD2023-Datascience-Ex06/assets/119393621/b98f3c29-8cd4-4003-af75-f9e38a18e631)


![275132640-6f5c1e47-de3d-488c-a0a9-e6a3e318cec5](https://github.com/divyakumars/ODD2023-Datascience-Ex06/assets/119393621/27d123b0-022c-426e-94ba-d26e223d4aad)

![275132642-221b51d9-69c6-417c-8eda-586a225fa011](https://github.com/divyakumars/ODD2023-Datascience-Ex06/assets/119393621/07807b18-9f78-434b-8fdc-8febe39b878f)

![275132645-7555347c-e9d6-45b9-ba5e-ecfc517a6d6a](https://github.com/divyakumars/ODD2023-Datascience-Ex06/assets/119393621/24dcf252-8bf2-4f3f-933a-57732dba8382)




![275132651-218fa41e-8ecd-41a0-b2e5-077e02563ab5](https://github.com/divyakumars/ODD2023-Datascience-Ex06/assets/119393621/3af62436-8e9f-4b4c-8db5-642bcd37b151)



![275132654-ea744e15-ca1e-44a4-bad6-97f3cc44ed55](https://github.com/divyakumars/ODD2023-Datascience-Ex06/assets/119393621/4954f046-8dba-48c0-a769-8b3105c8576b)


![275132659-9140db78-3b28-4aa2-b4cf-7aa331cc4089](https://github.com/divyakumars/ODD2023-Datascience-Ex06/assets/119393621/14bb1e8a-b91a-4378-a858-fe1a216be509)


![275132663-36bc734c-ca61-4e86-bd30-20915f1fd7b4](https://github.com/divyakumars/ODD2023-Datascience-Ex06/assets/119393621/8d8a5a12-a127-4e75-b6f0-d0350321d19d)






























### RESULT:
Thus feature transformation is done for the given dataset.

































