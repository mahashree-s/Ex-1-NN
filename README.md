<H3> NAME  :MAHASHREE S </H3>
<H3> REGISTER NO: 212225230163</H3> 
<H3>EX. NO.1</H3>
<H3>DATE: 27-10-2027 </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("movie_ratings.csv")
print(df.head())

print("\n2. Dataset Information")
print(df.info())

print("\n3. Missing Values Before Cleaning")
print(df.isnull().sum())
df.dropna(inplace=True)
print("Missing values removed.")

print("\n4. Duplicate Values Before Cleaning")
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicate rows removed.")

print("\n5. Removing Unnecessary Features")
df.drop(columns=['unnecessary_column'], inplace=True, errors='ignore')

print("\n5. Standardization using StandardScaler")
cols = df.select_dtypes(include="number").columns
scaler = StandardScaler()
df[cols] = scaler.fit_transform(df[cols])
print(df.head())

print(df.shape)


print("Splitting X and y")

X = df.drop("Rating", axis=1)
y = df["Rating"]

print("\nInput Features (X):")
print(X.head())

print("\nTarget Variable (y):")
print(y.head())

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)



from sklearn.model_selection import train_test_split

print("Train-Test Split")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

```
## OUTPUT:
<img width="805" height="811" alt="image" src="https://github.com/user-attachments/assets/3db9627d-bcd1-4c2c-8567-ba02033c3f3c" />

<img width="849" height="566" alt="image" src="https://github.com/user-attachments/assets/1b675e90-73d4-4b32-aa40-f61556aa63b5" />

<img width="859" height="578" alt="image" src="https://github.com/user-attachments/assets/f156bd2d-410c-4804-878f-8cd542e4f54c" />

<img width="866" height="355" alt="image" src="https://github.com/user-attachments/assets/71f2ca0a-e7c3-4c7c-801f-d265f9c752a8" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


