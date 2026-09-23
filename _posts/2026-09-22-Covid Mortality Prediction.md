---
title: Predicting Mortality Risk in COVID-19 Patients for Early Intervention using Decsion Tree Classification Models
date: 2026-09-22 12:00:00 -0700 
categories: [Data Analysis, Python, Prediction, Data Science] 
tags: [python, pandas, SciKitLearn, MatPlotLib, Visualiztion, Machine Learning, Desicion Trees, Data Science]
math: true
---

### Goal
The objective of this project is to clean raw medical data, perform a breif data analysis, and finally utilize a decision tree classifier model to predict patient risk of mortality. 

This project uses the "COVID-19 Dataset" from Kaggle. (https://www.kaggle.com/datasets/meirnizri/covid19-dataset)  
- *"This dataset was provided by the Mexican government https://datos.gob.mx/busca/dataset/informacion-referente-a-casos-covid-19-en-mexico This dataset contains anonymized patient-related information including pre-conditions. The raw dataset consists of 21 unique features and 1,048,576 unique patients. In the Boolean features, **1 means "yes" and 2 means "no". Values of 97 and 99 are missing data.***

Below are the features in the dataset 

**Binary Features**  *(Yes/No)* -    
    - **Pneumonia**  
    - **Pregnant**  
    - **Diabetes**  
    - **COPD**  
    - **Asthma**  
    - **Immunosuppressed**  
    - **Hypertension**  
    - **Cardiovascular Issues**  
    - **Chronic Renal Issues**  
    - **Other diseases**  
    - **Obesity**  
    - **Tobacco User**  
    - **Intubated** 
    - **Sex** 

### Dataset Review
Variable Features -  
- **Age**  
- **Classification -** Covid test findings. Values 1-3 mean that the patient was diagnosed with covid in different degrees. 4 or higher means that the patient is not a carrier of covid or that the test is inconclusive.  
- **Patient type -** Type of care the patient received in the unit. 1 for returned home and 2 for hospitalization.  
- **Usmr -** Indicates whether the patient treated medical units of the first, second or third level. 
- **Medical unit -** Type of institution of the National Health System that provided the care.  
- **Icu -** Indicates whether the patient had been admitted to an Intensive Care Unit.  
- **Date died -** If the patient died, column indicates the date of death, or 9999-99-99 otherwise.  


<details markdown="1">
<summary>Show code</summary>

```python
#import libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

</details>

from imblearn.under_sampling import RandomUnderSampler

```

### Loading and Exploring the Data

Lets take a brief look at a small snippet of the data below.


```python
#read data
df = pd.read_csv('Covid Data.csv')

#Display first 5 rows
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>USMER</th>
      <th>MEDICAL_UNIT</th>
      <th>SEX</th>
      <th>PATIENT_TYPE</th>
      <th>DATE_DIED</th>
      <th>INTUBED</th>
      <th>PNEUMONIA</th>
      <th>AGE</th>
      <th>PREGNANT</th>
      <th>DIABETES</th>
      <th>...</th>
      <th>ASTHMA</th>
      <th>INMSUPR</th>
      <th>HIPERTENSION</th>
      <th>OTHER_DISEASE</th>
      <th>CARDIOVASCULAR</th>
      <th>OBESITY</th>
      <th>RENAL_CHRONIC</th>
      <th>TOBACCO</th>
      <th>CLASIFFICATION_FINAL</th>
      <th>ICU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>03/05/2020</td>
      <td>97</td>
      <td>1</td>
      <td>65</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>97</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>03/06/2020</td>
      <td>97</td>
      <td>1</td>
      <td>72</td>
      <td>97</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>97</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>09/06/2020</td>
      <td>1</td>
      <td>2</td>
      <td>55</td>
      <td>97</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>12/06/2020</td>
      <td>97</td>
      <td>2</td>
      <td>53</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>97</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>21/06/2020</td>
      <td>97</td>
      <td>2</td>
      <td>68</td>
      <td>97</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>97</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



Nothing stands out initially above, however many of the data points are simply numerical and binary in nature.

Let's check for full count of records, data types provided, and confirming there are no NULL values.


```python
#checking data types within dataset

df.info()
```

    <class 'pandas.DataFrame'>
    RangeIndex: 1048575 entries, 0 to 1048574
    Data columns (total 21 columns):
     #   Column                Non-Null Count    Dtype
    ---  ------                --------------    -----
     0   USMER                 1048575 non-null  int64
     1   MEDICAL_UNIT          1048575 non-null  int64
     2   SEX                   1048575 non-null  int64
     3   PATIENT_TYPE          1048575 non-null  int64
     4   DATE_DIED             1048575 non-null  str  
     5   INTUBED               1048575 non-null  int64
     6   PNEUMONIA             1048575 non-null  int64
     7   AGE                   1048575 non-null  int64
     8   PREGNANT              1048575 non-null  int64
     9   DIABETES              1048575 non-null  int64
     10  COPD                  1048575 non-null  int64
     11  ASTHMA                1048575 non-null  int64
     12  INMSUPR               1048575 non-null  int64
     13  HIPERTENSION          1048575 non-null  int64
     14  OTHER_DISEASE         1048575 non-null  int64
     15  CARDIOVASCULAR        1048575 non-null  int64
     16  OBESITY               1048575 non-null  int64
     17  RENAL_CHRONIC         1048575 non-null  int64
     18  TOBACCO               1048575 non-null  int64
     19  CLASIFFICATION_FINAL  1048575 non-null  int64
     20  ICU                   1048575 non-null  int64
    dtypes: int64(20), str(1)
    memory usage: 178.0 MB
    

Quick check for NA values in the file.


```python
#Checking for NA/Null values

df.isna().sum()
```




    USMER                   0
    MEDICAL_UNIT            0
    SEX                     0
    PATIENT_TYPE            0
    DATE_DIED               0
    INTUBED                 0
    PNEUMONIA               0
    AGE                     0
    PREGNANT                0
    DIABETES                0
    COPD                    0
    ASTHMA                  0
    INMSUPR                 0
    HIPERTENSION            0
    OTHER_DISEASE           0
    CARDIOVASCULAR          0
    OBESITY                 0
    RENAL_CHRONIC           0
    TOBACCO                 0
    CLASIFFICATION_FINAL    0
    ICU                     0
    dtype: int64



We have no 'Null' or 'NA' values listed explicitly within this data set, but it was expressed they are listed as 97, 98, 99.  

With my working experience on medical data, I can attest that it is widly inconsistent with the information provided. Typically due to worker inconsistency with data placed into data systems. 

For this project, we will make the assumption that a "NA" value *(97, 98, 99)* implies "No". For other types of datasets, it may be best to review imputation methods to supplement data or remove them entirely.  

Lets review our numerical "NA" values below:


```python
#99, 98, and 97 values are technically 'NA' values within the dataset so they cleaned.

df.eq(99).sum() + df.eq(97).sum() + df.eq(98).sum()
```




    USMER                        0
    MEDICAL_UNIT                 0
    SEX                          0
    PATIENT_TYPE                 0
    DATE_DIED                    0
    INTUBED                 855869
    PNEUMONIA                16003
    AGE                        345
    PREGNANT                527265
    DIABETES                  3338
    COPD                      3003
    ASTHMA                    2979
    INMSUPR                   3404
    HIPERTENSION              3104
    OTHER_DISEASE             5045
    CARDIOVASCULAR            3076
    OBESITY                   3032
    RENAL_CHRONIC             3006
    TOBACCO                   3220
    CLASIFFICATION_FINAL         0
    ICU                     856032
    dtype: int64



### Further data cleaning

- Removing any 99, 98, 97 values and replacing them with 0  

- Adjusting our binary values for more traditional 1 (Yes) & 0 (No) values

- Adjustments to Sex variables:
1 = Female, 2 = Male  **->**  1 = Male, 0 = Female

- "PATIENT_TYPE" column name has been adjusted to 'HOSPITALIZED' as it signifies if a patient was sent home or given a room at the hospital

Quick output to double check final counts of 97,98,99 below.

<details markdown="1">
<summary>Show code</summary>

```python
# Establish Binary Columns
binary_columns = ['INTUBED','PREGNANT','ICU','PATIENT_TYPE',
       'PNEUMONIA', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
       'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
       'RENAL_CHRONIC', 'TOBACCO',]

na_values = [97,98,99]
df[binary_columns] = df[binary_columns].replace(na_values,0)

binary_columns_2 = ['INTUBED','PREGNANT','ICU',
       'PNEUMONIA', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
       'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
       'RENAL_CHRONIC', 'TOBACCO',]

df[binary_columns_2] = df[binary_columns_2].replace(2,0)

df['SEX'] = df['SEX'].replace(1,0)
df['SEX'] = df['SEX'].replace(2,1)

df['PATIENT_TYPE'] = df['PATIENT_TYPE'].replace(1,0)
df['PATIENT_TYPE'] = df['PATIENT_TYPE'].replace(2,1)
df = df.rename(columns = {'PATIENT_TYPE':"HOSPITALIZED"})

class_final = [4,5,6,7]
df['CLASIFFICATION_FINAL'] = df['CLASIFFICATION_FINAL'].replace(class_final,0)


df.eq(99).sum() + df.eq(97).sum() + df.eq(98).sum()
```
</details>





    USMER                     0
    MEDICAL_UNIT              0
    SEX                       0
    HOSPITALIZED              0
    DATE_DIED                 0
    INTUBED                   0
    PNEUMONIA                 0
    AGE                     345
    PREGNANT                  0
    DIABETES                  0
    COPD                      0
    ASTHMA                    0
    INMSUPR                   0
    HIPERTENSION              0
    OTHER_DISEASE             0
    CARDIOVASCULAR            0
    OBESITY                   0
    RENAL_CHRONIC             0
    TOBACCO                   0
    CLASIFFICATION_FINAL      0
    ICU                       0
    dtype: int64



### Adding targets for Model

Below, I appended a target feature in the dataframe, signifying if a patient died or not based the value within the 'DATE_DIED' column.  

 **0 for NO** and **1 for YES**

<details markdown="1">
<summary>Show code</summary>

```python
#populating DIED column for patient survived (0) or patient is deceased (1)

df["DIED"] = [0 if each=="9999-99-99" else 1 for each in df.DATE_DIED]
df.DIED.value_counts()
```
</details>




    DIED
    0    971633
    1     76942
    Name: count, dtype: int64



Quick sanity check with a ".head()" and ".columns" output.


```python
#Checking values were replaced correctly

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>USMER</th>
      <th>MEDICAL_UNIT</th>
      <th>SEX</th>
      <th>HOSPITALIZED</th>
      <th>DATE_DIED</th>
      <th>INTUBED</th>
      <th>PNEUMONIA</th>
      <th>AGE</th>
      <th>PREGNANT</th>
      <th>DIABETES</th>
      <th>...</th>
      <th>INMSUPR</th>
      <th>HIPERTENSION</th>
      <th>OTHER_DISEASE</th>
      <th>CARDIOVASCULAR</th>
      <th>OBESITY</th>
      <th>RENAL_CHRONIC</th>
      <th>TOBACCO</th>
      <th>CLASIFFICATION_FINAL</th>
      <th>ICU</th>
      <th>DIED</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>03/05/2020</td>
      <td>0</td>
      <td>1</td>
      <td>65</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>03/06/2020</td>
      <td>0</td>
      <td>1</td>
      <td>72</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>09/06/2020</td>
      <td>1</td>
      <td>0</td>
      <td>55</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>12/06/2020</td>
      <td>0</td>
      <td>0</td>
      <td>53</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>21/06/2020</td>
      <td>0</td>
      <td>0</td>
      <td>68</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
df = df.drop(columns = ["DATE_DIED"])

df.columns
```




    Index(['USMER', 'MEDICAL_UNIT', 'SEX', 'HOSPITALIZED', 'INTUBED', 'PNEUMONIA',
           'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
           'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
           'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL', 'ICU', 'DIED'],
          dtype='str')



# Data Visualization  

### Mortality Rate by Condition

Our first graph compares the mortality rate by each condition. Some features in the data have been dropped for these graphs as they are not patient condtions, they are listed below:  
- **USMER** -                   *Level of treatment at hospital*  
- **MEDICAL UNIT** -           *Level of hospital*  
- **HOSPITALIZED** -           *Was patient admitted to hospital*  
- **INTUBED** -                *Was patient intubated*  
- **PNEUMONIA** -              *Patient has Pneumonia, something ***caused*** by covid*  
- **CLASIFFICATION_FINAL** -   *Degrees of Covid-19 diagnosis*  
- **ICU** -                    *Was patient placed into intenxive care unit*
- **DIED** -                   *flag if patient died*   


While our proportion of individuals who did not survive Covid is low overall in this dataset, we can see that the top conditions with the highest rate of mortality are:  
- **Chronic Renal Disease**, **Chronic Obstructive Pulmonary Disease**, **Diabetes**, & **Cardiovascular Issues**.  

In simple terms, heart, lung, diabetes, and kidney issues have the highest mortality rates with regards to COVID-19 patients.

<details markdown="1">
<summary>Show code</summary>

```python
#List of features to avoid graphing created as insights were minimal with these features

graph_col = [ 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
       'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
       'RENAL_CHRONIC', 'TOBACCO']

# Plot with for loop going through all features
mortality_rates = pd.DataFrame({
    'Has Condition': [df[df[c] == 1]['DIED'].mean() for c in graph_col],
    'Does Not Have Condition': [df[df[c] == 0]['DIED'].mean() for c in graph_col]
}, index=graph_col)

mortality_rates = mortality_rates.sort_values('Has Condition', ascending=False)

# Plot
ax = mortality_rates.plot(kind='barh', figsize=(10, 6), color=["#bb4444", "#4082b1"])
plt.xlabel('Mortality Rate')
plt.title('Mortality rate by conditions: With vs. Without Condition')
plt.legend(title='')
plt.gca().invert_yaxis()  # highest risk at top
plt.tight_layout()
plt.show()
```
</details>

    
![png](assets/Covid_pics/CovidMortalityPrediction_18_0.png)
    


### Stacked Bar Chart Comparing Moratlity Poportion for Multiple Conditions

Looking at the graph below, we can see the more conditions a patient has, the higher the rate of mortality. Simply put, the more conditions a patient has the higher chance they of not surviving.  

The graph has been created using the following features:  
 - 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR','HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO'

<details markdown="1">
<summary>Show code</summary>

```python
# Creating copy of df for graph
co_df = df.copy()

# Sum of comorbidities for each patient

co_df['comorbidity_count'] = co_df[graph_col].sum(axis=1)

# Cap at 3 or more comorbities
co_df['comorbidity_group'] = co_df['comorbidity_count'].apply(lambda x: '3 or more' if x >= 3 else str(x))

# Cross-tab: comorbidity group vs. outcome
comorbidity_outcome = pd.crosstab(co_df['comorbidity_group'], co_df['DIED'])
comorbidity_outcome.columns = ['Survived', 'Did Not Survive']

comorbidity_pct = comorbidity_outcome.div(comorbidity_outcome.sum(axis=1), axis=0)

ax = comorbidity_pct.plot(kind='bar', stacked=True, figsize=(9, 6), color=["#5faa5f", "#c44949"])
plt.xlabel('Number of Conditions')
plt.ylabel('Proportion of Patients')
plt.title('Mortality Proportion by Number of Conditions')
plt.xticks(rotation=0)
plt.legend(title='Outcome', bbox_to_anchor=(1.02, 1), loc='upper left')

# Adding percentage labels to each segment
for container in ax.containers:
    labels = [f'{v.get_height():.1%}' if v.get_height() > 0.02 else '' for v in container]
    ax.bar_label(container, labels=labels, label_type='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()
```
</details>

    
![png](assets/Covid_pics/CovidMortalityPrediction_20_0.png)
    


### Distribution of Age Grouped by Mortality.

One aspect that was discussed heavily during the height of the 2020 Covd-19 pandemic was age being a risk factor for mortality.   

Looking at the graph below, we can see the distribution of Age grouped by if the patient survived or not.  

Comparatively, we can see that the count of fatalies is significantly lower, however the instances of fatality are grouped in a higher age range, with the median age being **62** *(Listed below graph)*

<details markdown="1">
<summary>Show code</summary>

```python
df['Outcome'] = df['DIED'].map({0: 'Survived', 1: 'Deceased'})

sns.histplot(data=df, x="AGE", kde=True, hue='Outcome', palette='bright')
plt.title('Age Distribution by Outcome')
plt.xlabel('Age')
plt.show()

median_age = df.groupby('Outcome')['AGE'].median()
print(median_age)
```
</details>

    
![png](assets/Covid_pics/CovidMortalityPrediction_22_0.png)
    


    Outcome
    Deceased    62.0
    Survived    39.0
    Name: AGE, dtype: float64
    

### Correlation Matrix

Below is correlation matrix between all conditions in the data set. The strongest correlations with out "DIED" target are:
- **AGE** *(.32)*
- **DIABETES** *(.22)*
- **HYPERTENSION** *(.20)*

We should keep these in mind as we create our model and see if it lists these features are the most important.  


On a side note, it is interesting to see the correlations of:
- **HYPERTENSION vs. AGE** 
- **HYPERTENSION vs. DIABETES**
- **DIABETES vs. AGE**

While we are not explicitly looking for these, its interesting to see these appear in the dataset while reviewing patient conditions. 

<details markdown="1">
<summary>Show code</summary>

```python
# DOUBLLE CHECK WITH PRE-CONDITIONS ONLY

pre_cond_col = ['SEX',
       'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
       'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
       'RENAL_CHRONIC', 'TOBACCO', 'DIED']

df = df[pre_cond_col]

corr = df.corr()
matrix = np.triu(corr)

plt.figure(figsize = (10,10))
sns.heatmap(df.corr(),mask = matrix, annot = True, fmt = '.2f')
```

</details>


    <Axes: >




    
![png](assets/Covid_pics/CovidMortalityPrediction_24_1.png)
    


### Decision Tree Model

Utilizing our SciKitLearn library, I have split our data into training and testing data, defined the decsision tree below, and ran our model with no specific parameters.  

Our initial results appear good, but we need to explore the results on a deeper level. Accuracy alone is not always a reliable measurement, as we will soon see.


```python
# Dividing data into seperate features and target dataframes

X = df.drop(columns="DIED")
y = df["DIED"]


```
<details markdown="1">
<summary>Show code</summary>

```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .25, random_state = 42)
dtree = DecisionTreeClassifier(random_state=42)


print(y_test.value_counts())

dtree.fit(X_train, y_train)
yhat = dtree.predict(X_test)

print(yhat.sum())

print('Training Accuracy: ' +str(dtree.score(X_train,y_train)))
print('Testing Accuracy: ' +str(accuracy_score(yhat, y_test)))
```

</details>

    DIED
    0    243130
    1     19014
    Name: count, dtype: int64
    2636
    Training Accuracy: 0.9312717835385431
    Testing Accuracy: 0.9247283935546875
    

A **92%** percent accuracy sounds great on paper, but let's dive deeper.

<details markdown="1">
<summary>Show code</summary>

```python
#creating empty lists for dataframe creation in the for loop
depth, treetrain, treetest = [],[],[]
for i in range(1,30):
    
    #define depth each loop 
    dtree = DecisionTreeClassifier(max_depth = i, random_state=0)
    
    #fit model and predict target y
    dtree.fit(X_train, y_train)
    yhat = dtree.predict(X_test)
    
    #append values from each depth
    depth.append(i)
    treetrain.append(dtree.score(X_train,y_train))
    treetest.append(accuracy_score(yhat, y_test))

treedf = pd.DataFrame({'Tree Depth':depth, 'Train Score':treetrain,'Test Score':treetest})
treedf.head()
```
</details>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tree Depth</th>
      <th>Train Score</th>
      <th>Test Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.926341</td>
      <td>0.927467</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.926341</td>
      <td>0.927467</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.926341</td>
      <td>0.927467</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.926341</td>
      <td>0.927467</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.926341</td>
      <td>0.927467</td>
    </tr>
  </tbody>
</table>
</div>



I chose to compute the accuracy score for the train and test data as the depth of the tree increases. In theory, our accuracy should increase as the tree depth increases, reaching a limit where the bias/variance tradeoff is at its apex. It is at this point we want our tree depth to stop and remain.

Looking at the graph below, we can see this pattern emerges. The model begins to fit to the training data too much as the depth increases, while testing data predictions are incorrect.  

One important thing to notice is our range in the graph. We have a total range of only **.006**, the alarm bells should be ringing here as this is remarkably small for an unoptimized and uncalibrated model.

<details markdown="1">
<summary>Show code</summary>

```python
#plot data frame of training VS testing scores

plt.plot("Tree Depth", "Train Score", data = treedf)
plt.plot("Tree Depth", "Test Score", data = treedf)
plt.legend(['Training Accuracy', "Test Accuracy"],loc=2)
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy Score')
plt.title("Accuracy Score VS Tree Depth")
```
</details>




    Text(0.5, 1.0, 'Accuracy Score VS Tree Depth')




    
![png](assets/Covid_pics/CovidMortalityPrediction_31_1.png)
    


Lets use a confusion matrix to review our model further and utilize a classification report for some more stats. Below are the standard measurements for reviewing model predictions on a deeper level besides "Accuracy".

<details markdown="1">
<summary>Show code</summary>

```python
#confusion matrix for desicion tree

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(y_test, yhat, normalize=None)
plt.title('Confusion Matrix for Decision Tree')
plt.show()
#Calculation for TPR and TNR

cm = confusion_matrix(y_test, yhat, labels=[0,1])
TN_tree, FP_tree, FN_tree, TP_tree = cm.ravel()

# print(y_test.value_counts())
# print(pd.Series(yhat).value_counts())
# print(cm)

TPR = TP_tree/(TP_tree+FN_tree)
TNR = TN_tree/(TN_tree+FP_tree)

print('True Positve Rate: '+str(TPR))
print('True Negative Rate: '+str(TNR))

print(classification_report(y_test, yhat))

```
</details>

    
![png](assets/Covid_pics/CovidMortalityPrediction_33_0.png)
    


    True Positve Rate: 0.05043652045860945
    True Negative Rate: 0.9930942294245877
                  precision    recall  f1-score   support
    
               0       0.93      0.99      0.96    243130
               1       0.36      0.05      0.09     19014
    
        accuracy                           0.92    262144
       macro avg       0.65      0.52      0.52    262144
    weighted avg       0.89      0.92      0.90    262144
    
    

The results above show a classic case of a model learning that the majority of our outcomes had the patient survive. This led to the model predicting that the patient survives for all instances due to a heavy skew in class sizes.

We can typically see this in other cases of data regarding fraud or disease classification, the disparity of outcomes is simply too large for the model to learn accurately on its own. We will nee dto intervene and adjust the data to help the model. 

One approach we could take is resampling the data, those methods are as follows:  

**Oversample** - Sample the minority class to balance the classes before training.  
**Undersample** - Sasmple the majority class to match the minority class size.  

Lets perform undersampling on this data. 

<details markdown="1">
<summary>Show code</summary>

```python
#create sampler and fit to training data. 

#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .25, random_state = 42)

sampler = RandomUnderSampler(random_state=0)
X_sampled,y_sampled = sampler.fit_resample(X_train,y_train)


print('Training values counts:')
print(y_sampled.value_counts())

print('Testing values counts:')
print(y_test.value_counts())

```
</details>


    Training values counts:
    DIED
    0    57928
    1    57928
    Name: count, dtype: int64
    Testing values counts:
    DIED
    0    243130
    1     19014
    Name: count, dtype: int64
    

With the sampled data now having an equal distribution, we can review our model performance again.

Below we are simply recreating the model, but this time with:
- Our resampled data
- A grid search with 5 cross fold validations to immediately find the best tree depth
- parameter of "class_weight = 'balanced'" to assist the model's learning

Decision trees try to minimize impurity in the child nodes and impurity calculations using class proportions within each node. These proportions are based on raw sample counts but with **class_weight='balanced'**, those proportions are calculated using the weighted counts instead. This aide in our 

<details markdown="1">
<summary>Show code</summary>

```python
#Reapting steps for training and testing but this time using sampled data. 

parameters = {"max_depth": range(1,20)}
grid_search = GridSearchCV(DecisionTreeClassifier(),parameters, cv=5)

grid_search.fit(X_sampled, y_sampled)

best_depth = grid_search.best_params_

dtree = DecisionTreeClassifier(class_weight= 'balanced',random_state=42,max_depth = best_depth['max_depth'])

dtree.fit(X_sampled, y_sampled)
yhat = dtree.predict(X_test)

print("The best depth for the model: "+ str(best_depth['max_depth']))
print('Training Accuracy: ' +str(dtree.score(X_train,y_train)))
print('Testing Accuracy: ' +str(accuracy_score(yhat, y_test)))
```

</details>

    The best depth for the model: 9
    Training Accuracy: 0.753284140630265
    Testing Accuracy: 0.7512969970703125
    

Our accuracy has dropped to .75 with our optimal depth being at 9.

.75 for a model is not great, but also not terrible. This is something we can expect in a real world model. Lets look further below is a confusion matrix and a classification report showing some better insight than accuracy alone.  

<details markdown="1">
<summary>Show code</summary>

```python
ConfusionMatrixDisplay.from_predictions(y_test, yhat, normalize=None)
plt.title('Confusion Matrix for Decision Tree')
plt.show()
#Calculation for TPR and TNR

cm = confusion_matrix(y_test, yhat, labels=[0,1])
TN_tree, FP_tree, FN_tree, TP_tree = cm.ravel()

TPR = TP_tree/(TP_tree+FN_tree)
TNR = TN_tree/(TN_tree+FP_tree)

print('True Positve Rate: '+str(TPR))
print('True Negative Rate: '+str(TNR))

print(classification_report(y_test, yhat))
```
</details>

    
![png](assets/Covid_pics/CovidMortalityPrediction_39_0.png)
    


    True Positve Rate: 0.8274955296097612
    True Negative Rate: 0.7453378850820549
                  precision    recall  f1-score   support
    
               0       0.98      0.75      0.85    243130
               1       0.20      0.83      0.33     19014
    
        accuracy                           0.75    262144
       macro avg       0.59      0.79      0.59    262144
    weighted avg       0.93      0.75      0.81    262144
    
    

Lets explore the stats above a bit more in depth.  
- **TPR/recall** for deaths *(class 1)* is strong (0.83), which means the model catches 83% of acutal deaths on an imbalanced test set.  
- **Precision** for deaths is **very low** (0.20), which tells 8/10 people the model predicts will die, acutally survive.  
- **F1** score for deaths (0.33) is being heavily affected by the low precision of the previous result mentioned.  
- **Accuracy** is a very misleading metric with this dataset due to the heavy imbalance, it does not give us a clear picture and is not something we want to rely on for model evaluation.  


Below are the most important features the model is looking at.  

We can see that **AGE** has a substantial lead over any other metric. Decision trees calculate feature importance based on the reduction in a criterion (ex: Gini impurity or entropy) used to select split points. Clearly the model views **AGE** as the largest reducer of Gini impurity.

<details markdown="1">
<summary>Show code</summary>

```python
#finding most important predictor for decision tree

column_names = df.iloc[:,0:13].columns.tolist()

importance = pd.DataFrame(dtree.feature_importances_, index = column_names, columns = ['Importance'])
importance.sort_values(by = "Importance", ascending = False)
```
</details>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AGE</th>
      <td>0.801296</td>
    </tr>
    <tr>
      <th>DIABETES</th>
      <td>0.086988</td>
    </tr>
    <tr>
      <th>SEX</th>
      <td>0.042584</td>
    </tr>
    <tr>
      <th>RENAL_CHRONIC</th>
      <td>0.022428</td>
    </tr>
    <tr>
      <th>OBESITY</th>
      <td>0.015625</td>
    </tr>
    <tr>
      <th>INMSUPR</th>
      <td>0.009533</td>
    </tr>
    <tr>
      <th>HIPERTENSION</th>
      <td>0.007548</td>
    </tr>
    <tr>
      <th>OTHER_DISEASE</th>
      <td>0.006983</td>
    </tr>
    <tr>
      <th>COPD</th>
      <td>0.002320</td>
    </tr>
    <tr>
      <th>ASTHMA</th>
      <td>0.001942</td>
    </tr>
    <tr>
      <th>CARDIOVASCULAR</th>
      <td>0.001648</td>
    </tr>
    <tr>
      <th>TOBACCO</th>
      <td>0.001066</td>
    </tr>
    <tr>
      <th>PREGNANT</th>
      <td>0.000040</td>
    </tr>
  </tbody>
</table>
</div>



### Conclusion  

The goal of the project was to identify which patient conditions were strongly associated with COVID-19 mortality and to create a model capable of flagging at risk patients. Exploratory data analysis revealed that diabetes, hypertension and age were the strongest individual predictors of a fatal outcome, according to the correlation matrix. Mortality rates increased as the number of conditions per patient increased. Patients with three or more conditions had a mortality rate of 25%, compared to 3.5% for patients with none.

The final decision tree model created here is a high sensitivty, low specificty screening tool. It's tuned to cast a wide net and flag any possibility of a patient death, even at the cost of high false positives (i.e. poor precision). In the use case of a hospital triage tool, this would be desireable, the cost of missing an at risk patient causing a death is greater than the cost of monitoring a patient closer.  

Inversely, having a model that classfies this many false positives would cause a great strain on resource allocation, as many patients would be classified for being at risk, when they may not entirely be. In my opinion, catching these cases is important as there are real risks to a patients outcome if not correctly classified.  

Initially, the first model learned the "safest" strategy to minimize error was to simply classify every patient as "Survived" due to the large imbalance of classes in the data. This is a very common occurance with skewed data and it requires careful review. Using other metrics is important here as accuracy alone will not give a full picture and can lead to false assumptions on our end.  

Future work could be adjusting the classification threshold to better balance precision and recall for specific use cases, testing ensemble methods (i.e. Random Forest, XGBoost) which may help with capturing interactions between pre-conditions, and validating the model against more recent data to assess the models generalizability.

