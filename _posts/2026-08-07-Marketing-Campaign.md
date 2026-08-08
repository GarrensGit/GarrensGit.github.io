---
title: Marketing Campaign Analysis 
date: 2026-08-07 12:00:00 -0700 
categories: [Data Analysis, Python, Prediction, Data Science] 
tags: [python, pandas, MatPlotLib, Visualiztion, ML, Marketing, KNN, Data Science, Regression, Random Forest]
math: true
---

### Data Overview

For this marketing analysis project, I have used the ["Marketing Campaign"](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign/data) dataset from Kaggle.com, a popular website for crowdsourced datasets. 

Here is some context on the provided columns:

**AcceptedCmp1 ... AcceptedCmp5** - 1 if customer accepted the offer in the 1st - 5th campaign, 0 otherwise  
**Response (target)** - 1 if customer accepted the offer in the last campaign, 0 otherwise  
**Complain** - 1 if customer complained in the last 2 years  
**DtCustomer** - date of customer’s enrolment with the company  

**Education** - customer’s level of education  
**Marital** - customer’s marital status  
**Kidhome** - number of small children in customer’s household  
**Teenhome** - number of teenagers in customer’s household  
**Income** - customer’s yearly household income  

**Mnt...Products** - amount spent on "  " products in the last 2 years  
- Products are: **Fish, Meat, Fruits, Sweet, Wines, Gold**

**NumDealsPurchases** - number of purchases made with discount  
**NumCatalogPurchases** - number of purchases made using catalogue  
**NumStorePurchases** - number of purchases made directly in stores  
**NumWebPurchases** - number of purchases made through company’s web site  

**NumWebVisitsMonth** - number of visits to company’s web site in the last month  
**Recency** - number of days since the last purchase  

### Goals  

For this project, I intend to:  
- Perform an exploratory data analysis for inital thoughts, along with cleaning and manipulating our data.
- Create interpretable visualizations that provide insights into customer data.
- Perform customer segmentation with a KMeans model to group our customers.
- Perform a campaign respose prediction with a RandomForestClassifier model to identify relevant features of customer behaviors.

### Initial Data Exploration and Feature Cleaning

<details markdown="1">
<summary>Show code</summary>

```python
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

```
</details>

```python
df = pd.read_csv("marketing_campaign.csv")
df.info()
```

    <class 'pandas.DataFrame'>
    RangeIndex: 2240 entries, 0 to 2239
    Data columns (total 29 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   ID                   2240 non-null   int64  
     1   Year_Birth           2240 non-null   int64  
     2   Education            2240 non-null   str    
     3   Marital_Status       2240 non-null   str    
     4   Income               2216 non-null   float64
     5   Kidhome              2240 non-null   int64  
     6   Teenhome             2240 non-null   int64  
     7   Dt_Customer          2240 non-null   str    
     8   Recency              2240 non-null   int64  
     9   MntWines             2240 non-null   int64  
     10  MntFruits            2240 non-null   int64  
     11  MntMeatProducts      2240 non-null   int64  
     12  MntFishProducts      2240 non-null   int64  
     13  MntSweetProducts     2240 non-null   int64  
     14  MntGoldProds         2240 non-null   int64  
     15  NumDealsPurchases    2240 non-null   int64  
     16  NumWebPurchases      2240 non-null   int64  
     17  NumCatalogPurchases  2240 non-null   int64  
     18  NumStorePurchases    2240 non-null   int64  
     19  NumWebVisitsMonth    2240 non-null   int64  
     20  AcceptedCmp3         2240 non-null   int64  
     21  AcceptedCmp4         2240 non-null   int64  
     22  AcceptedCmp5         2240 non-null   int64  
     23  AcceptedCmp1         2240 non-null   int64  
     24  AcceptedCmp2         2240 non-null   int64  
     25  Complain             2240 non-null   int64  
     26  Z_CostContact        2240 non-null   int64  
     27  Z_Revenue            2240 non-null   int64  
     28  Response             2240 non-null   int64  
    dtypes: float64(1), int64(25), str(3)
    memory usage: 507.6 KB
    

### Initial thoughts on the data:
- Data appears to be strongly formatted (i.e. low variance of data types)
- Majority of data is non-null
- Most columns are of "int64" type, likely going to be either counts or binary flags

Lets take a small look below:

### Data Cleaning

```python
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
      <th>ID</th>
      <th>Year_Birth</th>
      <th>Education</th>
      <th>Marital_Status</th>
      <th>Income</th>
      <th>Kidhome</th>
      <th>Teenhome</th>
      <th>Dt_Customer</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>...</th>
      <th>NumWebVisitsMonth</th>
      <th>AcceptedCmp3</th>
      <th>AcceptedCmp4</th>
      <th>AcceptedCmp5</th>
      <th>AcceptedCmp1</th>
      <th>AcceptedCmp2</th>
      <th>Complain</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5524</td>
      <td>1957</td>
      <td>Graduation</td>
      <td>Single</td>
      <td>58138.00</td>
      <td>0</td>
      <td>0</td>
      <td>2012-09-04</td>
      <td>58</td>
      <td>635</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2174</td>
      <td>1954</td>
      <td>Graduation</td>
      <td>Single</td>
      <td>46344.00</td>
      <td>1</td>
      <td>1</td>
      <td>2014-03-08</td>
      <td>38</td>
      <td>11</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4141</td>
      <td>1965</td>
      <td>Graduation</td>
      <td>Together</td>
      <td>71613.00</td>
      <td>0</td>
      <td>0</td>
      <td>2013-08-21</td>
      <td>26</td>
      <td>426</td>
      <td>...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6182</td>
      <td>1984</td>
      <td>Graduation</td>
      <td>Together</td>
      <td>26646.00</td>
      <td>1</td>
      <td>0</td>
      <td>2014-02-10</td>
      <td>26</td>
      <td>11</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5324</td>
      <td>1981</td>
      <td>PhD</td>
      <td>Married</td>
      <td>58293.00</td>
      <td>1</td>
      <td>0</td>
      <td>2014-01-19</td>
      <td>94</td>
      <td>173</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>



Checking the value counts for **Education** and **Marital_Status** columns to see if there are any inconsistencies or unexpected values. Note that these are our only **"str"** columns


```python

print(df['Education'].value_counts())
print(" ")
print(df['Marital_Status'].value_counts())
```

    Education
    Graduation    1127
    PhD            486
    Master         370
    2n Cycle       203
    Basic           54
    Name: count, dtype: int64
     
    Marital_Status
    Married     864
    Together    580
    Single      480
    Divorced    232
    Widow        77
    Alone         3
    Absurd        2
    YOLO          2
    Name: count, dtype: int64
    

We have a few oddities in our **Marital_Status** column, such as **"Absurd"** and **"YOLO"**. I'll group anything that isnt a clear defind relationship status into a "Single" category, and everything else into a "Relationship" category.

Along with this, lets get an **"Age"** column created and a few other columns based on existing info

<details markdown="1">
<summary>Show code</summary>

```python
# Creating a new column for age based on the year of birth and the current year
current_year = dt.datetime.now().year
df["Age"] = current_year - df["Year_Birth"]

# Creating a new column for relationship status based on marital status, simplifying the categories into "Single" and "Relationship"
df["Relationship_Status"] = df["Marital_Status"].apply(lambda x: "Single" if x in ["Single", "Divorced", "Widow", "Alone", "Absurd", "YOLO"] else "Relationship")

# Creating a new column for total amount spent on products
df["Total_Spent"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]

# Total purchases made by the customer
df["Total_Purchases"] = df["NumDealsPurchases"] + df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]

# Creating new column for has children or not
df["Has_Children"] = np.where((df["Kidhome"] > 1) | (df["Teenhome"] > 1), 1, 0)
```
</details>

I'm separating 4 columns to create a small subset for analysis. The columns are: "Age", "Total_Spent", "Income","Total_Purchases".

Lets review these for any signifcant outliers, remember:
- Any value greater (lower) 1.5 times the IQR above (below) the third quartile (the first quartile) is defined as a __mild outlier__  
- Any value greater (lower) 3 times the IQR above (below) the third quartile (the first quartile) is defined as an __extreme outlier__

Where IQR is defined as:

$$IQR= Q_3-Q_1$$


```python
stat_set = df[["Age", "Total_Spent", "Income","Total_Purchases"]]

pd.options.display.float_format = "{:.2f}".format
stat_set.describe()
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
      <th>Age</th>
      <th>Total_Spent</th>
      <th>Income</th>
      <th>Total_Purchases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2240.00</td>
      <td>2240.00</td>
      <td>2216.00</td>
      <td>2240.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>57.19</td>
      <td>605.80</td>
      <td>52247.25</td>
      <td>14.86</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.98</td>
      <td>602.25</td>
      <td>25173.08</td>
      <td>7.68</td>
    </tr>
    <tr>
      <th>min</th>
      <td>30.00</td>
      <td>5.00</td>
      <td>1730.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>49.00</td>
      <td>68.75</td>
      <td>35303.00</td>
      <td>8.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>56.00</td>
      <td>396.00</td>
      <td>51381.50</td>
      <td>15.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>67.00</td>
      <td>1045.50</td>
      <td>68522.00</td>
      <td>21.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>133.00</td>
      <td>2525.00</td>
      <td>666666.00</td>
      <td>44.00</td>
    </tr>
  </tbody>
</table>
</div>

### Outliers

We can also view outliers using graphical outputs, lets see the distributions of the these columns below

Below, we can see that these outliers greatly skew our dataset, with very small bumps at the tail ends on the right

<details markdown="1">
<summary>Show code</summary>


```python
graph_df = stat_set.melt(var_name="Variable", value_name="Value")

g1 = sns.FacetGrid(graph_df, col="Variable", col_wrap=4, sharex=False, sharey=False, height=4)
g1.map(sns.histplot, "Value", kde=True, bins=30, color="darkblue")

#g1 = sns.FacetGrid(graph_df, col='Variable', col_wrap=4, sharex=False, sharey=False, height=3.5)
#g1.map(sns.boxplot, 'Value', color = 'lightblue', fliersize=3, linewidth=1.5)

g1.set_titles("{col_name}")
g1.set_axis_labels("", "Count")
g1.set_xticklabels(rotation=45,ha = "right")
plt.tight_layout()
plt.show()

```
</details>

    
![png](/assets/Marketing_pics/Marketing_Campaign_15_0.png)
    


Box plots are a great way to view these results as well. The **plotly express** package allows for interactive plots, allowing for a more meaningful result that auidences can interpret on their own.

*P.S. This plot is interactable, hover over data points to see more info!*

<details markdown="1">
<summary>Show code</summary>

```python
graph_df2 = stat_set.melt(var_name='variable', value_name='value')

fig = px.box(
    graph_df2, x="value", facet_col="variable", facet_col_wrap=2,
    color="variable", height=1000, width=900,
    boxmode="overlay",
    color_discrete_sequence=px.colors.qualitative.Set2,
    facet_col_spacing=0.08, facet_row_spacing=0.12
)

fig.update_xaxes(matches=None, showticklabels=True, showline=True, linewidth=2, linecolor='grey')
fig.update_yaxes(showline=True, linewidth=2, linecolor='grey')

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_layout(showlegend=False, title_text="Distribution of Variables", title_x=0.5)

fig.show()
```

</details>

<iframe src="/assets/plotly/BoxPlot.html" width="110%" height="1000" frameborder="0" scrolling="no"></iframe>



We can see the outliers listed above:

- **Age** - We have 3 records listing and age of **126,127,133**. While these do not qualify as extreme outliers, they would be statiscal anamolies as the oldest living person records was **122**. Therfore, I have chosen to drop them, as they are likely no longer be a customer.  

- **Income** - We have a single value of **666,666** listed here. While it is not be impossible to acheive this, it is considered an extreme outlier, and will therfore be dropped.

Removal of outliers for Income and Age columns below.


```python
# Removal of outliers from the dataset based on Age and Income
stat_set.drop(stat_set[stat_set["Age"] > 120].index, inplace=True)
df.drop(df[df["Age"] > 120].index, inplace=True)

stat_set.drop(stat_set[stat_set["Income"] > 500000].index, inplace=True)
df.drop(df[df["Income"] > 500000].index, inplace=True)
```
### Imputation

Let's handle the missing values in the **Income** column

We have many approaches to handling missing values:
- Delete rows with null
- Delete columns with null
- Impute mean value of a given column
- Impute median of a given columns
- Use a model to impute data

Here we will use a KNN model to impute Income based off other key person features. It's important to ensure that features are scaled as this can affect KNN models due to them using distance between points. 

Below, we can see the output for our encoded values printed out in a dict format.

<details markdown="1">
<summary>Show code</summary>

```python
# Creating functions to scale and impute missing values using KNN, as well as label encode and decode categorical columns.

def knn_impute_scaled(dataset, cols, target_cols=None, n_neighbors=5):
    """
    Impute missing values using KNN, with scaling applied first so that
    features on different scales (i.e., Income vs Age) contribute fairly
    to the distance calculation.

    Parameters
    ----------
    dataset : pd.DataFrame
        The full dataset (modified in place for target_cols).
    cols : list of str
        Columns to use as the basis for imputation (features for KNN distance).
    target_cols : list of str, optional
        Columns to actually overwrite with imputed values. Defaults to all of `cols`.
    n_neighbors : int
        Number of neighbors for KNNImputer.

    Returns
    -------
    pd.DataFrame
        The dataset with target_cols imputed.
    """
    target_cols = target_cols or cols
    # Define scaler and apply
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dataset[cols])

    # Define imputer and apply with scaled data
    imputer = KNNImputer(n_neighbors=n_neighbors, metric='nan_euclidean')
    imputed_scaled = imputer.fit_transform(scaled)

    # Inverse transform the scaled imputed data back to original scale
    imputed = scaler.inverse_transform(imputed_scaled)
    imputed_df = pd.DataFrame(imputed, columns=cols, index=dataset.index)

    # Return the dataset with imputed values for target_cols
    dataset[target_cols] = imputed_df[target_cols]
    return dataset

# Encoding categorical columns using LabelEncoder
def label_encode_columns(dataset, columns):
    """
    Label encode one or more columns in place, returning the fitted encoders
    so you can inverse_transform or check the class mapping later.

    Parameters
    ----------
    dataset : pd.DataFrame
        The full dataset (modified in place).
    columns : str or list of str
        Name of the column to encode, or a list of column names.

    Returns
    -------
    dict
        Mapping of column name -> fitted LabelEncoder
        (encoder.classes_ shows the original category order for that column).
    """
    # Allow passing a single column name as a string
    if isinstance(columns, str):
        columns = [columns]

    #dict creation to store encodings for each column
    encoders = {}
    for column in columns:
        encoder = LabelEncoder()
        dataset[column] = encoder.fit_transform(dataset[column].astype(str))
        encoders[column] = encoder

    return encoders

# Decoding label-encoded columns back to their original values
def label_decode_columns(dataset, columns, encoders):
    """
    Decode one or more label-encoded columns back to their original values.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset containing encoded columns (modified in place).
    columns : str or list of str
        Column name(s) to decode.
    encoders : dict
        Dictionary of {column_name: fitted LabelEncoder}, as returned
        by label_encode_columns.
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        dataset[column] = encoders[column].inverse_transform(dataset[column])

    return dataset
```
</details>


```python
cols = ['Income', 'Age', 'Education']

# Encode Education
edu_encoder = label_encode_columns(df, 'Education')['Education']
print(dict(zip(edu_encoder.classes_, edu_encoder.transform(edu_encoder.classes_))))

# Impute missing values for Income using KNN with scaling
df = knn_impute_scaled(df, cols, target_cols=['Income'])

# Check remaining missing values
print(df["Income"].isna().sum())
```

    {'2n Cycle': np.int64(0), 'Basic': np.int64(1), 'Graduation': np.int64(2), 'Master': np.int64(3), 'PhD': np.int64(4)}
    0
    

# Data Visualiztion

### Correlation Matrix

Lets use a quick Correlation Matrix to see what features correspond with each other. While this may show some obvious insghts *(i.e. Total Spent & Total Purchases highly correlated)*  
Other features such as Total Purchases & Age having slight correlation is interesting

<details markdown="1">
<summary>Show code</summary>

```python
rele_encoder = label_encode_columns(df, 'Relationship_Status')['Relationship_Status']
print(dict(zip(rele_encoder.classes_, rele_encoder.transform(rele_encoder.classes_))))

corr_df = df[['Income', 'Age', 'Education', 'Total_Spent', 'Total_Purchases', 'Has_Children', 'Complain', 'Response', 'Recency', 'Relationship_Status']].corr()

matrix = np.triu(corr_df)

#defining size, plotting our corr() function, masking upper half
#annot gives us the labels, and fmt makes the labels 2 decimals long. 
plt.figure(figsize=(8,6))
sns.heatmap(corr_df, mask=matrix, annot = True, fmt = '.2f')
plt.xticks(rotation=45, ha='right')

df['Relationship_Status'] = rele_encoder.inverse_transform(df['Relationship_Status'])
df['Education'] = edu_encoder.inverse_transform(df['Education'])
```

    {'Relationship': np.int64(0), 'Single': np.int64(1)}
    
</details>

    
![png](/assets/Marketing_pics/Marketing_Campaign_24_1.png)
    


### Regression

Referencing our matrix above, lets compare Income vs. Total Spending and see what our results our. 

Using Income as our independent variable and Total Spent as our dependent variable, lets place them on a scatterplot and fit a line.

<details markdown="1">
<summary>Show code</summary>

```python
X = df[['Income']]
y = df['Total_Spent']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.scatter(X, y, alpha=0.6, label='Data points')
plt.plot(X, y_pred, color='red', label='Regression line')


plt.ylabel('Total Spending')
plt.xlabel('Income')
plt.title('Income vs. Total Spending')
plt.legend()
plt.show()

print(f"R² score: {model.score(X, y):.3f}")
print(f"Coefficient: {model.coef_[0]:.3f}")
print(f"Intercept: {model.intercept_:.3f}")
```

</details>
    
![png](/assets/Marketing_pics/Marketing_Campaign_26_0.png)
    


    R² score: 0.620
    Coefficient: 0.022
    Intercept: -542.925
    

Looking at our results above we get som odd results. Our regression line extends below zero with an intercept of **-542.92**. this is essentially saying when Income is 0, they will spend negative money (*We give them money?*).  

Obviously this is unrealistic, its simply where the fitted line crosses our intercept.  

Our $R^2$ is .62, telling us 62% of the variance in the total amount spent is explained by Income.  

The coeffcient of .022 tells us for each $1 increase in Income, the amount sepnt increases by .02 cents.

### Bar Chart

Lets see what the distrobution of age looks like comapred to the amount spent, based on the item type

*P.S. This plot is interactable! Click on the colors in the legend!*

<details markdown="1">
<summary>Show code</summary>

```python
columns = ['MntFishProducts', 'MntMeatProducts', 'MntFruits', 
           'MntSweetProducts', 'MntWines', 'MntGoldProds']

df_long = df.melt(id_vars='Age', value_vars=columns, 
                   var_name='Product', value_name='Spending')

df_grouped = df_long.groupby(['Age', 'Product'], as_index=False)['Spending'].sum()

fig = px.bar(df_grouped, x='Age', y='Spending', color='Product',
             barmode='overlay', opacity=0.6)
fig.show()
```
</details>

<iframe src="/assets/plotly/BarChart.html" width="100%" height="500" frameborder="0" scrolling="no"></iframe>

Our age distribution is slightly skewed, showing we have an older customer base. Filtering our colors in the top right, it does not appear that we sell any parituclar items more within a specific age demographic. 

All items appear to sell equaly amongst the varying ages, however some products customers are spending much more than others, indicated by the changing "Spending Values" on the left hand side. Filtering the graph for **"MntWines"** shows us that Wines have a heavy spending amount, followed by **"MntMeats"** second. 

# Customer Segmentation (RFM / Clustering)

RFM stands for Recency, Frequency, Monetary, a classic marketing framework for scoring customers based on:
- How recently they purchased  
- How often they purchase  
- How much they spend  

Combining it with K-Means clustering groups customers into distinct segments based on similarity across these dimensions.

In our case:  
- Recency = days since last purchase (*lower is better/more engaged*)
- Frequency = total number of purchases across all channels (*web, store, catalog, deals*)
- Monetary = total spend across all product categories

<details markdown="1">
<summary>Show code</summary>

```python
# Get RFM features
spend_cols = ['MntFishProducts', 'MntMeatProducts', 'MntFruits', 
              'MntSweetProducts', 'MntWines', 'MntGoldProds']
purchase_cols = ['NumDealsPurchases', 'NumCatalogPurchases', 
                  'NumStorePurchases', 'NumWebPurchases']
# Create the RFM total features
df['TotalSpend'] = df[spend_cols].sum(axis=1)
df['TotalPurchases'] = df[purchase_cols].sum(axis=1)

# Recency =  "days since last purchase", lower number = more recent/engaged
# Renaming columns for RFM analysis and ease of reference
rfm = df[['Recency', 'TotalPurchases', 'TotalSpend']].copy()
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# # Scale features (# KMeans clustering works by calculating Euclidean distance, standardization is important, or it may be biased towards features with larger scales)
# StandardScaler transforms each column to have mean = 0 and standard deviation = 1 (i.e., z-scores)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Elbow method to choose k
inertias = []
# Inertia = Sum of squared distances between each point and its assigned cluster center (lower inertia = tighter clusters).
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10) # N_init = number of times the algorithm will run with different centroid seeds (default=10)
    km.fit(rfm_scaled)
    inertias.append(km.inertia_)

# Using the KneeLocator to find the optimal k based on the elbow method
kl = KneeLocator(list(K_range), inertias, curve='convex', direction='decreasing')
optimal_k = kl.elbow

# Plot elbow method with the optimal k marked
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker='o')
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Elbow at k={optimal_k}')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Fit KMeans with chosen k (adjust based on elbow plot)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(rfm_scaled)

#Compress clusters to 2D via Pricnipal Component Analysis (PCA)
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)

# Visualize clusters in 2D via PCA
plt.figure(figsize=(10, 7))
sns.scatterplot(x=rfm_pca[:, 0], y=rfm_pca[:, 1], hue=df['Cluster'], 
                 palette='tab10', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Segments (RFM + KMeans)')
plt.legend(title='Cluster')
plt.show()

#Profile each cluster
cluster_profile = df.groupby('Cluster')[['Recency', 'TotalPurchases', 
                                          'TotalSpend', 'Income']].mean()
print(cluster_profile)
```
</details>


![png](/assets/Marketing_pics/Marketing_Campaign_32_0.png)
    



    
![png](/assets/Marketing_pics/Marketing_Campaign_32_1.png)
    


             Recency  TotalPurchases  TotalSpend   Income
    Cluster                                              
    0          24.57            8.64      128.33 36848.78
    1          73.25           21.49     1151.02 68463.91
    2          23.38           21.93     1123.94 68712.38
    3          74.84            8.97      138.66 37552.36
    

# Customer Segmentation (RFM / Clustering) Results

### What the results mean:
Once we fit KMeans and look at the cluster_profile matrix, each cluster shows a distinct pattern. The archetypes that emerge from this data are:

**New/Low-engagement - (Cluster 0)**: Low frequency, low monetary, could be recent sign ups who haven't ramped up yet.  
**At-risk high-value - (Cluster 1)**: High recency (haven't bought in a while), used to be frequent/high spend, needs win back campaigns.  
**Champions - (Cluster 2)**: Low recency (bought recently), high frequency, high monetary, our best customers.  
**Lapsed low-value - (Cluster 3)**: High frequency on NumDealsPurchases specifically, but lower overall monetary value, meaning they are price sensitive.  


### Marketing recommendations per cluster:


**0 - Low-engagement/New** - Onboarding nurture sequences, educational content about product categories, no aggressive discounting yet.  
**1 - At-risk** - Targeted campaigns, personalized discount codes, "we miss you" email sequences.  
**2 - Deal-seekers** - Keep sending discount-based campaigns via NumDealsPurchases channel, but avoid relying too much on price cuts, as it may erode margin.  
**3 - Champions** - Loyalty programs, early access to new products, referral incentives (don't waste discount budget here, they're already purchasing).    


# 2. Campaign Response Prediction (Classification Model)

In this section we are using a Random Forest model to predict the Response (1 = accepted the last campaign, 0 = didn't) based on customer demographics, spending, and behavior.  
The model learns patterns from historical data that distinguish responders from non-responders. 

<details markdown="1">
<summary>Show code</summary>

```python
# Prepare features, dropping columns that are not useful for modeling, or redundant with engineered features
df_model = df.copy()
df_model.drop(columns=['ID','Year_Birth','Marital_Status','Total_Spent','Total_Purchases','TotalSpend','TotalPurchases','Kidhome','Teenhome'], inplace=True)

# Encode categoricals
df_model = pd.get_dummies(df_model, columns=['Education', 'Relationship_Status'], drop_first=True)

# Drop columns not useful as raw features
drop_cols = ['DtCustomer', 'Response']  # target excluded
feature_cols = [c for c in df_model.columns if c not in drop_cols 
                 and df_model[c].dtype in ['int64', 'float64', 'bool', 'uint8']]

# Feature and target selection
X = df_model[feature_cols]
y = df_model['Response']

# Handle missing Income if present
X['Income'] = X['Income'].fillna(X['Income'].median())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale (helps some models, harmless for tree-based ones too)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
clf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = clf.predict(X_test_scaled)
y_proba = clf.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Campaign Response')
plt.show()

# ROC curve
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title('ROC Curve: Campaign Response Prediction')
plt.show()

# Feature importance
importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x=importances.head(15).values, y=importances.head(15).index)
plt.title('Top 15 Feature Importances')
plt.xlabel('Importance')
plt.show()
```

                  precision    recall  f1-score   support
    
               0       0.91      0.95      0.93       381
               1       0.63      0.49      0.55        67
    
        accuracy                           0.88       448
       macro avg       0.77      0.72      0.74       448
    weighted avg       0.87      0.88      0.88       448
    
    ROC-AUC: 0.877
    
</details>

    
![png](/assets/Marketing_pics/Marketing_Campaign_35_1.png)
    



    
![png](/assets/Marketing_pics/Marketing_Campaign_35_2.png)
    



    
![png](/assets/Marketing_pics/Marketing_Campaign_35_3.png)
    


# Campaign Response Prediction Results

### Classification report (precision/recall/F1) / Confusion Matrix:

- Precision for class 1 tells us: of everyone the model predicted would respond, what % actually did.  
-- High precision = less wasted money on people who won't convert.  
- Recall for class 1 tells us: of everyone who actually responded, what percent did the model catch.  
-- High recall = we're not missing potential responders.  

In marketing, we would often care more about **recall** (enusres we don't miss good prospects) unless campaign costs are high, in which case, **precision** matters more (ensures don't waste money).

### ROC-AUC score plot:

- 0.5 (random guessing)  
- 1.0 (perfect separation)   

Above 0.75–0.8 is considered good for marketing response models, which tend to be noisy by nature.

### Feature importance plot:

This is often the most actionable part. If Income, Recency, or specific AcceptedCmp history rank highest, it tells us what actually drives campaign acceptance. In this case it appears customers who accepted a previous campaign are more likely to accept future ones.


### Final Summary: 

Overall, this code performs a comprehensive analysis of a marketing campaign dataset. It includes data preprocessing, feature engineering, exploratory data analysis, visualizations and model training using a Random Forest Classifier to predict customer response. The model's performance is evaluated using classification metrics, confusion matrix, and ROC curve. Additionally, feature importance is visualized to understand the impact of different features on the model's predictions and future marketing actions.
