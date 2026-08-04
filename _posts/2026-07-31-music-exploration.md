--- 
title: Music Data Exploration with Pandas 
date: 2026-07-31 12:00:00 -0700 
categories: [Data Analysis, Python] 
tags: [python, pandas, music,pandas,visualiztion]
---

# Music Exploration
Garren Ramirez

### Motivation 

For this project I decided to look at popular music statistics and its qualites. I chose to use the the "Top Hits Spotfy from 2000-2019" from Kaggle.com ( https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019 ). I chose this dataset because I love music, I thoroughly enjoy listening to a variety of music genres and exploring music from other countries as well, so I feel trying to find the qualities of what makes a 'hit' could be an interesting project. I will be conducting a statistical analysis trying to find correlations between a tracks popularity and influences from other variables. The dataset is decribed as such from Kaggle, 'This dataset contains audio statistics of the top 2000 tracks on Spotify from 2000-2019. The data contains about 18 columns each describing the track and it's qualities.' 
First and foremost, we need to import the necessary libraries needed for this project, numpy, matplotlib, pandas, and seaborn are all of the necessary libraries I have imported. 


```python
#imported libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
```


```python
#naming dataframe, showing first 5 values
music_df = pd.read_csv("songs_normalize.csv")
music_df.head()
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
      <th>artist</th>
      <th>song</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>year</th>
      <th>popularity</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Britney Spears</td>
      <td>Oops!...I Did It Again</td>
      <td>211160</td>
      <td>False</td>
      <td>2000</td>
      <td>77</td>
      <td>0.751</td>
      <td>0.834</td>
      <td>1</td>
      <td>-5.444</td>
      <td>0</td>
      <td>0.0437</td>
      <td>0.3000</td>
      <td>0.000018</td>
      <td>0.3550</td>
      <td>0.894</td>
      <td>95.053</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>1</th>
      <td>blink-182</td>
      <td>All The Small Things</td>
      <td>167066</td>
      <td>False</td>
      <td>1999</td>
      <td>79</td>
      <td>0.434</td>
      <td>0.897</td>
      <td>0</td>
      <td>-4.918</td>
      <td>1</td>
      <td>0.0488</td>
      <td>0.0103</td>
      <td>0.000000</td>
      <td>0.6120</td>
      <td>0.684</td>
      <td>148.726</td>
      <td>rock, pop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Faith Hill</td>
      <td>Breathe</td>
      <td>250546</td>
      <td>False</td>
      <td>1999</td>
      <td>66</td>
      <td>0.529</td>
      <td>0.496</td>
      <td>7</td>
      <td>-9.007</td>
      <td>1</td>
      <td>0.0290</td>
      <td>0.1730</td>
      <td>0.000000</td>
      <td>0.2510</td>
      <td>0.278</td>
      <td>136.859</td>
      <td>pop, country</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bon Jovi</td>
      <td>It's My Life</td>
      <td>224493</td>
      <td>False</td>
      <td>2000</td>
      <td>78</td>
      <td>0.551</td>
      <td>0.913</td>
      <td>0</td>
      <td>-4.063</td>
      <td>0</td>
      <td>0.0466</td>
      <td>0.0263</td>
      <td>0.000013</td>
      <td>0.3470</td>
      <td>0.544</td>
      <td>119.992</td>
      <td>rock, metal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>*NSYNC</td>
      <td>Bye Bye Bye</td>
      <td>200560</td>
      <td>False</td>
      <td>2000</td>
      <td>65</td>
      <td>0.614</td>
      <td>0.928</td>
      <td>8</td>
      <td>-4.806</td>
      <td>0</td>
      <td>0.0516</td>
      <td>0.0408</td>
      <td>0.001040</td>
      <td>0.0845</td>
      <td>0.879</td>
      <td>172.656</td>
      <td>pop</td>
    </tr>
  </tbody>
</table>
</div>




```python
#displaying last 5 values
music_df.tail()
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
      <th>artist</th>
      <th>song</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>year</th>
      <th>popularity</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1995</th>
      <td>Jonas Brothers</td>
      <td>Sucker</td>
      <td>181026</td>
      <td>False</td>
      <td>2019</td>
      <td>79</td>
      <td>0.842</td>
      <td>0.734</td>
      <td>1</td>
      <td>-5.065</td>
      <td>0</td>
      <td>0.0588</td>
      <td>0.0427</td>
      <td>0.000000</td>
      <td>0.1060</td>
      <td>0.952</td>
      <td>137.958</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>Taylor Swift</td>
      <td>Cruel Summer</td>
      <td>178426</td>
      <td>False</td>
      <td>2019</td>
      <td>78</td>
      <td>0.552</td>
      <td>0.702</td>
      <td>9</td>
      <td>-5.707</td>
      <td>1</td>
      <td>0.1570</td>
      <td>0.1170</td>
      <td>0.000021</td>
      <td>0.1050</td>
      <td>0.564</td>
      <td>169.994</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>Blanco Brown</td>
      <td>The Git Up</td>
      <td>200593</td>
      <td>False</td>
      <td>2019</td>
      <td>69</td>
      <td>0.847</td>
      <td>0.678</td>
      <td>9</td>
      <td>-8.635</td>
      <td>1</td>
      <td>0.1090</td>
      <td>0.0669</td>
      <td>0.000000</td>
      <td>0.2740</td>
      <td>0.811</td>
      <td>97.984</td>
      <td>hip hop, country</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>Sam Smith</td>
      <td>Dancing With A Stranger (with Normani)</td>
      <td>171029</td>
      <td>False</td>
      <td>2019</td>
      <td>75</td>
      <td>0.741</td>
      <td>0.520</td>
      <td>8</td>
      <td>-7.513</td>
      <td>1</td>
      <td>0.0656</td>
      <td>0.4500</td>
      <td>0.000002</td>
      <td>0.2220</td>
      <td>0.347</td>
      <td>102.998</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>Post Malone</td>
      <td>Circles</td>
      <td>215280</td>
      <td>False</td>
      <td>2019</td>
      <td>85</td>
      <td>0.695</td>
      <td>0.762</td>
      <td>0</td>
      <td>-3.497</td>
      <td>1</td>
      <td>0.0395</td>
      <td>0.1920</td>
      <td>0.002440</td>
      <td>0.0863</td>
      <td>0.553</td>
      <td>120.042</td>
      <td>hip hop</td>
    </tr>
  </tbody>
</table>
</div>



Above is a small sample output of the dataframe, I named my data frame "music_df and I output the first and last 5 rows. The data is roughly ordered from oldest to most recent songs based on year released. One thing to note is that in the .head() output, we actually have songs that were released in 1999, my guess is that they were released on the edge of 1999 so they were grouped in with songs from 2000. 
For sake of quality and consistency, I am going to remove anything that does not fall in the 2000-2019 year range. Just to be sure, I will check the unique years present in the data frame in the next cell.
Looking at our variables, we have the standard self explanatory ones, such as artist and year, however there are others that could use some context. 

**Popularity-** "The higher the value the more popular the song is."

**Danceability-**  "Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable."

**Energy-**  "Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity."

**Key-**  "The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1."

**Loudness-**  "The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db."

**Mode-**  "Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0."

**Speechiness-**  "Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks."

**Acousticness-**  "A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic."

**Instrumentalness-**  "Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0."

**Valence-**  "A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)."

**Liveness-**  "Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live."

**Tempo-**  "The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration."

##### Reference

Koverha, Mark. “Top Hits Spotify from 2000-2019.” Kaggle, Kaggle.com, 31 May 2022, https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019.

*All descriptions above were taken directly from the Kaggle webpage for this dataset. Credit to Mark Koverha*

### Cleaning


```python
#displaying all years within the dataframe
years = music_df.year.unique()
years.sort()
years
```




    array([1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
           2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
           2020])




```python
#grouping years before 2000 and after 2019, then dropping them after
dropYear1 = music_df[(music_df['year'] < 2000)].index 
dropYear2 = music_df[(music_df['year'] > 2019)].index

music_df = music_df.drop(dropYear1)
music_df = music_df.drop(dropYear2)

#dropping na values, and displaying first 5 records
music_df.dropna()
music_df.head()
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
      <th>artist</th>
      <th>song</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>year</th>
      <th>popularity</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Britney Spears</td>
      <td>Oops!...I Did It Again</td>
      <td>211160</td>
      <td>False</td>
      <td>2000</td>
      <td>77</td>
      <td>0.751</td>
      <td>0.834</td>
      <td>1</td>
      <td>-5.444</td>
      <td>0</td>
      <td>0.0437</td>
      <td>0.3000</td>
      <td>0.000018</td>
      <td>0.3550</td>
      <td>0.894</td>
      <td>95.053</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bon Jovi</td>
      <td>It's My Life</td>
      <td>224493</td>
      <td>False</td>
      <td>2000</td>
      <td>78</td>
      <td>0.551</td>
      <td>0.913</td>
      <td>0</td>
      <td>-4.063</td>
      <td>0</td>
      <td>0.0466</td>
      <td>0.0263</td>
      <td>0.000013</td>
      <td>0.3470</td>
      <td>0.544</td>
      <td>119.992</td>
      <td>rock, metal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>*NSYNC</td>
      <td>Bye Bye Bye</td>
      <td>200560</td>
      <td>False</td>
      <td>2000</td>
      <td>65</td>
      <td>0.614</td>
      <td>0.928</td>
      <td>8</td>
      <td>-4.806</td>
      <td>0</td>
      <td>0.0516</td>
      <td>0.0408</td>
      <td>0.001040</td>
      <td>0.0845</td>
      <td>0.879</td>
      <td>172.656</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Eminem</td>
      <td>The Real Slim Shady</td>
      <td>284200</td>
      <td>True</td>
      <td>2000</td>
      <td>86</td>
      <td>0.949</td>
      <td>0.661</td>
      <td>5</td>
      <td>-4.244</td>
      <td>0</td>
      <td>0.0572</td>
      <td>0.0302</td>
      <td>0.000000</td>
      <td>0.0454</td>
      <td>0.760</td>
      <td>104.504</td>
      <td>hip hop</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Robbie Williams</td>
      <td>Rock DJ</td>
      <td>258560</td>
      <td>False</td>
      <td>2000</td>
      <td>68</td>
      <td>0.708</td>
      <td>0.772</td>
      <td>7</td>
      <td>-4.264</td>
      <td>1</td>
      <td>0.0322</td>
      <td>0.0267</td>
      <td>0.000000</td>
      <td>0.4670</td>
      <td>0.861</td>
      <td>103.035</td>
      <td>pop, rock</td>
    </tr>
  </tbody>
</table>
</div>




```python
#rechecking the years contained after dropping ones not in 2000-2019 year range.
years = music_df.year.unique()
years.sort()
years
```




    array([2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
           2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])



Above, in our first cell of the group, I found the unique years in the dataset, we actually had 1998, 1999, and 2020 values within the set, so I simply removed them by extracting values that are lower than 2000 and higher than 2019 to variables, and then dropped them from the 'music_df'. 
In our last cell from the group above, I simply checked to see if our method to drop years outside of 2000-2019 worked, and based on the output, it is clear to see that it did. I also used '.dropna()' just in case there are any NA values within the dataframe that we cannot see. 

Below, I actually found later while creating outputs, that the data set had a strange value in the genre column. It had 'set()' as a category and I believe this to be an error by the creator of the dataset. So I simply went back to here and followed the same process as above, but with '==set()' this time, and removed the values with it from the dataset.


```python
#grouping records that have 'set()' as a genre, then dropping them
genre = music_df[(music_df['genre'] == 'set()')].index
music_df = music_df.drop(genre)
```

In the following cell, I decided to create a function that returns the length of the song in minutes and seconds, and then added it to the the data frame. It is not truly numerical, but I think that it is easier to read than milliseconds. But for any calculations or measurement, 'duration_ms' will have to be used.


```python
#function to change milliseconds to minutes:seconds
def to_min(ms):
    
    secondsDeci=ms/1000
    
    minutes = int(secondsDeci//60)
    seconds = int(secondsDeci%60)
    seconds = str(seconds)
    
    if len(seconds) == 1:
        seconds = '0' +seconds 

    return f'{minutes}:{seconds}'

#applying function above to 'duration_ms' and then creating a new column that contains minutes and seconds
music_df['duration_mins'] = music_df['duration_ms'].apply(to_min)

#display 5 values to see new column
music_df.head()
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
      <th>artist</th>
      <th>song</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>year</th>
      <th>popularity</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>genre</th>
      <th>duration_mins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Britney Spears</td>
      <td>Oops!...I Did It Again</td>
      <td>211160</td>
      <td>False</td>
      <td>2000</td>
      <td>77</td>
      <td>0.751</td>
      <td>0.834</td>
      <td>1</td>
      <td>-5.444</td>
      <td>0</td>
      <td>0.0437</td>
      <td>0.3000</td>
      <td>0.000018</td>
      <td>0.3550</td>
      <td>0.894</td>
      <td>95.053</td>
      <td>pop</td>
      <td>3:31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bon Jovi</td>
      <td>It's My Life</td>
      <td>224493</td>
      <td>False</td>
      <td>2000</td>
      <td>78</td>
      <td>0.551</td>
      <td>0.913</td>
      <td>0</td>
      <td>-4.063</td>
      <td>0</td>
      <td>0.0466</td>
      <td>0.0263</td>
      <td>0.000013</td>
      <td>0.3470</td>
      <td>0.544</td>
      <td>119.992</td>
      <td>rock, metal</td>
      <td>3:44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>*NSYNC</td>
      <td>Bye Bye Bye</td>
      <td>200560</td>
      <td>False</td>
      <td>2000</td>
      <td>65</td>
      <td>0.614</td>
      <td>0.928</td>
      <td>8</td>
      <td>-4.806</td>
      <td>0</td>
      <td>0.0516</td>
      <td>0.0408</td>
      <td>0.001040</td>
      <td>0.0845</td>
      <td>0.879</td>
      <td>172.656</td>
      <td>pop</td>
      <td>3:20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Eminem</td>
      <td>The Real Slim Shady</td>
      <td>284200</td>
      <td>True</td>
      <td>2000</td>
      <td>86</td>
      <td>0.949</td>
      <td>0.661</td>
      <td>5</td>
      <td>-4.244</td>
      <td>0</td>
      <td>0.0572</td>
      <td>0.0302</td>
      <td>0.000000</td>
      <td>0.0454</td>
      <td>0.760</td>
      <td>104.504</td>
      <td>hip hop</td>
      <td>4:44</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Robbie Williams</td>
      <td>Rock DJ</td>
      <td>258560</td>
      <td>False</td>
      <td>2000</td>
      <td>68</td>
      <td>0.708</td>
      <td>0.772</td>
      <td>7</td>
      <td>-4.264</td>
      <td>1</td>
      <td>0.0322</td>
      <td>0.0267</td>
      <td>0.000000</td>
      <td>0.4670</td>
      <td>0.861</td>
      <td>103.035</td>
      <td>pop, rock</td>
      <td>4:18</td>
    </tr>
  </tbody>
</table>
</div>



In the cell below, I had to return back to my cleaning phase once again, because when I was trying to find the top rated songs and display them, I saw that there were actually repeat entries, something that I had not expected. So, I simply used the 'drop_duplicates' function on the 'song' column which did the trick. Later in this exploration, I show the most popular songs, so I will not display them here


```python
#getting rid of duplicate entries, filtering by song since the list is based off of top songs. 
music_df = music_df.drop_duplicates(subset="song")
```

### Basic analysis


```python
#displaying a table with some basic calculations using .describe()
#used .T to transpose the table
music_df.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>duration_ms</th>
      <td>1819.0</td>
      <td>228322.816932</td>
      <td>39093.005355</td>
      <td>113000.000000</td>
      <td>203499.50000</td>
      <td>223066.0000</td>
      <td>247133.000000</td>
      <td>484146.000</td>
    </tr>
    <tr>
      <th>year</th>
      <td>1819.0</td>
      <td>2009.624519</td>
      <td>5.739703</td>
      <td>2000.000000</td>
      <td>2005.00000</td>
      <td>2010.0000</td>
      <td>2015.000000</td>
      <td>2019.000</td>
    </tr>
    <tr>
      <th>popularity</th>
      <td>1819.0</td>
      <td>59.470588</td>
      <td>21.763217</td>
      <td>0.000000</td>
      <td>56.00000</td>
      <td>65.0000</td>
      <td>73.000000</td>
      <td>89.000</td>
    </tr>
    <tr>
      <th>danceability</th>
      <td>1819.0</td>
      <td>0.667239</td>
      <td>0.140941</td>
      <td>0.129000</td>
      <td>0.58100</td>
      <td>0.6760</td>
      <td>0.765000</td>
      <td>0.975</td>
    </tr>
    <tr>
      <th>energy</th>
      <td>1819.0</td>
      <td>0.722607</td>
      <td>0.152268</td>
      <td>0.054900</td>
      <td>0.62550</td>
      <td>0.7390</td>
      <td>0.840500</td>
      <td>0.999</td>
    </tr>
    <tr>
      <th>key</th>
      <td>1819.0</td>
      <td>5.384277</td>
      <td>3.611830</td>
      <td>0.000000</td>
      <td>2.00000</td>
      <td>6.0000</td>
      <td>8.000000</td>
      <td>11.000</td>
    </tr>
    <tr>
      <th>loudness</th>
      <td>1819.0</td>
      <td>-5.501156</td>
      <td>1.945194</td>
      <td>-20.514000</td>
      <td>-6.47650</td>
      <td>-5.2730</td>
      <td>-4.164000</td>
      <td>-0.276</td>
    </tr>
    <tr>
      <th>mode</th>
      <td>1819.0</td>
      <td>0.554700</td>
      <td>0.497136</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.0000</td>
      <td>1.000000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>speechiness</th>
      <td>1819.0</td>
      <td>0.105263</td>
      <td>0.097158</td>
      <td>0.023900</td>
      <td>0.04005</td>
      <td>0.0615</td>
      <td>0.132000</td>
      <td>0.576</td>
    </tr>
    <tr>
      <th>acousticness</th>
      <td>1819.0</td>
      <td>0.127608</td>
      <td>0.172942</td>
      <td>0.000019</td>
      <td>0.01355</td>
      <td>0.0548</td>
      <td>0.176000</td>
      <td>0.976</td>
    </tr>
    <tr>
      <th>instrumentalness</th>
      <td>1819.0</td>
      <td>0.015626</td>
      <td>0.089459</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.0000</td>
      <td>0.000064</td>
      <td>0.985</td>
    </tr>
    <tr>
      <th>liveness</th>
      <td>1819.0</td>
      <td>0.182514</td>
      <td>0.141887</td>
      <td>0.021500</td>
      <td>0.08825</td>
      <td>0.1260</td>
      <td>0.242000</td>
      <td>0.853</td>
    </tr>
    <tr>
      <th>valence</th>
      <td>1819.0</td>
      <td>0.552490</td>
      <td>0.220640</td>
      <td>0.038100</td>
      <td>0.39000</td>
      <td>0.5580</td>
      <td>0.731000</td>
      <td>0.973</td>
    </tr>
    <tr>
      <th>tempo</th>
      <td>1819.0</td>
      <td>120.153423</td>
      <td>26.903636</td>
      <td>60.019000</td>
      <td>98.98550</td>
      <td>120.0210</td>
      <td>134.052000</td>
      <td>210.851</td>
    </tr>
  </tbody>
</table>
</div>




```python
# finding min, max, and mean in minutes and seconds.
meanLength = music_df['duration_ms'].mean()
minLength = music_df['duration_ms'].min()
maxLength = music_df['duration_ms'].max()

#printing min, max, mean utilizing defined function above. 
print(f'Average top hit duration in minutes and seconds: {to_min(meanLength)}')
print(f'Duration of smallest song length {to_min(minLength)}')
print(f'Duration of largest song length {to_min(maxLength)}')
```

    Average top hit duration in minutes and seconds: 3:48
    Duration of smallest song length 1:53
    Duration of largest song length 8:04
    

I wanted to get some basic numbers for our variables within the dataframe. So I used '.describe()' in order to get a basic table that gives us the mean, standard deviation, min, percentiles, and max values. I used '.T' to transpose the table, becuase it did not read as easily when count, mean, etc. were on the y axis. I also printed out the mean of 'duration_ms' in minutes and seconds because I feel it is easier to read. 

Some interesting things to note here, our tempo (BPM) is averaging around 120, an example of this kind of tempo most people would know is Poker Face by Lady Gaga. Another interesting feature is our average popularity is only around 60. I would have expected this be around 70 or 80 considering these are all 'top hits', but I'm sure some top hits are more popular than others. Another feature is that our valence, which is described as mood of the song, is .55. This would techincally be a neutral mood, so I think this indicates we have a good mix of 'moods' in the top hits. 


```python
#returning the counts of unique artists within the dataframe.
artist = music_df['artist'].value_counts()
artist
```




    Rihanna              22
    Drake                20
    Calvin Harris        20
    Eminem               19
    David Guetta         18
                         ..
    Oliver Heldens        1
    Cher Lloyd            1
    Waka Flocka Flame     1
    The Strumbellas       1
    Pixie Lott            1
    Name: artist, Length: 786, dtype: int64



Above, I wanted to see basic distribution of our artists, so I used '.value_counts()' to group together the artists and return a simple table with the amount of times they appear and assigned it to 'artist'. I was a little surprised to see that there are only 786 unique artists with top hits in a list of about 2000 songs over a 20 year period. This indicates that a lot of artists had more than one hit, making this distribution a little unequal.

Below, I wanted to see what our top 20 artists are for this 20 year period. I must say that I am not surprised by the output of this, as a lot of these artists I hear on the radio quite frequently and they are very common names within popular american music. One thing to note is that all of these artists still have over 10 hits alone, relating back to the unequal distribution of artists and top hits. Most of the popular artists from the output below fall into the top genre categories presented later, which is also not surprsing. 


```python
#displaying the artists with the most songs within the dataframe. 
artist.nlargest(20)
```




    Rihanna              22
    Drake                20
    Calvin Harris        20
    Eminem               19
    David Guetta         18
    Britney Spears       18
    Katy Perry           16
    Chris Brown          15
    Beyoncé              15
    Kanye West           15
    P!nk                 14
    Black Eyed Peas      14
    Taylor Swift         13
    Maroon 5             13
    Ariana Grande        13
    Bruno Mars           11
    Justin Timberlake    11
    Coldplay             11
    Nelly                11
    Usher                11
    Name: artist, dtype: int64



In the cell below, I was actually curious what the most popular songs are within this data frame. I created a new dataframe, sorting by popularity rating, and then I displayed the top 10 songs. Interestingly, Eminem appears 3 times within the top 10 list, already taking up 30% of our 10 samples. 


```python
#creating new dataframe, sorting songs by popularity, displaying top 10 songs
songs = music_df.sort_values('popularity', ascending = False)
songs[0:10]
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
      <th>artist</th>
      <th>song</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>year</th>
      <th>popularity</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>genre</th>
      <th>duration_mins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1322</th>
      <td>The Neighbourhood</td>
      <td>Sweater Weather</td>
      <td>240400</td>
      <td>False</td>
      <td>2013</td>
      <td>89</td>
      <td>0.612</td>
      <td>0.807</td>
      <td>10</td>
      <td>-2.810</td>
      <td>1</td>
      <td>0.0336</td>
      <td>0.04950</td>
      <td>0.017700</td>
      <td>0.1010</td>
      <td>0.398</td>
      <td>124.053</td>
      <td>rock, pop</td>
      <td>4:00</td>
    </tr>
    <tr>
      <th>1311</th>
      <td>Tom Odell</td>
      <td>Another Love</td>
      <td>244360</td>
      <td>True</td>
      <td>2013</td>
      <td>88</td>
      <td>0.445</td>
      <td>0.537</td>
      <td>4</td>
      <td>-8.532</td>
      <td>0</td>
      <td>0.0400</td>
      <td>0.69500</td>
      <td>0.000017</td>
      <td>0.0944</td>
      <td>0.131</td>
      <td>122.769</td>
      <td>pop</td>
      <td>4:04</td>
    </tr>
    <tr>
      <th>201</th>
      <td>Eminem</td>
      <td>Without Me</td>
      <td>290320</td>
      <td>True</td>
      <td>2002</td>
      <td>87</td>
      <td>0.908</td>
      <td>0.669</td>
      <td>7</td>
      <td>-2.827</td>
      <td>1</td>
      <td>0.0738</td>
      <td>0.00286</td>
      <td>0.000000</td>
      <td>0.2370</td>
      <td>0.662</td>
      <td>112.238</td>
      <td>hip hop</td>
      <td>4:50</td>
    </tr>
    <tr>
      <th>1613</th>
      <td>WILLOW</td>
      <td>Wait a Minute!</td>
      <td>196520</td>
      <td>False</td>
      <td>2015</td>
      <td>86</td>
      <td>0.764</td>
      <td>0.705</td>
      <td>3</td>
      <td>-5.279</td>
      <td>0</td>
      <td>0.0278</td>
      <td>0.03710</td>
      <td>0.000019</td>
      <td>0.0943</td>
      <td>0.672</td>
      <td>101.003</td>
      <td>pop, R&amp;B, Dance/Electronic</td>
      <td>3:16</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Eminem</td>
      <td>The Real Slim Shady</td>
      <td>284200</td>
      <td>True</td>
      <td>2000</td>
      <td>86</td>
      <td>0.949</td>
      <td>0.661</td>
      <td>5</td>
      <td>-4.244</td>
      <td>0</td>
      <td>0.0572</td>
      <td>0.03020</td>
      <td>0.000000</td>
      <td>0.0454</td>
      <td>0.760</td>
      <td>104.504</td>
      <td>hip hop</td>
      <td>4:44</td>
    </tr>
    <tr>
      <th>1819</th>
      <td>Billie Eilish</td>
      <td>lovely (with Khalid)</td>
      <td>200185</td>
      <td>False</td>
      <td>2018</td>
      <td>86</td>
      <td>0.351</td>
      <td>0.296</td>
      <td>4</td>
      <td>-10.109</td>
      <td>0</td>
      <td>0.0333</td>
      <td>0.93400</td>
      <td>0.000000</td>
      <td>0.0950</td>
      <td>0.120</td>
      <td>115.284</td>
      <td>pop, Dance/Electronic</td>
      <td>3:20</td>
    </tr>
    <tr>
      <th>1227</th>
      <td>Bruno Mars</td>
      <td>Locked out of Heaven</td>
      <td>233478</td>
      <td>False</td>
      <td>2012</td>
      <td>85</td>
      <td>0.726</td>
      <td>0.698</td>
      <td>5</td>
      <td>-4.165</td>
      <td>1</td>
      <td>0.0431</td>
      <td>0.04900</td>
      <td>0.000000</td>
      <td>0.3090</td>
      <td>0.867</td>
      <td>143.994</td>
      <td>pop</td>
      <td>3:53</td>
    </tr>
    <tr>
      <th>1514</th>
      <td>Avicii</td>
      <td>The Nights</td>
      <td>176658</td>
      <td>False</td>
      <td>2014</td>
      <td>85</td>
      <td>0.527</td>
      <td>0.835</td>
      <td>6</td>
      <td>-5.298</td>
      <td>1</td>
      <td>0.0433</td>
      <td>0.01660</td>
      <td>0.000000</td>
      <td>0.2490</td>
      <td>0.654</td>
      <td>125.983</td>
      <td>pop, Dance/Electronic</td>
      <td>2:56</td>
    </tr>
    <tr>
      <th>1512</th>
      <td>The Neighbourhood</td>
      <td>Daddy Issues</td>
      <td>260173</td>
      <td>False</td>
      <td>2015</td>
      <td>85</td>
      <td>0.588</td>
      <td>0.521</td>
      <td>10</td>
      <td>-9.461</td>
      <td>1</td>
      <td>0.0329</td>
      <td>0.06780</td>
      <td>0.149000</td>
      <td>0.1230</td>
      <td>0.337</td>
      <td>85.012</td>
      <td>rock, pop</td>
      <td>4:20</td>
    </tr>
    <tr>
      <th>209</th>
      <td>Eminem</td>
      <td>'Till I Collapse</td>
      <td>297786</td>
      <td>True</td>
      <td>2002</td>
      <td>85</td>
      <td>0.548</td>
      <td>0.847</td>
      <td>1</td>
      <td>-3.237</td>
      <td>1</td>
      <td>0.1860</td>
      <td>0.06220</td>
      <td>0.000000</td>
      <td>0.0816</td>
      <td>0.100</td>
      <td>171.447</td>
      <td>hip hop</td>
      <td>4:57</td>
    </tr>
  </tbody>
</table>
</div>



In the cell below, I wanted to see what the top genres were. Based off of our top artists, it is not hard to accept this output, as most of the top artists fall into these genres. Pop was by far the leader in our list, with hip hop, R&B, and Dance/Electronic following close behind. I plotted this list out after this cell for some visual comparison. 


```python
genre = music_df['genre'].value_counts()
genre.nlargest(20)
```




    pop                               386
    hip hop, pop                      259
    hip hop, pop, R&B                 225
    pop, Dance/Electronic             206
    pop, R&B                          161
    hip hop                           115
    hip hop, pop, Dance/Electronic     73
    rock                               55
    rock, pop                          36
    Dance/Electronic                   36
    rock, metal                        35
    pop, rock                          26
    pop, latin                         25
    latin                              15
    hip hop, Dance/Electronic          15
    hip hop, pop, latin                14
    R&B                                13
    pop, rock, Dance/Electronic        12
    pop, rock, metal                   12
    country                             9
    Name: genre, dtype: int64



### Visualization
Now that we have seen some basic numbers, lets get some visuals.

Below I plotted out the list of top genres in the data frame. As we know Pop, Hip Hop, R&B, and Dance/Electronic are the top genres by a longshot. Rock barely even comes close in comparison. I used a simple .plot graph to display these values against one another. 



```python
#define plot size
plt.figure(figsize = (16,10))

#using .value_counts() on our genre column to return the number of unique entries,
#sort them by ascending, and then plot them on a bar chart
sorted_genre = music_df['genre'].value_counts().sort_values(ascending = True).plot(kind = 'barh')
sorted_genre
```




    <AxesSubplot:>




    
![png](/assets/Music_exp_pics/Music_Exploration_29_1.png)
    


In the next cell, I made a list of the average song length based on year. To my surprise, there was actually decent correlation between these two. (I made a heat map later of correlations between variables, which is why I decided to come back and visualize this, I thought it was quite interesting.) I think the reason for this could be an increase in streaming platforms. Most artists get money from streams on popular platforms, Spotify being one of them. Earning money on these platforms comes from full listens, so shorter songs with an increased amount on an album could warrant a better monetary gain. But this is complete speculation on my part and can not be inferred completely from this output. I have head this theory from others online, but once again, we cannot confirm that this theory is true from the information we have here.




```python
#making new dataframe from 'duration_ms' and 'year', then grouping by year, 
#applying mean, and resting index
song_length = music_df[['duration_ms','year']].groupby('year').mean().reset_index()
song_length['minutes'] = song_length['duration_ms'].apply(to_min)
song_length['total_seconds'] = song_length['duration_ms']/100
song_length = song_length.drop(['duration_ms'],axis = 1)

song_length
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
      <th>year</th>
      <th>minutes</th>
      <th>total_seconds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000</td>
      <td>4:10</td>
      <td>2509.512817</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>4:02</td>
      <td>2428.715189</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2002</td>
      <td>4:13</td>
      <td>2534.734167</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>3:56</td>
      <td>2367.429111</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2004</td>
      <td>3:53</td>
      <td>2332.580000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2005</td>
      <td>3:57</td>
      <td>2372.730722</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2006</td>
      <td>3:59</td>
      <td>2394.295556</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2007</td>
      <td>3:51</td>
      <td>2316.606163</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2008</td>
      <td>3:53</td>
      <td>2335.350787</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2009</td>
      <td>3:54</td>
      <td>2342.343846</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2010</td>
      <td>3:40</td>
      <td>2208.941837</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2011</td>
      <td>3:48</td>
      <td>2288.227340</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2012</td>
      <td>3:43</td>
      <td>2237.989633</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2013</td>
      <td>3:51</td>
      <td>2313.380120</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2014</td>
      <td>3:38</td>
      <td>2187.258602</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2015</td>
      <td>3:40</td>
      <td>2200.791512</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2016</td>
      <td>3:40</td>
      <td>2205.384023</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2017</td>
      <td>3:31</td>
      <td>2113.828692</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2018</td>
      <td>3:28</td>
      <td>2089.547917</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2019</td>
      <td>3:17</td>
      <td>1973.481047</td>
    </tr>
  </tbody>
</table>
</div>




```python
#defining figure size, then using regplot to create linear regression plot
plt.figure(figsize = (10,5))
lengthPlot = sns.regplot(x= 'year', y = 'total_seconds',  data = song_length)
```


    
![png](/assets/Music_exp_pics/Music_Exploration_32_0.png)
    


As we can tell from our graph, the average length of top hit songs is slowly decreasing. I sadly could not figure out a way to properly get my minutes column attached to the side of the y-axis. Since it was a combo of string and int, most plotting functions didn't like it, so I decided to keep it in seconds. Regardless, I still wanted to visually convey the point that the average song length is decreasing as the years move forward.

Next, I thought it would be good to show some density plots in order to get a visual of what our top hit variables look like overall. I plotted 9 charts, each showing the density and KDE of a different numerical variable from our data frame.


```python
#using distplot brings up an error, this blocks it
import warnings
warnings.filterwarnings('ignore')

#defining plot matrix and overall size
fig, ax = plt.subplots(3,3, figsize=(24,15))

#creating 9 plots from music_df columns and plotting on the subplot using ax[][] for coordinates
sns.distplot(music_df['speechiness'],     color = "b", kde =True, ax = ax[0][0])
sns.distplot(music_df['valence'],         color = 'r', kde =True, ax = ax[0][1])
sns.distplot(music_df['instrumentalness'],color = 'g', kde =True, ax = ax[0][2])
sns.distplot(music_df['acousticness'],    color = "y", kde =True, ax = ax[1][0])
sns.distplot(music_df['tempo'],           color = 'k', kde =True, ax = ax[1][1])
sns.distplot(music_df['danceability'],    color = 'm', kde =True, ax = ax[1][2])
sns.distplot(music_df['key'],             color = 'purple', kde =True, ax = ax[2][0])
sns.distplot(music_df['liveness'],        color = 'orange', kde =True, ax = ax[2][1])
sns.distplot(music_df['energy'],          color = 'c', kde =True, ax = ax[2][2])

plt.show()
```


    
![png](/assets/Music_exp_pics/Music_Exploration_34_0.png)
    


From the graphs, above we can see what makes a song a top hit. Speechiness, Instrumentalness, and Acousticness are all very low, which indicates that most of the songs from the data frame do not contain these features. Energy, Danceability and Valance were all measured very highly, showing that a top song is going have these kinds of features more frequently. Remember that Valance is measured from 0 being 'sad' and 1 being 'happy', so our top hits have a solid overall spread of 'mood' if you will. Our Tempo shows that most of the top hits are either around 125 or slightly below 100 BPM. I think this shows that Tempo and Energy go well together, one might think that this would hold true for danceability as well, but if we look below at the next plot, we will see Danceability and Tempo have a slightly negative correlation. 


```python
#getting correlations between numerical variables
corr = music_df.corr()
#using this variable to be the 'mask' of the heatmap plot
matrix = np.triu(corr)

#defining size, plotting our corr() function, masking upper half
#annot gives us the labels, and fmt makes the labels 2 decimals long. 
plt.figure(figsize=(15,8))
sns.heatmap(music_df.corr(), mask=matrix, annot = True, fmt = '.2f')
```




    <AxesSubplot:>




    
![png](/assets/Music_exp_pics/Music_Exploration_36_1.png)
    


I first wanted to see if our popularity variable correlated with anything, however, since this list of data already has the most popular songs over the years, none of the attributes correlated to strongly with it. I decided to map out all of the variables, to show if any others were related in some way, since simply looking at strictly popularity yeilded little results for me. 

First we have year and duration, as we saw earlier, the length of songs are decreasing as time goes on, giving a negative correlation. Loudness and energy also have a strong correlation, this makes sense as a loud song will probably be more energentic, and vice versa. Danceability and valence have a positive correlation, meaning a more 'danceable' song will probably have a better mood to it. This also applies to our valence and energy correlation, as a more energitic song will likely have a higher mood to it. 

One observation that stood out to me was that speechiness had a slight positive correlation with excplicitness. This possibly might be from hip hop songs, as hip hop tends to have more spoken word and swearing in it, but we cannot assume this is the only reason why of course. 

Another interesting find is that, year and valence have a slightly negative correlation. Could this mean that top hit songs have gotten sadder over the years? We cannot assume this as a fact, but this correlation did surprise me a little. 

## Conclusion
Well, this concludes my paper, I believe I have explored what makes a top hit enough to probably go and make one myself. In total, we need:

**1. High danceability-** Good rythm, maybe a tempo of around 120 BPM, and good energy.

**2. Little to no "instruments"-** No guitars, drums, etc., use music software.

**3. Small amount of spoken words-** Keep it simple, straight to the point, maybe a good chorus and make the vocals "part of the song".

**4. Recorded in a studio-** This top hits list had little to no live performances in it. 

**5. Needs to be either neutral of happy sounding-** Upbeat in general most likely.

**6. Roughly 3-4 minutes long** The top hit average song length was 3:48.

**7. Genre should be Pop, Hip Hip, or EDM.**

Congratulations, now you can make your own top hit!

Overall, I am happy with how this project turned out. I had some slight hiccups and needed to go back and repeat things, but it gave me the oppourtunity to re-evaluate and properly clean and then present the data with what I believe to be little or no errors. I did enjoy exploring this data, I thought it was very cool to see what the general population likes to listen too and what attributes were associated with these popular songs. I would like to see how this data was collected or even how some of these variables were measured because the numbers that were in it did seem accurate in terms of what I would 'rate' them as, yet I don't whether to embrace and accept them as is. I will likely remain skeptical with the data, as it's never a bad thing to be skepitcal when it comes to science and analysis, especially when you don't know how the data was collected or measured. I'm sure there is someone else out there who could do a much better job of analyzing or even presenting/interpreting variables such as these. 

