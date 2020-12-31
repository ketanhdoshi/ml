import pandas as pd

# lets create a list of songs.
songs = ['In the name of love','Scream','Till the sky falls down','In and out of Love']
# lets also create a list of corresponding artists. FYI: 'MG' stands # for Martin Garrix, 'TI' for Tiesto, 'DB' for Dash Berlin, 'AV'for # Armin Van Buuren.
artists = ['MG','TI','DB','AV']
# likewise lets create a dictionary that contains artists and songs.
song_arts = {'MG':'In the name of love','TI':'Scream','DB':'Till the sky falls down','AV':'In and out of Love'}

# create a Series object whose data is coming from songs list.
ser_num = pd.Series(data=songs)
print(ser_num)
# get the element that corresponds to index 3.
print (ser_num[3])

# make artists the index this time.
ser_art = pd.Series(data=songs,index=artists)
# Just pass the name of the artist and you get their song
print(ser_art)
print(ser_art['MG'], ser_art['AV'], ser_art['DB'])
# even numbers still work as index
print (ser_art[0], ser_art[1])


# create a series object from dictionary by passing the dictionary element to pd.Series()
ser_dict= pd.Series(song_arts)
print(ser_dict)
print (ser_dict[['MG', 'DB']])

# make a group assignment
ser_dict[['MG', 'DB']] = 'wow'

# assign a name to the series and the index. And change the index labels
ser_dict.name = 'my series name'
ser_dict.index.name = 'my index name'
ser_dict.index = ['A', 'B', 'C', 'D']
print(ser_dict)

# get the indices only, or get only values of the series
print (ser_art.index, ser_art.values)

# filter elements in the series and do maths operations
my_series2 = pd.Series([5, 6, 7, 8, 9, 10], index=['a', 'b', 'c', 'd', 'e', 'f'])
print(my_series2[my_series2 > 7])
print(my_series2[my_series2 > 7] * 2)

# create a dataframe - each column is a series object
df = pd.DataFrame({
    'country': ['Kazakhstan', 'Russia', 'Belarus', 'Ukraine'],
    'population': [17.04, 143.5, 9.5, 45.5],
    'square': [2724902, 17125191, 207600, 603628]})
print(df)
print(df['country'])
print(df.index, df.columns)

# add an index to the dataframe
df.index = ['KZ', 'RU', 'BE', 'UK']
df.index.name = 'Country Code'
print(df)

# access a row using the index label or index number
print(df.loc['KZ'])
print(df.iloc[1])

# select particular rows and columns
print(df.loc[['KZ', 'RU'], 'population'])

# slice the dataframe using a row index list and a column list
print(df.loc['KZ':'BE', :]) 

# filter the dataframe
# df.population and df[‘population’] are the same operations
print(df[df.population > 10][['country', 'square']])

# add a new column for population density
df['density'] = df['population'] / df['square'] * 1000000
print(df)

# delete the column
df.drop(['density'], axis='columns')

# rename a column
df = df.rename(columns={'population': 'pop'})
print(df)

# Group by
ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'Kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)
print (df)
print (df.groupby(['Year', 'Team']) ['Points'].sum())

# pivot table
pvt = df.pivot_table(index=['Year'], columns=['Team'], values='Points', aggfunc='sum')
print(pvt)

# time series data
df = pd.read_csv('apple-stock-price.csv', index_col='Date', parse_dates=True)
df = df.sort_index()
print(df.loc['2012-Feb', 'Close'].mean())   # average closing price in Feb
print(df.loc['2012-Feb':'2015-Feb', 'Close'].mean())    # average closing price between time periods
df.resample('W')['Close'].mean()    # average closing price by week

# visualisation
import matplotlib.pyplot as plt
new_df = df.loc['2012-Feb':'2017-Feb', ['Close']]
new_df.plot()
plt.show()