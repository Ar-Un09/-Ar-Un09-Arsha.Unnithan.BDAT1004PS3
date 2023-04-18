#!/usr/bin/env python
# coding: utf-8

# <h1><center> <font color='brown'> BDAT 1004 – Data Programming</font></center></h1>
# <h2><center> Problem Set 3</center></h2>
# <h2><center> Prepared by Arsha Unnithan (200546119)</center></h2>

# <h3><font color='darkblue'>Question 1 - Occupations </h3>

# <i>Step 1. Import the necessary libraries

# In[6]:


import pandas as pd
import numpy as np


# <i>Step 2. Import the dataset from this address.

# In[7]:


data = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user'


# <i>Step 3. Assign it to a variable called users.

# In[8]:


users = pd.read_csv(data, sep='|', index_col='user_id')
users.head(10)


# <i>Step 4. Discover what is the mean age per occupation 

# In[9]:


users[["occupation","age"]].groupby("occupation").mean()


# <i> Step 5. Discover the Male ratio per occupation and sort it from the most to the least

# In[20]:


users_df = pd.DataFrame(users)

gender_counts = users_df.pivot_table(index='occupation', columns='gender', values='age', aggfunc='size', fill_value=0)

total_counts = gender_counts[['F', 'M']].sum(axis=1)

result['male_ratio'] = gender_counts['M'] / total_counts * 100

result.sort_values('male_ratio',axis=0,ascending=False)


# <i>Step 6. For each occupation, calculate the minimum and maximum ages

# In[11]:


users.groupby('occupation')['age'].agg(['min', 'max'])


# <i>Step 7: For each combination of occupation and sex, calculate the mean age

# In[71]:


pd.pivot_table(users,index=['occupation'], columns= ['gender'], values = 'age',   aggfunc={'age' : np.mean}).fillna(0)


# <i>Step 8. For each occupation present the percentage of women and men

# In[95]:


percentage = pd.pivot_table(users, index=['occupation'], columns=['gender'], values='age', aggfunc='count').fillna(0)
percentage['Male'] = (percentage['M'] / (percentage['F'] + percentage['M']) * 100)
percentage['Female'] = (percentage['F'] / (percentage['F'] + percentage['M']) * 100)
percentage.iloc[:, [2, 3]]


# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------

# <h3><font color='darkblue'>Question 2 - Euro Teams </h3>

# <i>Step 1. Import the necessary libraries
#     
# <i>Step 2. Import the dataset from this address

# In[101]:


data = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv'


# <i>Step 3. Assign it to a variable called euro12

# In[106]:


euro12 = pd.read_csv(data)
euro12.head(10)


# <i>Step 4. Select only the Goal column:

# In[111]:


euro12[['Goals']]


# <i>Step 5. How many team participated in the Euro2012?

# In[115]:


teams_count = euro12['Team'].count()
print("Number of teams participated :", teams_count)


# <i>Step 6. What is the number of columns in the dataset?

# In[116]:


columns_count = len(euro12.columns)
print("Number of columns in the dataset:", columns_count)


# <i>Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline.

# In[130]:


discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]
discipline


# <i>Step 8. Sort the teams by Red Cards, then to Yellow Cards

# In[140]:


discipline.sort_values(['Red Cards', 'Yellow Cards'], ascending=[True, True])


# <i>Step 9. Calculate the mean Yellow Cards given per Team

# In[139]:


euro12.groupby('Team').agg({'Yellow Cards': 'mean'})


# <i> Step 10. Filter teams that scored more than 6 goals

# In[142]:


euro12[euro12['Goals'] > 6]


# <i>Step 11. Select the teams that start with G

# In[144]:


euro12[euro12['Team'].str.startswith('G')]


# <i>Step 12. Select the first 7 columns

# In[145]:


euro12.iloc[:, :7]


# <i>Step 13. Select all columns except the last 3

# In[146]:


euro12.iloc[:, :-3]


# <i>Step 14. Present only the Shooting Accuracy from England, Italy and Russia

# In[147]:


euro12.loc[euro12['Team'].isin(['England', 'Italy', 'Russia']), ['Team', 'Shooting Accuracy']]


# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------

# <h3><font color='darkblue'>Question 3 - Housing </h3>

# <i> Step 1. Import the necessary libraries
# 
# Step 2. Create 3 differents Series, each of length 100, as follows:     
# <i> • The first a random number from 1 to 4 
#     • The second a random number from 1 to 3 
#     • The third a random number from 10,000 to 30,000  

# In[160]:


series_1 = pd.Series(np.random.randint(1, 5, 100))
series_2 = pd.Series(np.random.randint(1, 4, 100))
series_3 = pd.Series(np.random.randint(10000, 30001, 100))


# <i>Step 3. Create a DataFrame by joining the Series by column

# In[172]:


df = pd.concat([series_1, series_2, series_3], axis=1)
df


# <i>Step 4. Change the name of the columns to bedrs, bathrs, price_sqr_meter

# In[176]:


df.columns = ['bedrs', 'bathrs', 'price_sqr_meter']
df


# <i>Step 5. Create a one column DataFrame with the values of the 3 Series and assign it to 'bigcolumn'

# In[183]:


bigcolumn = pd.concat([series_1, series_2, series3], axis=0, ignore_index=True)
bigcolumn = pd.DataFrame(bigcolumn)
bigcolumn


# <i>Step 6. Ops it seems it is going only until index 99. Is it true?

# In[184]:


print(bigcolumn.index)


# <i>Step 7. Reindex the DataFrame so it goes from 0 to 299

# In[190]:


bigcolumn = bigcolumn.reset_index(drop=True)
bigcolumn.index = pd.RangeIndex(start=0, stop=300, step=1)
bigcolumn


# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------

# <h3><font color='darkblue'>Question 4 - Wind Statistics </h3>

# <i>Step 1. Import the necessary libraries

# In[191]:


import pandas as pd
import numpy as np
import csv
import datetime


# <i>Step 2. Import the dataset from the attached file wind.txt

# In[194]:


data = pd.read_csv('wind.txt', sep='\s+', parse_dates=[[0,1,2]])
data


# <i>Step 3. Assign it to a variable called data and replace the first 3 columns by a proper datetime index.

# In[195]:


data = data.set_index('Yr_Mo_Dy')
data


# <i>Step 4. Year 2061? Do we really have data from this year? Create a function to fix it and apply it.

# In[199]:


def fix_year(x):
    year = x.year - 100 if x.year > 1989 else x.year
    return pd.Timestamp(year=year, month=x.month, day=x.day)
    
data.index = data.index.map(fix_year)
data


# <i>Step 5. Set the right dates as the index. Pay attention at the data type, it should be datetime64[ns].
#    

# In[200]:


data.index = pd.to_datetime(data.index)
data


# <i>Step 6. Compute how many values are missing for each location over the entire record.They should be ignored in all calculations below.

# In[236]:


print (f'Missing values in each location is')
data.isna().sum()


# <i>Step 7. Compute how many non-missing values there are in total.

# In[237]:


print (f'Non-missing values in each location is')
data.notnull().sum()


# In[238]:


print (f'Total number of non-missing values is', data.notnull().sum().sum())


# <i>Step 8. Calculate the mean windspeeds of the windspeeds over all the locations and all the times. A single number for the entire dataset.

# In[239]:


mean_wind_speed = data.mean().mean()
print (f"The mean windspeeds of the windspeeds over all the locations and all the times is ", mean_wind_speed)


# <i>Step 9. Create a DataFrame called loc_stats and calculate the min, max and mean windspeeds and standard deviations of the windspeeds at each location over all the days A different set of numbers for each location. 

# In[240]:


loc_stats = pd.DataFrame({
    'min': data.min(),
    'max': data.max(),
    'mean': data.mean(),
    'std': data.std()
})
loc_stats


# <i> Step 10. Create a DataFrame called day_stats and calculate the min, max and mean windspeed and standard deviations of the windspeeds across all the locations at each day. A different set of numbers for each day. 

# In[241]:


day_stats = pd.DataFrame({
    'min': data.min(axis=1),
    'max': data.max(axis=1),
    'mean': data.mean(axis=1),
    'std': data.std(axis=1)
})
day_stats


# <i>Step 11. Find the average windspeed in January for each location. Treat January 1961 and January 1962 both as January. 

# In[242]:


january_data = data[data.index.month == 1]
january_data.mean()


# <i>Step 12. Downsample the record to a yearly frequency for each location. 

# In[243]:


yearly_data = data.resample('Y').mean()
yearly_data


# <i>Step 13. Downsample the record to a monthly frequency for each location.

# In[244]:


monthly_data = data.resample('M').mean()
monthly_data


# <i>Step 14. Downsample the record to a weekly frequency for each location. 

# In[245]:


weekly_data = data.resample('W').mean()
weekly_data


# <i>Step 15. Calculate the min, max and mean windspeeds and standard deviations of the windspeeds across all locations for each week (assume that the first week starts on January 2 1961) for the first 52 weeks.

# In[255]:


week_stats = data.resample('W', closed='left', label='left').agg(['min', 'max', 'mean', 'std'])[:52*7]
week_stats = week_stats.iloc[1:53]
week_stats


# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------

# <h3><font color='darkblue'>Question 5 - Chipotle </h3>

# <i>Step 1. Import the necessary libraries

# In[4]:


import pandas as pd
import numpy as np


# <i>Step 2. Import the dataset from this address

# In[12]:


url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'


# <i>Step 3. Assign it to a variable called chipo. 

# In[13]:


chipo = pd.read_csv(url, delimiter='\t')


# <i>Step 4. See the first 10 entries 

# In[7]:


chipo.head(10)


# <i>Step 5. What is the number of observations in the dataset?

# In[8]:


#The rows represent the observations and the columns represent the variables. 
print(len(chipo.index))


# <i>Step 6. What is the number of columns in the dataset?

# In[9]:


#The rows represent the observations and the columns represent the variables.
print(len(chipo.columns))


# <i>Step 7. Print the name of all the columns.

# In[10]:


for columns in chipo.columns:
    print(columns)


# <i>Step 8. How is the dataset indexed?

# In[13]:


print(chipo.index)


# <i>Step 9. Which was the most-ordered item? 

# In[14]:


item_count = chipo.groupby('item_name')['quantity'].sum()
item_count.sort_values(ascending=False).head(1)


# <i>Step 10. For the most-ordered item, how many items were ordered?

# In[15]:


most_ordered_item = item_count.sort_values(ascending=False).head(1).index[0]
quantity_ordered = item_count[most_ordered_item]
print("The most ordered item was", most_ordered_item)
print("The number of items ordered was", quantity_ordered)


# <i>Step 11. What was the most ordered item in the choice_description column?

# In[16]:


choice_description_count = chipo.groupby('choice_description')['quantity'].sum()
most_ordered_choice = choice_description_count.sort_values(ascending=False).head(1).index[0]
print("The most ordered item in the choice_description column was", most_ordered_choice)


# <i>Step 12. How many items were ordered in total?

# In[17]:


total_items_ordered = chipo['quantity'].sum()
print("The total number of items ordered was", total_items_ordered)


# <i> Step 13.• Turn the item price into a float • Check the item price type • Create a lambda function and change the type of item price • Check the item price type 

# In[357]:


chipo.item_price.dtype


# In[358]:


print(chipo['item_price'].dtype)


# In[15]:


chipo['item_price'] = chipo['item_price'].apply(lambda x: float(x[1:]) if isinstance(x, str) else x)
print(chipo['item_price'].dtype)


# In[17]:


chipo['item_price'] = chipo['item_price'].apply(lambda x: float(x[1:]) if isinstance(x, str) and len(x) > 1 else x)
print(chipo['item_price'].dtype)


# <i>Step 14. How much was the revenue for the period in the dataset?

# In[20]:


revenue = (chipo['quantity'] * chipo['item_price']).sum()
print("The revenue for the period was $", round(revenue, 2))


# <i>Step 15. How many orders were made in the period? 

# In[21]:


orders = chipo['order_id'].nunique()
print("The number of orders made in the period was", orders)


# <i>Step 16. What is the average revenue amount per order?

# In[22]:


average_revenue_per_order = revenue / orders
print("The average revenue amount per order was $", round(average_revenue_per_order, 2))


# <i>Step 17. How many different items are sold? 

# In[23]:


unique_items = chipo['item_name'].nunique()
print("The number of different items sold was", unique_items)


# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------

# <h3><font color='darkblue'>Question 6 - Line Plot </h3>

# <i>Create a line plot showing the number of marriages and divorces per capita in the U.S. between 1867 and 2014. Label both lines and show the legend. Don't forget to label your axes!

# In[48]:


import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Loading the data into a pandas dataframe
df = pd.read_csv("us-marriages-divorces-1867-2014.csv")

# Step 2: Setting the 'Year' column as the index
df = df.set_index('Year')

# Step 3: Calculating the marriage and divorce rates per capita. 
#Per capita is calculated by dividing the total number of marriages or divorces in U.S by its total population.
df['Marriage Rate'] = df['Marriages'] / df['Population']
df['Divorce Rate'] = df['Divorces'] / df['Population']

# Step 4: Creating the line plot
plt.plot(df.index, df['Marriage Rate'], label='Marriage Rate Per Capita', color='brown')
plt.plot(df.index, df['Divorce Rate'], label='Divorce Rate Per Capita', color='orange')

# Step 5: Adding labels and legend
plt.xlabel('Year')
plt.ylabel('Per Capita Rates')
plt.title('Marriage and Divorce Rates in the U.S. (1867-2014)')
plt.legend()

# Step 6: Displaying the plot
plt.show()


# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------

# <h3><font color='darkblue'>Question 7 - Vertical Bar Chart </h3>

# <i>Create a vertical bar chart comparing the number of marriages and divorces per capita in the U.S. between 1900, 1950, and 2000. Don't forget to label your axes! 

# In[54]:


import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Loading the data into a pandas dataframe
df = pd.read_csv("us-marriages-divorces-1867-2014.csv")

# Step 2: Extracting the rows corresponding to the years 1900, 1950, and 2000
df = df[df['Year'].isin([1900, 1950, 2000])]

# Step 3: Calculating the marriage and divorce rates per capita.
# Per capita is calculated by dividing the total number of marriages or divorces in U.S by its total population.
df['Marriage Rate'] = df['Marriages'] / df['Population']
df['Divorce Rate'] = df['Divorces'] / df['Population']

# Step 4: Creating the vertical bar chart
ax = df.plot(x='Year', y=['Marriage Rate', 'Divorce Rate'], kind='bar', width=0.8, figsize=(8, 6),color=['#483D8B', '#E6E6FA'])

# Step 5: Adding labels and legend
ax.set_xlabel('Year')
ax.set_ylabel('Per Capita Rate')
ax.set_title('Marriage and Divorce Rates in the U.S. (1900, 1950, 2000)')
ax.legend(['Divorce Rate Per Capita', 'Marriage Rate Per Capita'])

# Step 6: Displaying the plot
plt.show()


# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------

# <h3><font color='darkblue'>Question 8 - Horizontal Bar Chart </h3>

# <i> Create a horizontal bar chart that compares the deadliest actors in Hollywood. Sort the actors by their kill count and label each bar with the corresponding actor's name. Don't forget to label your axes!

# In[69]:


import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Loading the data into a pandas dataframe
df = pd.read_csv("actor_kill_counts.csv")

# Step 2: Sorting the dataframe by 'Count' column in descending order
df = df.sort_values(by='Count', ascending=False)

# Step 3: Creating the horizontal bar chart
ax = df.plot(x='Actor', y='Count', kind='barh', figsize=(10,8), color='#20B2AA')

# Step 4: Adding labels to the chart
ax.set_xlabel('Kill Count')
ax.set_ylabel('Actor')
ax.set_title('Deadliest Actors in Hollywood')

# Step 5: Labeling each bar with the corresponding actor's name
for i, v in enumerate(df['Count']):
    ax.text(v + 1, i, str(v), color='maroon', fontsize=12)

# Step 6: Displaying the plot
plt.show()


# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------

# <h3><font color='darkblue'>Question 9 - Pie Chart </h3>

# <i>Create a pie chart showing the fraction of all Roman Emperors that were assassinated.  Make sure that the pie chart is an even circle, labels the categories, and shows the percentage breakdown of the categories. 

# In[64]:


import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Loadig the data into a pandas dataframe
df = pd.read_csv("roman-emperor-reigns.csv")

# Step 2: Calculating the number of emperors that were assassinated and the number that were not
# this code ignores all other causes of death and only considers them as assasinated and not assassinated
assassinated = df[df['Cause_of_Death'] == 'Assassinated']['Emperor'].count()
not_assassinated = df[df['Cause_of_Death'] != 'Assassinated']['Emperor'].count()

# Step 3: Creating the pie chart
labels = ['Assassinated', 'Other Causes']
sizes = [assassinated, not_assassinated]
colors = ['#8B8386', '#EEE0E5'] 
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')

# Step 4: Adding a title to the chart
plt.title('Fraction of Roman Emperors that were assassinated')

# Step 5: Displaying the plot
plt.show()


# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------

# <h3><font color='darkblue'>Question 10 - Scatter Plot </h3>

# <i> Create a scatter plot showing the relationship between the total revenue earned by arcades and the number of Computer Science PhDs awarded in the U.S. between 2000 and 2009.  Don't forget to label your axes! Color each dot according to its year. 

# In[67]:


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

# Step 1: Load the data into a pandas dataframe
df = pd.read_csv("arcade-revenue-vs-cs-doctorates.csv")

# Step 2: Create the scatter plot
sb.scatterplot(x="Total Arcade Revenue (billions)", y="Computer Science Doctorates Awarded (US)", hue="Year", data=df)

# Step 3: Add labels to the plot
plt.title("Relationship between Arcade Revenue and CS Doctorates Awarded")
plt.xlabel("Total Arcade Revenue (billions)")
plt.ylabel("Computer Science Doctorates Awarded (US)")

# Step 4: Display the plot
plt.show()


# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
