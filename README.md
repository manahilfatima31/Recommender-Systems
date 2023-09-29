# Recommender-Systems
pip install plotly

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string

# working on first 10000 rows
zomato_real=pd.read_csv(r"C:\Desktop\zomato.csv", nrows=10000)
zomato_real.head()

# Deleting Unnnecessary Columns
zomato=zomato_real.drop(['url','dish_liked','phone'],axis=1)

zomato.head()

zomato.info()

# Removing the Duplicates
zomato.duplicated().sum()
zomato.drop_duplicates(inplace=True)

zomato.head()

zomato.describe()

#Remove the NaN values from the dataset
zomato.isnull().sum()
zomato.dropna(how='any',inplace=True)

#Changing the column names
zomato = zomato.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'list_type', 'listed_in(city)':'city'})

zomato.head()

# Changing the cost to string
zomato['cost'] = zomato['cost'].astype(str)
# Using lambda function to replace ',' from cost
zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.'))
zomato['cost'] = zomato['cost'].astype(float)

# Removing '/5' from Rates
zomato = zomato.loc[zomato.rate !='NEW']
zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')


# Adjust the column names
zomato.name = zomato.name.apply(lambda x:x.title())
zomato.online_order.replace(('Yes','No'),(True, False),inplace=True)
zomato.book_table.replace(('Yes','No'),(True, False),inplace=True)

zomato.head()

# Computing Mean Rating
restaurants = list(zomato['name'].unique())
zomato['Mean Rating'] = 0

for i in range(len(restaurants)):
    zomato['Mean Rating'][zomato['name'] == restaurants[i]] = zomato['rate'][zomato['name'] == restaurants[i]].mean()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (1,5))
zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)

def clean_text(text):
    """Cleans a text string by removing punctuation, stopwords, and URLs."""

    # Convert the text to lowercase
    text = text.lower()

    # Remove punctuation
    punctuation = string.punctuation
    text = text.translate(str.maketrans('', '', punctuation))

    # Remove stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stopwords])

    # Remove URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)

    return text

zomato["reviews_list"] = zomato["reviews_list"].apply(clean_text)

zomato[['reviews_list', 'cuisines']].sample(5)


def get_top_words(column, top_n, ngram_range=(1, 2)):
    """Returns the top n most frequent words in a column, using n-grams of the specified range.

    Args:
        column: A pandas Series of text.
        top_n: The number of top words to return.
        ngram_range: The range of n-grams to consider.

    Returns:
        A list of tuples, where each tuple contains a word and its frequency.
    """

    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    bag_of_words = vectorizer.fit_transform(column)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_n]

# Drop the columns we don't need
zomato = zomato.drop(['address', 'rest_type', 'list_type', 'menu_item', 'votes'], axis=1)

# Randomly sample 60% of the data
df_percent = zomato.sample(frac=0.6)

# Get the top 10 most frequent words
top_words = get_top_words(df_percent['reviews_list'], 10)

# Print the top 10 most frequent words
print(top_words)


#TF-IDF
df_percent.set_index('name', inplace=True)

indices = pd.Series(df_percent.index)

# Creating tf-idf matrix
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')

tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend(name, cosine_similarities = cosine_similarities):

    # Find the index of the hotel entered
    idx = indices[indices == name].index[0]

    # Find the restaurants with a similar cosine-sim value and order them from bigges number
    top30_indexes = cosine_similarities[idx].argsort()[-30:][::-1]

    # Names of the top 30 restaurants
    recommend_restaurant = list(df_percent.index[top30_indexes])

    # Creating the new data set to show similar restaurants
    df_new = df_percent[['cuisines','Mean Rating', 'cost']].loc[recommend_restaurant]

    # Drop the same named restaurants and sort only the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['cuisines','Mean Rating', 'cost'], keep=False).sort_values(by='Mean Rating', ascending=False).head(10)

    print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR REVIEWS: ' % (str(len(df_new)), name))

    return df_new

recommend('Jalsa')




recommend('Jalsa')

import seaborn as sns
sns.catplot(data=zomato, x='rate', kind='count', aspect=2)
ax = plt.gca()
ax.set(xticks=range(0, 32), title='Rating Distribution')
ax.tick_params('x', labelsize=10)
ax.tick_params('y', labelsize=15)

plt.figure(figsize=(10,5))
ax = plt.gca()
sns.boxplot(data=zomato, x='rate', ax=ax)
plt.xscale('log')

# Create a word cloud
from wordcloud import WordCloud
all_words = ' '.join([text for text in zomato['reviews_list']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

all_words = ' '.join([text for text in zomato['cuisines']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

all_words = ' '.join([text for text in zomato['location']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Finding the locations with the most number of retaurants 

zomato.location.value_counts().nlargest(10).plot(kind='barh')
plt.title("Number of restaurants by location")
plt.xlabel("Restaurant counts")
plt.show()
#From the barchart it can be seen that BTM has the most number of restaurants

# Finding how many people order online 
import plotly.offline as py
trace = go.Pie(labels = ['Online_orders', 'No_online_orders'], values = zomato['online_order'].value_counts(), 
               textfont=dict(size=15), opacity = 0.8,
               marker=dict(colors=['silver','gold'], 
                           line=dict(color='#000000', width=1.5)))


layout = dict(title =  'Distribution of order variable')
           
fig = dict(data = [trace], layout=layout)
py.iplot(fig)
#From the pie chart it can be seen that people order online more than going out



sns.countplot(x = zomato['rate'], hue = zomato['online_order'], palette= 'Set2')
plt.title("Distribution of restaurant rating over online order facility")
plt.show()
#This plot shows that rating clearly depends on the online ordering facility provision, restaurants with online facilities have a higher rating

zomato.name.value_counts().nlargest(20).plot(kind = 'barh')
plt.legend()
plt.show()
#Cafe coffee day is the most popular restaurant in Bangalore

# Plotting a pie chart for online orders

trace = go.Pie(labels = ['Table_booking_available', 'No_table_booking_available'], values = zomato['book_table'].value_counts(), 
               textfont=dict(size=15), opacity = 0.8,
               marker=dict(colors=['silver','gold'], 
                           line=dict(color='#000000', width=1.5)))


layout = dict(title =  'Distribution of order variable')
           
fig = dict(data = [trace], layout=layout)
py.iplot(fig)
#87.5% of the restaurants have online table booking(reservation) facilities

plt.figure(figsize=(20,10))
sns.countplot(x = zomato['online_order'], hue = zomato['rate'], palette= 'Set2')
plt.title("Distribution of restaurant rating over table booking facility")
plt.show()
#The distribution below clearly shows that ratings depend on online table booking facility. The restaurants with the online reservation facility have a higher rating

plt.rcParams['figure.figsize'] = 14,7
plt.subplot(1,2,1)

zomato.name.value_counts().head().plot(kind = 'barh', color = sns.color_palette("hls", 5))
plt.xlabel("Number Of Restaurants")
plt.title("Biggest Restaurant Chain (Top 5)")

plt.subplot(1,2,2)

zomato[zomato['rate'] >= 4.5]['name'].value_counts().nlargest(5).plot(kind = 'barh', color = sns.color_palette("Paired"))
plt.xlabel("Number Of Restaurants")
plt.title("Biggest Restaurant Chain (Top 5) - Rating more than 4.5")
plt.tight_layout()
'''
The bigger chained restaurants in Bangalore do not necessarily have the highest rating. Cafe coffee day has almost 
100 cafes while truffles has just over 40. Truffles has a higher rating than cafe coffee day
Therefore, quality over quantity
'''

sns.heatmap(zomato.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":15})







# Doing some ML stuff


zomato['online_order']= pd.get_dummies(zomato['online_order'], drop_first=True)

zomato['book_table'] = pd.get_dummies(zomato['book_table'], drop_first=True)

# One Hot Encoding

get_dummies_location = pd.get_dummies(zomato.location)
get_dummies_location.head(3)

ml = pd.concat([zomato, get_dummies_location], axis = 1)
ml.head(2)

ml = ml.drop(["name","location", 'cuisines', 'reviews_list', 'city'],axis = 1)
ml.head()

#splitting into indep
x = ml.drop(['rate'], axis = 1)
x.head()

#dependent
y = ml['rate']
y.head()

from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()
model.fit(x,y)

print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()



from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30) #30% split

from sklearn.linear_model import LinearRegression

lr = LinearRegression()



lr.fit(x_train, y_train)

lr_pred = lr.predict(x_test)

r2 = r2_score(y_test,lr_pred)
print('R-Square: ',r2*100)


lr_errors = abs(lr_pred - y_test)
print('Error MEAN ABSOLUTE: ', round(np.mean(lr_pred), 2), ' in degrees')


mape = 100 * (lr_errors / y_test)


lr_accuracy = 100 - np.mean(mape)
print('LR ACCURACY: ', round(lr_accuracy, 2), '%')

sns.distplot(y_test-lr_pred)

plt.figure(figsize=(12,7))

plt.scatter(y_test,x_test.iloc[:,2],color="black")
plt.title("True rate vs Predicted rate, Linear regression",size=20,pad=15)
plt.xlabel('Rating',size = 15)
plt.ylabel('Frequency',size = 15)
plt.scatter(lr_pred,x_test.iloc[:,2],color="yellow")

from sklearn.metrics import mean_absolute_error,mean_squared_error

print('MAE :',metrics.mean_absolute_error(y_test, lr_pred))

print('MSE: ',metrics.mean_squared_error(y_test, lr_pred))


