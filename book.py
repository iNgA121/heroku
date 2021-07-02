# Import
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from IPython.core.display import HTML

st.write("DB username:", st.secrets["db_username"])
st.write("DB password:", st.secrets["db_password"])
# Load ratings
ratings = st.cache(pd.read_csv)('BX-Book-Ratings.csv',encoding='cp1251', sep=';')
ratings = ratings[ratings['Book-Rating']!=0]

# Load books
col_names = ['ISBN','Book-Title','Book-Author','Year-Of_Publication','Publisher','Image-URL-S','Image-URL-M','Image-URL-L']
books = st.cache(pd.read_csv)('BX-Books.csv', encoding='cp1251', sep =';', names=col_names, dtype='unicode',skiprows=[0])

# Users_ratigs = pd.merge(ratings, users, on=['User-ID'])
dataset = pd.merge(ratings, books, on=['ISBN'])
dataset_lower = dataset.apply(lambda x: x.str.lower() if x.name in ['Book-Title', 'Book-Author'] else x, axis=0)

st.title('BOOK RECOMMENDATION') 
st.subheader('This simple app allows you to select and visualize the chart of recommended books')


a = ['Searching ...']
b = dataset_lower['Book-Title'].unique().tolist()
menu = a + b

book = st.selectbox('', menu)

# Specific User-ID's for selected book
if book != menu[0]:
  readers = dataset_lower['User-ID'][dataset_lower['Book-Title']==book]
  readers = np.unique(readers.tolist())
  st.write('**You selected: **', book)
else:
  st.write('**Empty list, please select your favourite book**')

# Final dataset
if book != menu[0]:
  all_books_readers = dataset_lower[(dataset_lower['User-ID'].isin(readers))]

# Book occurence through readers, general similarity
  col =['User-ID','Book-Title','Book-Author']
  add_column = all_books_readers.copy(deep=False)
  add_column['Column'] = add_column[col].apply(lambda x: ' '.join(x.values.astype(str)), axis=1)

# Numbering and droping duplicates
  df = add_column.drop_duplicates(subset=['Book-Title'])
  df.insert(0,'Index', np.arange(len(df)))
  data = list(df['Column'])

# Count the number of texts in created column
  cv = CountVectorizer()
  count_matrix = cv.fit_transform(data)
  cosine_sim = cosine_similarity(count_matrix)


# Define list of values cosine similarity with appropriate book indexes
  def get_index_from_title(title):
    return df[df['Book-Title']==book]["Index"].values[0]
  book_index = get_index_from_title(book)

# Allocation book indexes to cosine values
  similar_books = list(enumerate(cosine_sim[book_index]))

# Sorted list and 20 most similiar books
  sorted_similar_books = sorted(similar_books, key=lambda x:x[1], reverse=True)
  def get_title_from_index(index):
    return df[df['Index'] == index]['Book-Title'].values[0]

  mylist = []
  i=0
  for book in sorted_similar_books:
    mylist.append(get_title_from_index(book[0]))
    i+=1
    if i>20:
      break

# Create empty lists
  book_titles = []
  avgrating = []
  url_s = []

  for title in mylist:
    book_titles += [title]
    rating = all_books_readers[all_books_readers['Book-Title']==title].groupby(all_books_readers['Book-Title']).mean().round(2)
    rating = rating['Book-Rating'].min()
    url = all_books_readers[all_books_readers['Book-Title']==title]
    url = url['Image-URL-M'].values[0]
    avgrating += [rating]
    url_s += [url]

# Final dataframe  
  union = pd.DataFrame(zip(book_titles, avgrating, url_s), columns=['Book_title','Avg_rating','URL'])

# Edit image url format 
  def path_to_image_html(path):
    return '<img src="'+ path + '" >'
  union['URL']=path_to_image_html(union['URL'])
else:
  print('END')

# Rendering the dataframe as HTML table
if book != menu[0]:
  st.header('The most favorable books')
  final = HTML(union.to_html(escape=False, formatters=dict(path_to_image_html=[])))
  st.write(final)
