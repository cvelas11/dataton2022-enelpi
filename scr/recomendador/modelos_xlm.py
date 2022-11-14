# %%capture
# #Installs some requirements
# !pip install --upgrade pip
# !pip install keybert
# !python -m nltk.downloader stopwords
# !pip install sentence_transformers
# !pip install tensorflow_hub
# !pip install tensorflow_text
# !pip install umap-learn

import pandas as pd
import csv
import copy
import numpy as np
from numpy.linalg import norm
import re
import operator
import functools
import collections
from sklearn.feature_extraction.text import CountVectorizer
import nltk.corpus as nltk
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
from keybert import KeyBERT


# Este script implementa un procedimiento 2 de los 4 modelos
# para clasificar las noticias según su contenido en 8 categorías. Los modelos
# aquí entrenados son los que usan keyBert como base


# loads the embedding model
module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'
model = hub.load(module_url)
data_folder = "../Data"


def embed_text(input):
    # Creates the function to embed text
    return model(input)


# Loads the model to use for KEYBert
sentence_model_xlm = SentenceTransformer("sentence-transformers/paraphrase-xlm-r-multilingual-v1")
# Calls the model to be loaded
kw_model = KeyBERT(sentence_model_xlm)
# Files path
file_noticias = data_folder + "/noticias.csv"
file_stopwords = data_folder + "/stopwords_file.csv"
# Reads the data
df_noticias = pd.read_csv(file_noticias)

# Creates a list for the news_ids and the news content
articles_list = df_noticias['news_text_content'].tolist()
articles_id = df_noticias['news_id'].tolist()


# Creates vector of words for the article
vectorizer = CountVectorizer()
vectorizer_fit = vectorizer.fit_transform(articles_list)

# Gets the words vectorized and the frequency for the words
# Later creates a dictionary to join both lists
words = vectorizer.get_feature_names()
count = list(np.asarray(vectorizer_fit.sum(axis=0))[0])
words_count = dict(zip(words, count))

# Creates a list of the low frequency words
# This are words that occurr one time in all the news corpus
words_low_freq = [k for k, v in words_count.items() if v < 2]
print(len(words_low_freq))

# Clean the data
# Print text without stopwords
clean_articles = []
for i, article in enumerate(articles_list):
    id_news = articles_id[i]
    # remove all dots from numbers
    article = re.sub(r'(?<=\d)\.(?=\d)', r"", article).lower()
    # removes all numbers
    numb_pattern = r'[0-9]'
    clean_article = re.sub(numb_pattern, "", article)
    clean_article = re.sub('[^A-Za-z0-9]+', ' ', clean_article)
    clean_article = re.sub(" +", " ", clean_article)
    clean_articles.append((id_news, clean_article))


# Creates the stoword list fron nltk library
stoplist_nltk = nltk.stopwords.words('spanish')

# Creates a stopword list from csv file found
# This was needed to make sure we remove more stopwords
my_file = open(file_stopwords, 'r')
reader = csv.reader(my_file)
regular_list = list(reader)
stoplist_csvfile = [item for sublist in regular_list for item in sublist][1:]
# Concatenates the two lists
stoplist_all = stoplist_nltk + stoplist_csvfile + words_low_freq
# Creates a dataframe with the news content clean
df_noticias_clean = pd.DataFrame(clean_articles, columns=['news_id', 'news_clean_text'])
# Joins the data with the curent df_noticias
df_noticias = pd.merge(df_noticias_clean, df_noticias, on='news_id')

# Creates an array of the categories
categorias = ['Macroeconomia', 'Sostenibilidad', 'Innovacion', 'Regulacion', 'Alianzas', 'Reputacion', 'Deporte', 'Politica', 'Educacion', 'Justicia', 'Salud', 'Tecnologia', 'Infraestructura']

# Embedding the categorias
categoria_embedding = embed_text(categorias)

# creates the empty list for all items of article
articles_output = []
for n in range(len(df_noticias)):
    # print(' ****** Articulo numero: ', n , ' procesado ******* ')
    items_article = []
    # URL News
    noticia_url = df_noticias.iloc[n]['news_url_absolute']
    # id_news
    id_news = df_noticias.iloc[n]['news_id']
    # Get noticia title
    noticia_title = df_noticias.iloc[n]['news_title'].lower()
    # Get the content for the news
    noticia_content_clean = df_noticias.iloc[n]['news_clean_text']
    noticia_content = df_noticias.iloc[n]['news_text_content']

    count_words_noticia = len(noticia_content.split(' '))
    # Creates the items to build a list with article properties
    items_article.insert(0, id_news)
    # id_news
    # If the news has a length less than 30 words is flagged as 'Descartable'
    # Otherwise, we pass it to the KEYBert Model
    if count_words_noticia <= 30 | count_words_noticia >= 10000:
        items_article.insert(3, 'Descartable')
        # Categoria
        print("------ CLASIFICADO COMO: ", 'Descartable', '\n')

    else:

        # Extracts the keywords using the XLM previously loaded without seeds
        keywords_simple = kw_model.extract_keywords(noticia_content_clean,
                                                    keyphrase_ngram_range=(1, 2),
                                                    stop_words=stoplist_all,
                                                    highlight=False)
        # Extracts the keywords using the XLM previously loaded
        keywords_guiaded = kw_model.extract_keywords(noticia_content_clean,
                                                     keyphrase_ngram_range=(2, 3),
                                                     stop_words=stoplist_all,
                                                     highlight=False,
                                                     seed_keywords=categorias)
        # Creates a list of the keywords
        keywords_list_simple = list(dict(keywords_simple).keys())
        keywords_list_guiaded = list(dict(keywords_guiaded).keys())

        # Creates the embeddings for the KEYBert Results
        keyword_simple_embedding = embed_text(keywords_list_simple)
        keyword_guiaded_embedding = embed_text(keywords_list_guiaded)

        # Creates a dictionary to append the cosine similarity by category
        cosine_similarity_categoria_dict_simple = dict()
        cosine_similarity_categoria_dict_guiaded = dict()

        # List for dictionary embeddings
        keyword_simple_cate_dic = []
        keyword_guiaded_cate_dic = []

        # Measures the cosine similarity between categories and key n-grams
        # Does this for simple and guiaded KeyBert
        for k, categoria in enumerate(categorias):
            cosine_sum_simple = 0
            for i in range(len(keyword_simple_embedding)):
                cosine_article_line_category = np.dot(keyword_simple_embedding[i], categoria_embedding[k]) / (norm(keyword_simple_embedding[i]) * norm(categoria_embedding[k]))
                cosine_sum_simple += cosine_article_line_category
            cosine_similarity_categoria_dict_simple.update({categoria : cosine_sum_simple})
            cosine_sum_guiaded = 0
            for i in range(len(keyword_guiaded_embedding)):
                cosine_article_line_category = np.dot(keyword_guiaded_embedding[i], categoria_embedding[k]) / (norm(keyword_guiaded_embedding[i]) * norm(categoria_embedding[k]))
                cosine_sum_guiaded += cosine_article_line_category
            cosine_similarity_categoria_dict_guiaded.update({categoria : cosine_sum_guiaded})

        keyword_simple_cate_dic.append(cosine_similarity_categoria_dict_simple)
        keyword_guiaded_cate_dic.append(cosine_similarity_categoria_dict_guiaded)

        # Orders the dicitionary
        keyword_simple_cate_dic = dict(functools.reduce(operator.add, map(collections.Counter, keyword_simple_cate_dic)))
        keyword_guiaded_cate_dic = dict(functools.reduce(operator.add, map(collections.Counter, keyword_guiaded_cate_dic)))

        # orders the dictionary and get the max item of the dictionary
        # With the Max we classify the article
        keyword_simple_cate_dic = dict(sorted(dict(collections.Counter(keyword_simple_cate_dic)).items(), key=lambda x: x[1], reverse=True))
        max_score_cat_simple = max(keyword_simple_cate_dic.items(), key=operator.itemgetter(1))[0]

        keyword_guiaded_cate_dic = dict(sorted(dict(collections.Counter(keyword_guiaded_cate_dic)).items(), key=lambda x: x[1], reverse=True))
        max_score_cat_guiaded = max(keyword_guiaded_cate_dic.items(), key=operator.itemgetter(1))[0]

        # Saves the categories for the keybert simple and guiaded
        items_article.insert(1, max_score_cat_simple)  # Categoria simple
        items_article.insert(2, max_score_cat_guiaded)  # Categoria simple

    # Saves the output in a list
    articles_output.append(copy.deepcopy(items_article))

# Creates the df with the list
df_news_categorized = pd.DataFrame(articles_output, columns=['id_news', 'Categoria_Simple', 'Categoria_Guiada'])
# Saves df to csv file
df_news_categorized.to_csv(data_folder + '/xlm_all_results.csv', index=False)
