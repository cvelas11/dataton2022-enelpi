# Import libraries
# nltk.download('stopwords')
# nltk.download('punkt')
import pandas as pd
import re
import nltk


# Este script implementa el modelo de recomendación de las noticias mediante un
# procedimieno adhoc que usa cinco reglas de las cuales 3 se basan en datos
# de los modelos de participación y categorización.

data_folder = "../Data"


def stop_words():
    import nltk.corpus as nltk
    import csv
    file_stopwords = data_folder + "/stopwords_file.csv"
    # Creates the stoword list fron nltk library
    stoplist_nltk = nltk.stopwords.words('spanish')

    # Creates a stopword list from csv file found
    # This was needed to make sure we remove more stopwords
    my_file = open(file_stopwords, 'r')
    reader = csv.reader(my_file)
    regular_list = list(reader)
    stoplist_csvfile = [item for sublist in regular_list for item in sublist][1:]
    # Concatenates the two lists
    stoplist_all = stoplist_nltk + stoplist_csvfile

    return stoplist_all


# Loads stopwords list
stop_words = stop_words()


def clean_articles(df):

    from nltk.tokenize import word_tokenize

    # Creates a list for the news_ids and the news content
    articles_list = df['news_text_content'].tolist()
    articles_id = df['news_id'].tolist()
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
        text_tokens = word_tokenize(clean_article)
        if len(text_tokens) < 10000:
            list_clean_words = [word for word in text_tokens if word not in stop_words]
            len_article = len(list_clean_words)
            clean_article = ' '.join(list_clean_words)
        else:
            clean_article = ''
            len_article = 0
        clean_articles.append((id_news, clean_article, len_article))

    # Creates a dataframe with the news content clean
    df_noticias_clean = pd.DataFrame(clean_articles, columns=['news_id', 'news_clean_text', 'news_length'])

    # Joins the data with the curent df_noticias
    df = pd.merge(df_noticias_clean, df, on='news_id')

    return df


# ##### NOTICIAS   ######
# Reads news data
file_noticias = data_folder + "/noticias.csv"
df_noticias = pd.read_csv(file_noticias)
# Creates a column deleting spcial characters and number from the text
# Later, count the words to calculate the length of the news.
df_noticias = clean_articles(df_noticias)

# Creates the list of the source
url_list_split = (df_noticias['news_url_absolute'].str.split('//', n=1, expand=True))
source_list = url_list_split[1].str.split('/', n=1, expand=True)[0]

# Add the source to df
df_noticias['source'] = source_list
# Reads categorizacion output
file_output = data_folder + "/ouput_para_ranking.csv"
df_output = pd.read_csv(file_output)
df_output['categoria'] = df_output['categoria_final']
# OUTPUTBORRAR CON TU OUTPUT

# Converts the output to 1 or 0
df_output.loc[df_output['contiene_cliente'] == 'No Cliente', 'cliente_peso'] = 0
df_output.loc[df_output['contiene_cliente'] != 'No Cliente', 'cliente_peso'] = 1

# Converts the output to 1 or 0
df_output.loc[df_output['contiene_sector'] == 'No Aplica', 'sector_peso'] = 0
df_output.loc[df_output['contiene_sector'] == 'Sector', 'sector_peso'] = 1

# Creates the dictionary with the categories established and their corresponding weights
dic_weigth_catgorias = {'reputacion': 2.1, 'macroeconomia': 2, 'innovacion': 1.7, 'sostenibilidad': 1.8, 'regulacion': 1.6, 'alianzas': 1.4, 'otra': 0.5, 'descartable': 0}
# Gets the max value of the dictionary to later be used to normalized
max_value = max(dic_weigth_catgorias.values())

# Iterates over the dictionary to assigned the weight to the dataframe
for key, val in dic_weigth_catgorias.items():
    norm_value = round(val / max_value, 2)
    dic_weigth_catgorias[key] = [norm_value]

# Creates a df from the dictionary normnalized with the categories
df_categoria_w = pd.DataFrame().from_dict(dic_weigth_catgorias, orient='index', columns=['categoria_peso'])
df_categoria_w.reset_index(inplace=True)
df_categoria_w = df_categoria_w.rename(columns={'index': 'categoria'})
# Joins the dictionary df with the df output
df_output = pd.merge(df_output, df_categoria_w, on='categoria')
# Joins df to get the columns used for ranking
# TODO -----------------------------------------------------  'nit'
df_ranking = pd.merge(df_noticias[['news_id', 'news_length', 'source', 'news_text_content']], df_output[['news_id', 'nit', 'cliente_peso', 'sector_peso', 'categoria_peso', 'categoria', 'participacion']], on='news_id')
# Creates the score for the source
# Groups to get the source count by categoria
df_source_cat_source = df_ranking.groupby(['source', 'categoria'], as_index=False).agg({'news_id': 'count'})
df_max_source_cat = df_source_cat_source.rename(columns={'news_id': 'total_source'})
# Joins to df_ranking
df_ranking = pd.merge(df_ranking, df_max_source_cat, on=['categoria', 'source'])
# Gets the max by categoria
df_max_source_cat = df_source_cat_source.groupby(['categoria'], as_index=False).agg({'news_id': 'max'})
df_max_source_cat = df_max_source_cat.rename(columns={'news_id': 'max_source_categoria'})
df_ranking = pd.merge(df_ranking, df_max_source_cat, on='categoria')
df_ranking['source_peso'] = df_ranking['total_source'] / df_ranking['max_source_categoria']
# Drops the column that are not longer needed
df_ranking = df_ranking.drop(['total_source', 'max_source_categoria'], axis=1)


# Creates the score for the lenght
# Groups to get the source count by categoria
df_source_cat_source = df_ranking.groupby(['news_length', 'categoria'], as_index=False).agg({'news_id': 'count'})
df_max_source_cat = df_source_cat_source.rename(columns={'news_id': 'total_source'})

# Joins to df_ranking
df_ranking = pd.merge(df_ranking, df_max_source_cat, on=['categoria', 'news_length'])
# Gets the max by categoria
df_max_source_cat = df_source_cat_source.groupby(['categoria'], as_index=False).agg({'news_id': 'max'})
df_max_source_cat = df_max_source_cat.rename(columns={'news_id': 'max_source_categoria'})
df_ranking = pd.merge(df_ranking, df_max_source_cat, on='categoria')
df_ranking['noticia_len_peso'] = df_ranking['total_source'] / df_ranking['max_source_categoria']
# Drops the column that are not longer needed
df_ranking = df_ranking.drop(['total_source', 'max_source_categoria'], axis=1)

# Sum the columns to create the ranking
df_ranking.loc[df_ranking['news_length'] >= 31, 'ranking'] = df_ranking['cliente_peso'] + df_ranking['sector_peso'] + df_ranking['categoria_peso'] + df_ranking['source_peso'] + df_ranking['noticia_len_peso']
df_ranking.loc[df_ranking['news_length'] < 31, 'ranking'] = 0
df_ranking.to_csv('ranking.csv', index=False)
df_ranking['recomendacion'] = df_ranking.groupby(by='nit')['ranking'].rank(method='dense', ascending=False)

# Saves the final recomendacion file
df_ranking['nombre_equipo'] = 'enelpi'
df_ranking[['nombre_equipo', 'nit', 'news_id', 'participacion', 'categoria', 'recomendacion']].to_csv(data_folder + '/recomendacion.csv')
