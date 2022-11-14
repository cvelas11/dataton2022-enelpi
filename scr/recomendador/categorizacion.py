
# !pip install spacy
# !python -m spacy download es_core_news_lg
# !pip install guidedlda
# !python3 -m pip install --upgrade pip
# !pip install nltk
# !pip install sentence_transformers


# Este script implementa un procedimiento adhoc para detectar la presencia del
# nombre de una empresa en una noticia, tres modelos para establecer si la
# noticia habla sobre el sector al que pertenece la empresa y combina 4 modelos
# para clasificar las noticias según su contenido en 8 categorías
# El código sigue el estándar PEP8 en un 95% (no se respeta longitud de línea)

from helper import cleaning_sentence, get_dictionary, getting_n_grams
import pandas as pd
import guidedlda
import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# nltk.download('stopwords')
from nltk.corpus import stopwords
import torch
from sentence_transformers import SentenceTransformer, util
import pickle


def unique(nit, entity):
    if entity not in clientes_dict:
        return entity
    else:
        if clientes_dict[entity]['count'] < 2:
            return entity
    return False


def cliente_predict(i):
    response = 'No Cliente'
    nit = i[1].nit
    title = i[1].nombre
    texto = i[1].texto_titulo
    NER_text = NER(texto[0:min(100000, len(texto))].strip())
    candidatos_entity = NER_text.ents
    n_grama_title = getting_n_grams(cleaning_sentence(title))
    for cand in candidatos_entity:
        cand_cleaned = cand.text.lower().strip()
        if cand_cleaned in n_grama_title:
            if unique(nit, cand_cleaned):
                response = cand.text
    return response


def incluye_sector(sector, topico):
    topicos = clusters_de_sectores[sector]
    if topico in topicos:
        return True
    return False


def embedding(texto, fuente):
    # Se descartan noticias o textos de sector muy cortos
    if (len(texto) < 350 and fuente == 'noticia') or (len(texto) < 6 and fuente == 'sector'):
        return []
    else:
        texto_sen = texto.split('.')
        texto_sen = [x for x in texto_sen if len(x) > 20]
        return model.encode(texto_sen, device='cuda')


NER = spacy.load("es_core_news_lg")
NER.max_length = 15000000

data_folder = "../Data"

file_noticias = data_folder + "/noticias.csv"
file_clientes = data_folder + "/clientes.csv"
file_clientes_noticias = data_folder + "/clientes_noticias.csv"
file_output = data_folder + "/categorizacion.csv"

# Se crean dataframes con los datos
clientes = pd.read_csv(file_clientes)
noticias = pd.read_csv(file_noticias)
clientes_noticias = pd.read_csv(file_clientes_noticias)

# Contiene Cliente: Esta parte determina si las noticias mencionan
# explícitamente a los clientes asociados. Se hace un preentrenamiento para
# encontrar palabras que están en los nombres pero no sirven para identificar
# los clientes y luego se usa el módulo de Named Entity Recognition de spacy
# para identificar las entidades de cada texto. Estas entidades se comparan con
# los n-gramas que se pueden formar con el nombre del cliente y luego se
# descartan los que a pesar de coincidir con un n-grama, están como que pueden
# identificar a varios clientes. Si luego de esos filtros, el entity aún no ha
# sido  filtrado, se considera que el texto sí menciona al cliente


# Esta parte crea un diccionario con ngramas de los nombres de los clientes e
# identifica cuales de ellos definen de manera única a los clientes
cleaned_names = [(x, cleaning_sentence(x)) for x in clientes.nombre]
clientes_dict = get_dictionary(clientes)
noticias['texto_titulo'] = noticias['news_title'] + ' ' + noticias['news_text_content']

# Se crea una base que contiene > 74k filas en las que cada cliente
# con las noticias asociadas, se concatena el título de la noticia con el texto
# y todos los campos referentes al sector se concatenan en uno solo
lgc_noticias_texto = noticias.merge(clientes_noticias,
                                    how='inner', on='news_id')
lgc_noticias_texto = lgc_noticias_texto.merge(clientes, how='inner', on='nit')
# lgc_noticias_texto['texto_titulo'] = lgc_noticias_texto['news_title'] + ' ' + lgc_noticias_texto['news_text_content']
lgc_noticias_texto['sectores'] = lgc_noticias_texto.desc_ciiu_division + '-' + lgc_noticias_texto.desc_ciuu_grupo + '-' + lgc_noticias_texto.desc_ciiuu_clase + '-' + lgc_noticias_texto.subsec


# Se predice para cada noticia-cliente si la noticia menciona explícitamente
# al cliente y se guarda el resultado


cliente = [cliente_predict(i) for i in lgc_noticias_texto.iterrows()]
lgc_noticias_texto['contiene_cliente'] = cliente
lgc_noticias_texto.to_csv(data_folder + '/noticias_texto.csv')


# LDA Guiado para Sectores: Para determinar si los textos mencionan el sector
# al que pertenee el cliente se usa un sistema de votación entre tres modelos.
# El primero de ellos es un LDA. Se corre un LDA y se clasifica cada
# texto, luego se analizan las dos categorías generadas por el LDA más comunes
# en cada sector. Si una noticia es catalogada en una de esas tres categorías
# se clasifica como que sí menciona al sector

# Entrenamiento
corpus = set(lgc_noticias_texto.texto_titulo)
corpus = set([x.strip().lower() for x in corpus])
vectorizer = CountVectorizer(max_df=0.2, min_df=15, stop_words=stopwords.words('spanish'),
                             token_pattern="[^\W\d_]+", ngram_range=(1, 1))
X = vectorizer.fit_transform(corpus)
vocab = vectorizer.get_feature_names()
vocab_set = set(vocab)


model = guidedlda.GuidedLDA(n_topics=100, n_iter=100, random_state=7,
                            refresh=20)
model.fit(X)

# Se imprimen los tópicos para interpretarlos
topic_word = model.topic_word_
n_top_words = 25
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


# Inferencia para toda la base
X = vectorizer.transform([x.lower().strip()
                         for x in noticias.texto_titulo])
X_transformed = model.transform(X)
topics = [i.argmax() for i in X_transformed]
noticias['topic'] = topics
noticias.to_csv(data_folder + '/noticias_con_topic.csv')

data = pd.read_csv(data_folder + '/noticias_texto.csv')
noticias_con_topic = pd.read_csv('noticias_con_topic.csv')
data_topic = data.merge(noticias_con_topic, how='left', on='news_id')
data_topic['sectores'] = data_topic.desc_ciiu_division + ' ' + data_topic.desc_ciuu_grupo + ' ' + data_topic.desc_ciiuu_clase + ' ' + data_topic.subsec


sectores = set(data_topic.sectores)
clusters_de_sectores = {}
for i in sectores:
    vc = data_topic[data_topic.sectores == i].topic.value_counts()[0:2]
    indexes = vc.index
    values = vc.values
    clusters_de_sectores[i] = {indexes[i]: values[i]
                               for i in range(len(indexes))}
contiene_sector = [incluye_sector(x[1].sectores, x[1].topic)
                   for x in data_topic.iterrows()]
data_topic['contiene_sector'] = contiene_sector
data_topic['sector_lda'] = data_topic['contiene_sector']
data_topic.to_csv(data_folder + '/noticias_con_cliente_sector.csv')
data_topic = pd.read_csv(data_folder + '/noticias_con_cliente_sector.csv')


# Categorías: Para determinar la categoría a la que pertenece una noticia se
# hizo un sistema de votación entre cuatro métodos.


# LDA Guíado: Se corre un LDA y se itera varias veces
# mejorando en cada iteración las semillas que definen una categoría
# el texto se clasifica según la categoría a la que es más probable que
# pertenezca

# Entrenamiento
corpus = set(lgc_noticias_texto.texto_titulo)
corpus = set([x.strip().lower() for x in corpus])
vectorizer = CountVectorizer(max_df=0.2, min_df=15, stop_words=stopwords.words('spanish')
                             , token_pattern="[^\W\d_]+", ngram_range=(1, 1))
X = vectorizer.fit_transform(corpus)
vocab = vectorizer.get_feature_names()
vocab_set = set(vocab)


# Nota: Estas semillas se construyen con al menos tres iteraciones de LDA simple
seed_topic_list = [
    ['macroeconomia', 'economia', 'precios', 'inflacion', 'crecimiento',
     'dolar'],
    ['regulaciones', 'ley', 'leyes', 'reforma', 'decreto', 'reforma',
     'legislacion', 'tributaria'],
    ['sostenibilidad', 'ambiente', 'emisiones', 'sostenible'],
    ['innovacion', 'tecnologia', 'digital', 'patente', 'desarrollo', 'investigacion'],
    ['alianza', 'fusion', 'adquisiciones', 'opa', 'alianzas'],
    ['ranking', 'reputacion', 'investigacion', 'multa', 'sancion', 'penalizacion',
     'condena', 'lavado', 'narcotrafico', 'narcotraficante', 'paramilitar',
     'terrorismo', 'condena'],
    ['salud'], ['deporte'], ['cultura'], ['insfraestructura'],
    ['universidad', 'educacion', 'programa'], [''], ['']]
seed_topic_list = [list(set(a).intersection(vocab_set))
                   for a in seed_topic_list]


word2id = dict((v, idx) for idx, v in enumerate(vocab))
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id

model = guidedlda.GuidedLDA(n_topics=13, n_iter=100, random_state=7,
                            refresh=20)
model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)


# Antes de clasificar cada documento en una categoría se deben interpretar los
# factores que provee el LDA guiado

topic_word = model.topic_word_
n_top_words = 25
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

noticias['texto_titulo'] = noticias['news_title'] + ' ' + noticias['news_text_content']
X = vectorizer.transform([x.lower().strip()
                          for x in noticias.texto_titulo])
X_transformed = model.transform(X)
categoria = [i.argmax() for i in X_transformed]

noticias = pd.read_csv(data_folder + '/noticias.csv')
noticias['categoria_'] = categoria

categorias = {0: 'Macroeconomia', 1: 'Regulacion', 2: 'Sostenibilidad',
              3: 'Innovacion', 4: 'Reputacion', 5: 'Otra', 6: 'Otra',
              7: 'Otra', 8: 'Otra', 9: 'Otra', 10: 'Otra', 11: 'Otra',
              12: 'Otra'}
noticias['categoria_texto'] = [
    categorias[x] for x in noticias.categoria_]
data = pd.read_csv(data_folder + '/noticias_con_cliente_sector.csv')
data = data.merge(noticias[['news_id', 'categoria_texto']], how='left', on='news_id')


# Se establece participación usando el método de LDA junto con la determinación
# de si la noticia contiene o no el cliente.
participacion = []
for i in data.iterrows():
    if i[1].contiene_cliente != 'No Cliente':
        participacion.append('Cliente')
    elif i[1].contiene_sector:
        participacion.append('Sector')

    else:
        participacion.append('No Aplica')


data['participacion'] = participacion
data['categoria'] = categoria
data['equipo'] = 'enelpi'
data_topic_sector_categoria = data[['equipo', 'nit', 'news_id', 'participacion', 'categoria', 'contiene_cliente', 'sector_lda']]
data_topic_sector_categoria.to_csv(data_folder + '/sector_cliente_categoria_lda.csv')


# Embeddings párrafos: El segundo método usado para determinar la categoría de
# un texto es el de usar modelos de embeddings pre entrenados que derivan del
# model BERT. En este caso creamos unas semillas que representan a las categorías
# de interés y se calcula la similaridad (cosine) de sus embedding con los
# embeddings de cada uno de los párrafos del texto. Se asigna la noticia a la
# categoría con la que tiene más similaridad. Nótese que se crean categorías
# adicionales que se encontraron en las iteraciones, política, salud, deporte, etc.


# Se iniciliza el modelo y las semillas
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
clases = ['macroeconomia', 'sostenibilidad', 'innovacion', 'regulacion', 'alianzas', 'reputacion', 'otra', 'otra', 'otra', 'otra', 'otra', 'otra', 'otra']
semillas = ['macroeconomia economia precios inflacion crecimiento dolar desempleo', 'sostenibilidad ambiente emisiones sostenible carbon', 'innovacion tecnologia investigacion patente digital', 'regulacion leyes decreto tributaria', 'alianzas fusion adquisicion adquirir compraventa', 'reputacion sancion multa condena investigacion narcotrafico lavado de activos contraloria procuraduria', 'Deporte', 'Politica', 'Educacion', 'Justicia', 'Salud', 'Tecnologia', 'Infraestructura']

# Se calculan los embeddings de las semillas y de todos los párrafos de todos
# los textos. En esta parte también se calculan los embeddings de las columnas
# que hcaen referencia al sector para el segundo y tercer modelo de sector

# Se útiliza GPUs para los embeddings. Estos embeddings corren en 12 minutos con
# GPUs mientras que tardan más de 30 horas con CPUs
semillas_encod = model.encode(semillas, device='cuda')
sectores = set(lgc_noticias_texto.sectores)
sec_embed = {x: embedding(x, 'sector') for x in sectores}
noticias_embed = {x[1].news_id: embedding(x[1].texto_titulo, 'noticia') for x in noticias.iterrows()}

# Se guardan los embeddings en pickles para no tener que calcularlos de nuevo
with open('noticias_embed.pkl', 'wb') as f:
    pickle.dump(noticias_embed, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('sec_embed.pkl', 'wb') as f:
    pickle.dump(sec_embed, f, protocol=pickle.HIGHEST_PROTOCOL)


# se cargan los embeddings
with open('sec_embed.pkl', 'rb') as handle:
    sec_embed = pickle.load(handle)
with open('noticias_embed.pkl', 'rb') as h:
    noticias_embed = pickle.load(h)

# En esta parte se corre el segundo modelo de sector, que consiste en calcular
# la similitud entre cada una de las columnas referentes al sector y los párrafos
# de cada noticia. Si hay más de 5 combinaciones de sector-párrafo cuya similaridad
# es superior a 0.4, se considera que sí se menciona al sector.
sector_2 = []
for row in lgc_noticias_texto.iterrows():
    noticia = row[1].news_id
    sector = row[1].sectores
    if len(noticias_embed[noticia]) > 0 and len(sec_embed[sector]) > 0:
        c = util.cos_sim(sec_embed[sector] , noticias_embed[noticia])
        if np.sum(list(c.flatten() > 0.4)) > 5:
            sector_2.append('Sector')
        else:
            sector_2.append('No Aplica')
    else:
        sector_2.append('No Aplica')

# En esta parte se corre el tercer modelo de sector, que consiste en calcular
# la similitud entre cada una de las columnas referentes al sector y los párrafos
# de cada noticia. Si hay más de 3 combinaciones de sector-párrafo cuya similaridad
# es superior a 0.5, se considera que sí se menciona al sector.
sector_3 = []
for row in lgc_noticias_texto.iterrows():
    noticia = row[1].news_id
    sector = row[1].sectores
    if len(noticias_embed[noticia]) > 0 and len(sec_embed[sector]) > 0:
        c = util.cos_sim(sec_embed[sector] , noticias_embed[noticia])
        if np.sum(list(c.flatten() > 0.5)) > 3:
            sector_3.append('Sector')
        else:
            sector_3.append('No Aplica')
    else:
        sector_3.append('No Aplica')

# Se carga de nuevo la categoría creada con LDA y se transforma de True y False
# a Sector y No Aplica para que los tres métodos sean homogéneos
# Se añaden los resultados de los otros dos métodos al data frame y se guarda
data = pd.read_csv(data_folder + '/sector_cliente_categoria_lda.csv')
data.sector_lda = ['No Aplica' if x is False else 'Sector' for x in data.sector_lda]
data['participacion_lda'] = data['participacion']
data['sector_2'] = sector_2
data['sector_3'] = sector_3
data.to_csv(data_folder + '/doble_sector_cliente_categoria_lda.csv')

# Aquí se aplica el método de párrafos para la definición del sector, se promedia
# la similitud de los embeddings de cada categoría con los párrafos y se escoge
# la de mayor promedio

clasificacion = []
cont = 0
for tex in noticias.iterrows():
    noticia = tex[1].news_id
    if len(noticias_embed[noticia]) > 0:
        c = util.cos_sim(semillas_encod, noticias_embed[noticia])
        d = np.array([torch.mean(x) for x in c])
        clasi = clases[d.argmax()]
        clasificacion.append(clasi)
    else:
        clasificacion.append('otra')
noticias['clasificacion_parrafos'] = clasificacion

# Se agrega a la clasificacion lda la clasificacion de párrafos y se pone
# en minúscula para hacerla compatible con otras clasificaciones para el sistema
# de votación
data = pd.read_csv(data_folder + '/doble_sector_cliente_categoria_lda.csv')
data = data.merge(noticias[['news_id', 'clasificacion_parrafos']], how='left' , on='news_id')
categoria = [x.lower() if x.lower() != 'descartable' else 'otra' for x in data.categoria]
data['categoria'] = categoria
data.to_csv(data_folder + '/doble_sector_cliente_categoria_lda_categoria_parrafos.csv')

# Se cargan los resultados de los modelos de embeddings con n-grams usando diferentes
# encoders y ses unen a las demás predicciones de categoría
xlm = pd.read_csv(data_folder + '/final_xlm_all_results.csv')
data = pd.read_csv(data_folder + '/doble_sector_cliente_categoria_lda_categoria_parrafos.csv')
data = data.merge(xlm, how='left', left_on='news_id', right_on='id_news')

# Se define la categoría del texto basado en una votación en la que tiene prelación
# la clasificación denominada párrafos ya que en varios muestreos demostró ser la mejor.
# El sistema de votación mejora en 20% los resultados comparados con usar párrafos únicamente
categoria_votacion = []
for row in data.iterrows():
    a = pd.Series([row[1].categoria_simple, row[1].categoria, row[1].categoria_guiada, row[1].clasificacion_parrafos]).value_counts()
    if len(a) == 1:
        categoria_votacion.append(a.keys()[0])
    elif a[0] == a[1]:
        categoria_votacion.append(row[1].clasificacion_parrafos)
    else:
        categoria_votacion.append(a.keys()[0])
data['categoria_votacion'] = categoria_votacion

# Aquí se determina si la noticia contiene el sector o no mediante un sistema de votación entre LDA y dos
# modelos de embeddings con diferentes hiperparámetros. Se asigna Sector o No Aplica según la mayoría (2 de 3)
data = data.merge(lgc_noticias_texto[['nit', 'news_id', 'texto_titulo', 'sectores']], how='left', on=['news_id', 'nit'])
contiene_sector = []
for row in data.iterrows():
    if len(row[1].texto_titulo) > 350 and len(row[1].sectores) > 5:
        a = pd.Series([row[1].sector_lda, row[1].sector_2, row[1].sector_3]).value_counts()
        contiene_sector.append(a.keys()[0])
    else:
        contiene_sector.append('No Aplica')
data['contiene_sector'] = contiene_sector

# Se define la columna participación mezclando Contiene Cliente con el sector obtenido anteriormente
participacion = []
for row in data.iterrows():
    if row[1].participacion_lda == 'Cliente':
        participacion.append('Cliente')
    else:
        participacion.append(row[1].contiene_sector)

data['participacion'] = participacion

# Se aplica la lógica de Descartable a la categoría para hacerla la categoría final
categoria_final = []
for i in data.iterrows():
    if i[1].participacion == 'No Aplica' and i[1].categoria_votacion == 'otra':
        categoria_final.append('descartable')
    elif i[1].categoria_votacion in clases:
        categoria_final.append(i[1].categoria_votacion)
    else:
        categoria_final.append('otra')
data['categoria_final'] = categoria_final

# Se guarda un archivo de interés para revisión y muestreso
data[['equipo', 'nit', 'news_id', 'participacion', 'categoria',
      'clasificacion_parrafos', 'id_news', 'contiene_cliente', 'contiene_sector', 'categoria_simple',
      'categoria_guiada', 'categoria_votacion', 'participacion_lda', 'sector_2' , 'sector_3', 'categoria_final']].to_csv(data_folder + '/doble_sector_cliente_categoria_final.csv', index=False)
data = pd.read_csv(data_folder + '/doble_sector_cliente_categoria_final.csv')
to_review = data.merge(lgc_noticias_texto[['nit', 'news_id', 'texto_titulo', 'sectores']], how='left', on=['news_id', 'nit'])

# Se guarda archivo para ser usado en el algoritmo de recomendación
to_review[['nit', 'news_id', 'contiene_cliente', 'contiene_sector', 'categoria_final', 'participacion']].to_csv('ouput_para_ranking.csv', index='False')

data['nombre_equipo'] = 'enelpi'
# Se guarda el archivo final de categorización
data[['nombre_equipo', 'nit', 'news_id', 'participacion', 'categoria']].to_csv(data_folder + '/categorizacion.csv')
