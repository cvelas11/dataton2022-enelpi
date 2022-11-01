# Este script implementa un procedimiento adhoc para detectar la presencia del
# nombre de una empresa en una noticia, un LDA para identificar si la noticia
# habla sobre el sector al que pertenece la empresa y un LDA guiado para
# clasificar las noticias según su contenido en 8 categorías

from helper import cleaning_sentence, get_dictionary, getting_n_grams
import pandas as pd
import guidedlda
import spacy

# Contiene Cliente: Esta parte determina si las noticias mencionan
# explícitamente a los clientes asociados. Se hace un preentrenamiento para
# encontrar palabras que están en los nombres pero no sirven para identificar
# los clientes y luego se usa el módulo de Named Entity Recognition de spacy
# para identificar las entidades de cada texto. Estas entidades se comparan con
# los n-gramas que se pueden formar con el nombre del cliente y luego se
# descartan los que a pesar de coincidir con un n-grama, están como que pueden
# identificar a varios clientes. Si luego de esos filtros, el entity aún no ha
# sido  filtrado, se considera que el texto sí menciona al cliente


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
    NER_text = NER(texto[0:min(400000, len(texto))].strip())
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


NER = spacy.load("es_core_news_lg")
NER.max_length = 15000000

# Se crean dataframes con los datos
clientes = pd.read_csv('clientes.csv')
noticias = pd.read_csv('noticias.csv')
clientes_noticias = pd.read_csv('clientes_noticias.csv')

# Esta parte crea un diccionario con ngramas de los nombres de los clientes e
# identifica cuales de ellos definen de manera única a los clientes
cleaned_names = [(x, cleaning_sentence(x)) for x in clientes.nombre]
clientes_dict = get_dictionary(clientes)

# Se crea una base que contiene > 74k filas en las que cada cliente
# con las noticias asociadas, se concatena el título de la noticia con el texto
# y todos los campos referentes al sector se concatenan en uno solo
lgc_noticias_texto = noticias.merge(clientes_noticias,
                                    how='inner', on='news_id')
lgc_noticias_texto = lgc_noticias_texto.merge(clientes, how='inner', on='nit')
lgc_noticias_texto['texto_titulo'] = lgc_noticias_texto['news_title'] + ' ' + lgc_noticias_texto['news_text_content']
lgc_noticias_texto['sectores'] = lgc_noticias_texto.desc_ciiu_division + ' '
+ lgc_noticias_texto.desc_ciuu_grupo + ' '
+ lgc_noticias_texto.desc_ciiuu_clase + ' ' + lgc_noticias_texto.subsec

# Se predice para cada noticia-cliente si la noticia menciona explícitamente
# al cliente y se guarda el resultado
cliente = [cliente_predict(i) for i in lgc_noticias_texto.iterrows()]
lgc_noticias_texto['contiene_cliente'] = cliente
lgc_noticias_texto.to_csv('noticias_texto.csv')

# LDA Guiado para Sectores: Para determinar si los textos mencionan el sector
# al que pertenee el cliente se usa un LDA guiado
# (Aún no es guíado, en construcción). Se corre un LDA y se clasifica cada
# texto, luego se analizan las tres categorías generadas por el LDA más comunes
# en cada sector. Si una noticia es catalogada en una de esas tres categorías
# se clasifica como que sí menciona al sector

# Entrenamiento
corpus = set(lgc_noticias_texto.texto_titulo)
corpus = set([x.strip().lower() for x in corpus])
vectorizer = CountVectorizer(max_df=0.2, min_df=15, stop_words=stopwords,
                             token_pattern="[^\W\d_]+", ngram_range=(1, 1))
X = vectorizer.fit_transform(corpus)
vocab = vectorizer.get_feature_names()
vocab_set = set(vocab)
model = guidedlda.GuidedLDA(n_topics=100, n_iter=100, random_state=7,
                            refresh=20)
model.fit(X)

# Inferencia para toda la base
X = vectorizer.transform([x.lower().strip()
                         for x in lgc_noticias_texto.texto_titulo])
X_transformed = model.transform(X)
topics = [i.argmax() for i in X_transformed]
data = pd.read_csv('noticias_texto.csv')
data['topic'] = topics
data.to_csv('noticias_texto_topic.csv')
data_topic = pd.read_csv('noticias_texto_topic.csv')
data_topic['sectores'] = data_topic.desc_ciiu_division + ' ' + data_topic.desc_ciuu_grupo + ' ' + data_topic.desc_ciiuu_clase + ' ' + data_topic.subsec

sectores = set(data_topic.sectores)
clusters_de_sectores = {}
for i in sectores:
    vc = data_topic[data_topic.sectores == i].topic.value_counts()[0:3]
    indexes = vc.index
    values = vc.values
    clusters_de_sectores[i] = {indexes[i]: values[i]
                               for i in range(len(indexes))}


data_topic['contiene_sector'] = [incluye_sector(x[1].sectores, x[1].topic)
                                 for x in data_topic.iterrows()]
data_topic.to_csv('noticias_texto_topic_sector.csv')


# LDA Guíado para categorías: Para clasificar las noticias en las 8 categorías
# se usa también un LDA guíado. Se corre un LDA y se itera varias veces
# mejorando en cada iteración las semillas que definen una categoría
# el texto se clasifica según la categoría a la que es más probable que
# pertenezca

# Entrenamiento
corpus = set(lgc_noticias_texto.texto_titulo)
corpus = set([x.strip().lower() for x in corpus])
vectorizer = CountVectorizer(max_df=0.2, min_df=15, stop_words=stopwords
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
    ['innovacion', 'tecnologia', 'digital', 'patente'],
    ['alianza', 'fusion', 'adquisiciones', 'opa', 'alianzas'],
    ['reputacion', 'investigacion', 'multa', 'sancion', 'penalizacion',
     'condena', 'lavado', 'narcotrafico', 'narcotraficante', 'paramilitar',
     'terrorismo'],
    ['salud'], ['deporte'], ['cultura'], ['insfraestructura'],
    [''], [''], ['']]
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
noticias = pd.read_csv('noticias.csv')
noticias['categoria_'] = categoria
noticias['categoria_texto'] = [
    categorias(x) for x in noticias.categoria_]
data = pd.read_csv('noticias_texto_topic_sector.csv')
data = data.merge(noticias[['news_id', 'categoria_texto']], how='left', on='news_id')
categorias = {0: 'Macroeconomia', 1: 'Regulacion', 2: 'Sostenibilidad',
              3: 'Innovacion', 4: 'Reputacion', 5: 'Otra', 6: 'Otra',
              7: 'Otra', 8: 'Otra', 9: 'Otra', 10: 'Otra', 11: 'Otra',
              12: 'Otra'}
participacion = []
categoria = []

for i in data.iterrows():
    if i[1].contiene_cliente != 'No Cliente':
        participacion.append('Cliente')
        categoria.append(i[1].categoria_texto)
    elif i[1].contiene_sector:
        participacion.append('Sector')
        categoria.append(i[1].categoria_texto)

    else:
        participacion.append('No Aplica')
        if i[1].categoria_texto == 'Otra':
            categoria.append('Descartable')
        else:
            categoria.append(i[1].categoria_texto)


data['participacion'] = participacion
data['categoria'] = categoria
data['equipo'] = 'enelpi'
data_topic_sector_categoria = data[['equipo', 'nit', 'news_id', 'participacion', 'categoria']]
data_topic_sector_categoria.to_csv('categorizacion.csv')
