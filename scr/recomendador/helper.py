
from nltk import ngrams
import spacy


NER = spacy.load("es_core_news_lg")
NER.max_length = 15000000


def cleaning_sentence(sentence):
    sentence = sentence + ' '
    sentence = sentence.lower().replace('s.a.s', '').replace(
        's.a', '').replace('ltda', '').replace('y cia', '').replace(
        ' sa ', '').replace(' sas ', '').replace(' s a ', '')
    return sentence.strip()


def getting_n_grams(sentence):
    sentence = sentence.strip().lower()
    n_grams = []
    for j in range(len(sentence.strip().split())):
        n_grams += list(ngrams(sentence.split(), j))
    n_grams = [' '.join(x) for x in n_grams] + [sentence]
    return n_grams


def get_dictionary(titles):
    dict_n_grams = {}
    for sen in titles.iterrows():
        sen_n = cleaning_sentence(sen[1].nombre)
        nit = sen[1].nit
        n_grams = getting_n_grams(sen_n)

        for i in n_grams:
            if i in dict_n_grams:
                dict_n_grams[i]['count'] += 1
                dict_n_grams[i]['nit'] += [nit]
            else:
                dict_n_grams[i] = {}
                dict_n_grams[i]['count'] = 1
                dict_n_grams[i]['nit'] = [nit]
    return dict_n_grams
