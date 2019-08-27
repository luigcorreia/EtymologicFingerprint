#!/usr/bin/env python

import sys
import os
import nltk
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer 
import pandas as pd
import re
import collections
import string

re_punctuation = re.compile('[{0}]+'.format(string.punctuation))
def get_brown_documents():

    lemmatizer = WordNetLemmatizer() 
    documents = {}
    for filename in brown.fileids():
        
        # Obter tokens a partir do documento
        tokens = brown.words(filename)
        # Remover stopwords
        stop_words = stopwords.words('english')
        tokens = [word.lower() for word in tokens if word not in stop_words]
        # Remover sinais de pontuação
        tokens = [x for x in tokens if not re_punctuation.fullmatch(x)]

        # Lemmatização
        tokens = list(map(lambda word: lemmatizer.lemmatize(word), tokens))

        # POS Tagging
        #tokens_tagged = nltk.pos_tag(tokens)
        
        documents[filename] = tokens

    return documents

def load_etymology():
    # Carregar dados da etytree
    used_rel_types = ['rel:etymology']
    etymwn = pd.read_csv('documentos/etymwn.tsv', sep='\t')
    etymwn.columns = ['word','relation','parent_word']
    etymwn = etymwn[etymwn['relation'].apply(lambda rel_type: rel_type in used_rel_types)]

    # Definir indice
    etymwn.index = etymwn['word']

    return etymwn


def origin_of(word, etymwn, lang='eng', level=1):
    # Consultar árvore etimologica e extrair idioma ancestral da primeira ocorrência encontrada
    entrie = '{0}: {1}'.format(lang, word)
    try:
        lang, word = etymwn.loc[entrie]['parent_word'].split(': ')
        if level == 1:
            return lang, word
        else:
            return origin_of(word, etymwn, lang=lang, level=level-1)
    except:
        return None, None

def etymological_sig(document, etymwn):

    # Filtrar categoria de POS-Tagging

    word_count = pd.DataFrame()

    for word in document:
        lang, parent_word = origin_of(word, etymwn)
        #print(lang)
        if lang is not None:
            if lang not in word_count:
                word_count[lang] = [0]
            word_count[lang] = word_count[lang] + 1

    return word_count

def generate_sig_dataset(corpus, etymwn):

    sig = pd.DataFrame()
    for name, document in corpus.items():
        sig = sig.append(etymological_sig(document, etymwn))

    # Preencher valores ausentes com zero
    sig = sig.fillna(0)
    # Normalizar por total de palavras
    sig = sig.divide(sig.sum(axis=1),axis=0)
    # Salvar em disco
    return sig


def generate_dataset():

    print("Processando tokens dos documentos do corpus Brown")
    brown_tokens = get_brown_documents()
    print("Carregando árvore etimológica")
    etymwn = load_etymology()
    print("Extraíndo assinatura etimológica dos documentos")
    fingerprints = generate_sig_dataset(brown_tokens, etymwn)
    # Indexar por nome do documento
    fingerprints.index = brown.fileids()
    fingerprints.to_csv('brown_fingerprints.csv')

def get_brown_categories():
    categories = [brown.categories(name)[0] for name in brown.fileids()]
    y = pd.DataFrame({'category': categories}, index=brown.fileids())
    return y
