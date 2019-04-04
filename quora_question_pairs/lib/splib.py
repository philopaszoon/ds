from spacy.symbols import *
import pandas as pd
import numpy as np

from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm

import math
from math import sqrt
from scipy.spatial import distance

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

np_labels = set([nsubj, nsubjpass, dobj, iobj, pobj]) # Probably others too
SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd"] 
MODS = ['advmod', 'prep', 'ccomp'] # added by me
noun_phrase_labels = \
           set(['nsubj', 'nsubjpass', 'dobj', 'iobj', 'pobj', 'attr']) # Probably others too

verb_postags = set(['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ'])
qwords = ['what', 'how', 'who', 'where', 'why', 'when']
nwords = ['NOUN', 'PRON']
emptywords = []

def iter_nps(nlp, sen):
    doc = nlp(sen)
    for word in doc:
        if word.dep in np_labels:
            #yield word.subtre
            for i in word.subtree: print word, word.dep_, '--->', i
            
def full_tree(nlp, sen):
    doc = nlp(sen)
    for word in doc:
        for child in word.subtree: 
            print word, word.dep_, '--->', child
            
def tree_out(nlp, sen):
    doc = nlp(sen)
    senlen = 0
    rootwords = []
    for word in doc:senlen = senlen + 1
    for word in doc:
        ### the tree is all the words that view this word as a head
        treelist = []; 
        for i in word.subtree: treelist.append(i)
        # rl_list = [];  
        # for rl in word.rights: rl_list.append(rl.orth_)
        rootwords.append((word, word.orth_, word.lemma_, word.tag_, word.dep_, word.pos_, word.head, word.head.pos_, treelist))
    df = pd.DataFrame(rootwords)
    df.rename(columns={0:'word', 1:'orth', 2:'lemma', 3:'tag', 4:'dep', 5:'pos', 6:'head', 7:'head_pos', 8:'tree'}, \
    inplace=True)
    return df

def punct_space(token): return token.is_punct or token.is_space

def lemmatized_sentence(data):
    doc = nlp(data)
    for num, sent in enumerate(doc.sents):
        yield u' '.join([token.lemma_ for token in sent if not punct_space(token)])



def not_empty(any_structure):
    if any_structure: return True
    else:             return False
    
def capture_noun_phrase(nlp, sentence):
    doc = nlp(sentence)
    phrases = []
    for np in doc.noun_chunks: phrases.append(np.text)
    return phrases
    


def dep_string(sentence):
    SUBJECT_TYPES = ["nsubj", "pobj", "dobj", "attr"]
    SUBJECT_BACKUPS = [] #['acomp']
    Q1 = ['what', 'which']
    
    #sentence = unicode(sentence, "utf-8")
    doc = nlp(sentence)
    sns = list(doc.sents)
    tagged = []
    #tags.append(("word","tag","pos","dep","lemma","head","left_list","right_list","vec"))
    for word in doc:
        thisword = word.orth_
        
        wl_list = []
        for wl in word.lefts: 
            wl_word = wl.orth_
            if(wl.pos_ in verb_postags): wl_word = wl.lemma_
            wl_list.append(wl_word)
            
        rl_list = []
        for rl in word.rights: 
            rl_word = rl.orth_
            if(rl.pos_ in verb_postags): rl_word = rl.lemma_
            rl_list.append(rl_word)
        
        # get the subtree for the word
        treelist = []
        if word.dep_ in noun_phrase_labels:
            #yield word.subtre
            for si in word.subtree: 
                treelist.append(si)
                
        # dealing with a few outliers that spacy can't give a good dep_ for
        word_dep = word.dep_
        if(word.orth_ == "timetravel"): word_dep = "nsubj"

        tagged.append((thisword, word.pos_, word_dep, word.dep_, wl_list, rl_list, treelist))
    df = pd.DataFrame(tagged)
    df.rename(columns={0:'orth', 1:'pos', 2:'dep', 3:'dep_2', 4:'lefts', 5:'rights', \
                       6:'treelist'},\
              inplace=True)
    
    # simplify df with obvious substitutions
    for dfi in df.index:
        w = df.loc[dfi].orth
        if(w in Q1): df.set_value(dfi, 'orth', Q1[0])
            
        # if something is a named entity, mark it as such
        # Note that spacy will only evaluate capitalized words as named entities,
        # so you either have to only evaluate capitalized words or capitalize anything
        # you want to evaluate
        # TODO: should treat things in quotes as named
        if(w[0].isupper()):
            ndoc = nlp(w.title()); 
            if(not_empty(ndoc.ents)): df.set_value(dfi, 'dep', 'named_entity')
        
    
    ## selecting the subject_anchor
    anchorindex = 0
    longest = 0
    stype_rows = df[df.dep.isin(SUBJECT_TYPES)]
    if(len(stype_rows) == 0): stype_rows = df[df.dep.isin(SUBJECT_BACKUPS)]
    for sind in stype_rows.index:
        leftlist = stype_rows.loc[sind].lefts
        leftsize = len(leftlist)
        if(leftsize >= longest): longest = leftsize; anchorindex = sind;

    if(anchorindex > 0):df.set_value(anchorindex, 'dep', 'subject_anchor')
    return df


def get_spacy_toc(nlp, token):
    spacy_doc = None
    token_type =  type(token).__name__    
    if(token_type == "unicode"):
        spacy_doc = nlp(token)
    if(token_type == 'Token'): # <type 'spacy.tokens.token.Token'>
        spacy_doc = nlp(token.orth_)
    return spacy_doc

def make_matrix(nlp, sen):
    doc = nlp(sen)
    bmatrix = []
    for token in doc:
        newtok = token
        # newtok = get_spacy_toc(nlp, token)
        bmatrix.append(newtok.vector)
    return bmatrix

def all_tags(nlp, sentence):
    #sentence = unicode(sentence, "utf-8")
    doc = nlp(sentence)
    sns = list(doc.sents)
    #for sn in sns: root = sn.root; print 'root:', sn.root
    #for child in root.children: print child, ',', child.dep_
    #print '\n'
    tags = []
    tags.append(("word","tag","pos","dep","lemma","head","left_list","right_list","vec"))
    for word in doc:
        wl_list = []
        rl_list = []
        for wl in word.lefts: wl_list.append(wl)
        for rl in word.rights: rl_list.append(rl)
        # tags.append((word,word.tag_,word.pos_,word.dep_,word.lemma_, word.head, wl_list,  rl_list, word.vector))
        tags.append((word,word.tag_,word.pos_,word.dep_,word.lemma_, word.head, wl_list,  rl_list))
    return pd.DataFrame(tags)

def lefts_and_rights(sentence):
    #sentence = unicode(sentence, "utf-8")
    doc = nlp(sentence)
    tags = []
    for word in doc:
        wl_list = []
        rl_list = []
        for wl in word.lefts: wl_list.append(wl)
        for rl in word.rights: rl_list.append(rl)
        tags.append((word, wl_list,  rl_list))
    return tags

# stright forward dependency matrix
def dstr(sen):
    print sen
    doc = nlp(sen)
    tagged = []
    for word in doc:
        thisword = word.orth_
        
        wl_list = []
        for wl in word.lefts: 
            wl_word = wl.orth_
            if(wl.pos_ in verb_postags): wl_word = wl.lemma_
            wl_list.append(wl_word)
            
        rl_list = []
        for rl in word.rights: 
            rl_word = rl.orth_
            if(rl.pos_ in verb_postags): rl_word = rl.lemma_
            rl_list.append(rl_word)
 
        treelist = []
        if word.dep_ in noun_phrase_labels:
            for si in word.subtree: treelist.append(si)
                
            # dealing with a few outliers that spacy can't give a good dep_ for
        word_dep = word.dep_
        if(word.orth_ == "timetravel"): word_dep = "nsubj"

        tagged.append((thisword, word.pos_, word_dep, word.dep_, wl_list, rl_list, \
                           treelist, word.lemma_))
    df1 = pd.DataFrame(tagged)
    df1.rename(columns={0:'orth', 1:'pos', 2:'dep', 3:'dep_2', 4:'lefts', 5:'rights', \
                       6:'treelist', 7:'lemma'},inplace=True)
    
    print df1.orth.tolist()
    return df1

def dep_matrix(nlp, sen):
    # print sen
    doc = nlp(sen)
    tagged = []
    for word in doc:
        thisword = word.orth_
        
        wl_list = []
        for wl in word.lefts: 
            wl_word = wl.orth_
            if(wl.pos_ in verb_postags): wl_word = wl.lemma_
            wl_list.append(wl_word)
            
        rl_list = []
        for rl in word.rights: 
            rl_word = rl.orth_
            if(rl.pos_ in verb_postags): rl_word = rl.lemma_
            rl_list.append(rl_word)
 
        treelist = []
        if word.dep_ in noun_phrase_labels:
            for si in word.subtree: treelist.append(si)
                
            # dealing with a few outliers that spacy can't give a good dep_ for
        word_dep = word.dep_
        if(word.orth_ == "timetravel"): word_dep = "nsubj"

        tagged.append((word,thisword, word.tag_, word.pos_, word_dep, word.dep_, word.lemma_, word.head, \
                      wl_list, rl_list, treelist))
    df1 = pd.DataFrame(tagged)
    df1.rename(columns={0:'word',1:'orth', 2:'tag', 3:'pos', 4:'dep_1', 5:'dep_2', 6:'lemma', 7:'head',\
                       8:'lefts', 9:'rights', 10:'treelist'},inplace=True)
    
    return df1

### EVALUATION
def DTWDistance(s1, s2):
    DTW={}

    for i in range(len(s1)): DTW[(i, -1)] = float('inf')
    for i in range(len(s2)): DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] \
                 = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return sqrt(DTW[len(s1)-1, len(s2)-1])

def vDTWDistance(m1, m2):
    DTW={}

    for i in range(len(m1)): DTW[(i, -1)] = float('inf')
    for i in range(len(m2)): DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(m1)):
        for j in range(len(m2)):
            dist= (distance.euclidean(m1[i],m2[j]))**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return sqrt(DTW[len(m1)-1, len(m2)-1])

def viDTWDistance(m1, m2):
    DTW={}

    for i in range(len(m1)): DTW[(i, -1)] = float('inf')
    for i in range(len(m2)): DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(m1)):
        for j in range(len(m2)):
            dist= (distance.euclidean(m1[i],m2[j]))**2
            # dist = DTWDistance(m1[i],m2[j])
            # dist, ang = angle_between(m1[i],m2[j])
            # dist = cosim(m1[i], m2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    longer = len(m1)
    if(len(m2) > longer): longer = len(m2)
    rval =  (sqrt(DTW[len(m1)-1, len(m2)-1]))/longer
    # return sqrt(DTW[len(m1)-1, len(m2)-1])
    return rval

def angle_between(a,b):
    arccosInput = dot(a, b) / (norm(a) * norm(b))
    arccosInput = 1.0 if arccosInput > 1.0 else arccosInput
    arccosInput = -1.0 if arccosInput < -1.0 else arccosInput
    return arccosInput, math.acos(arccosInput)

def cosim(a,b):
    arccosInput = dot(a, b) / (norm(a) * norm(b))
    arccosInput = 1.0 if arccosInput > 1.0 else arccosInput
    arccosInput = -1.0 if arccosInput < -1.0 else arccosInput
    return arccosInput

def sen_similarity(m1, m2):
    av1 = sum(m1)/len(m1) 
    av2 = sum(m2)/len(m2)
    cos, angle = angle_between(av1, av2)
    return cos, angle


def compare_strings(nlp, q1, q2):
    tv1 = make_matrix(nlp, q1)
    tv2 = make_matrix(nlp, q2)

    dist = viDTWDistance(tv1, tv2)
    cos, angle = sen_similarity(tv1, tv2)
    return (dist, cos, angle)

def gensim_compare_strings(gensimmodel, q1, q2):
    tv1 = vectorize(gensimmodel, q1)
    tv2 = vectorize(gensimmodel, q2)

    dist = vDTWDistance(tv1, tv2)
    cos, angle = sen_similarity(tv1, tv2)
    return (dist, cos, angle)

def compare_vector_phrases(vector_matrix1, vector_matrix2):
    dist = vDTWDistance(vector_matrix1, vector_matrix2)
    cos, angle = sen_similarity(vector_matrix1, vector_matrix2)
    return (dist, cos, angle)

def compare_vectors(v1, v2):
    dist = DTWDistance(v1, v2)
    cos, angle = angle_between(v1, v2)
    return (dist, cos, angle)

#def modify_vector(nlp, word, modifier):
#    word_doc = nlp(word)
#    word_token = word_doc[0]
#    word_vector = word_token.vector
#
#    mod_doc  = nlp(modifier)
#    mod_token = mod_doc[0]
#    mod_vector = mod_token.vector
#
#    # merged = dot(word_vector, mod_vector)
#    merged = word_vector * mod_vector
#    return merged

def tknize(sen):
    tkns = [word.lower() for word in tokenizer.tokenize(sen)]
    return tkns

def vectorize(gensimmodel, sen):
    vecs = [gensimmodel[word] for word in tokenizer.tokenize(sen) if word in gensimmodel.wv.vocab]
    return vecs

def blurize(gensimmodel, phrase):
    vecs = [gensimmodel[word] for word in tokenizer.tokenize(phrase) if word in gensimmodel.wv.vocab]
    blurred = sum(vecs)/len(vecs)
    return blurred

def blurize_tokens(gensimmodel, tokens):
    vecs = [gensimmodel[word] for word in tokens if word in gensimmodel.wv.vocab]
    blurred = sum(vecs)/len(vecs)
    return blurred

def unite_words_as_vec(gensimmodel, sen):
    # tokens = [word.lower() for word in tokenizer.tokenize(sen) if word in gensimmodel]
    tokens = [word for word in tokenizer.tokenize(sen) if word in gensimmodel]
    united_vector = []
    if(len(tokens) > 0):
        sims = pd.DataFrame(gensimmodel.wv.most_similar(positive=tokens, topn=15))
        united_vector = blurize_tokens(gensimmodel, sims[0].tolist())
    return united_vector






