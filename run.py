import os
import re
import nltk
import pickle
import pymer4
import logging
import scipy.io
import scipy.stats
import numpy as np
import pandas as pd
from pymer4.models import Lmer
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from statsmodels.stats import multitest
from sklearn.metrics.pairwise import cosine_similarity
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger


def load_glove_model(filename):
    print('Loading GloVe model ...')
    glove_model = {}
    with open(filename, 'r', errors='ignore', encoding='utf-8') as f:
        for line in f:
            split_line = line.split(' ')
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(len(glove_model), 'words loaded!')

    return glove_model


def strip_punctuation(word):
    result = word
    strip_left = re.compile(r"^[-!.,)'?(:]*")
    strip_right = re.compile(r"[-!.,)'?(:]*$")
    result = re.sub(strip_left, "", result)
    result = re.sub(strip_right, "", result)
    return result


def cosim(word1, word2, glove_model):
    try:
        return cosine_similarity([glove_model[word1]], [glove_model[word2]])[0][0]
    except KeyError:
        return None


def standardise_variable(df_column):
    return (df_column - np.mean(df_column)) / np.std(df_column)


def fit_model(df, erp='N400', add_glove=True):
    print('\n' + '-' * (58 if add_glove else 61))
    print('Fitting linear mixed-effects model for', erp, 'with' + ('' if add_glove else 'out'), 'GloVeDist')
    print('-' * (58 if add_glove else 61) + '\n')
    formula = erp + '_NoRej ~ WordLength + PosInSentence + NgramSrp + PsgSrp' + ('+ GloVeDist' if add_glove else '') # fixed effects
    formula += '+ (1|Subject) + (1|Item)' # random by-subject and by-item intercepts
    formula += '+ (0 + WordLength|Subject) + (0 + PosInSentence|Subject) + (0 + NgramSrp|Subject) + (0 + PsgSrp|Subject)' # random by-subject slopes
    model = Lmer(formula, data=df)
    print(model.fit(REML=False, control="optimizer = 'bobyqa', optCtrl = list(maxfun = 2e5)"))
    return model


def main():
    # Disable R warnings
    rpy2_logger.setLevel(logging.ERROR)
    
    # Load stopwords
    stopwords_en = set(stopwords.words('english'))
    
    # Load GloVe models if "semantic distances" aren't precomputed
    glove_model = None
    if not os.path.exists('glove_dists.pickle'):
        glove_model = load_glove_model('glove.840B.300d.txt')
    
    # Load ERP data
    mat = scipy.io.loadmat('stimuli_erp_Frank_et_al_2015.mat')
    
    # Arrange data
    data = {'SentenceID': [], 'Token': [], 'WordLength': [], 'PosInSentence': [], 'Rejected': [], 'N400': [], 'P600': [], 'NgramSrp': [], 'PsgSrp': []}

    for s in range(len(mat['sentences'])):
        # Sentence indices:
        data['SentenceID'] += [s for _ in range(len(mat['sentences'][s][0][0]))]
        
        # Tokens:
        data['Token'] += [mat['sentences'][s][0][0][w][0] for w in range(len(mat['sentences'][s][0][0]))]
        
        # Word length:
        data['WordLength'] += [mat['wordlength'][s][0][0][w] for w in range(len(mat['sentences'][s][0][0]))]
        
        # Position in sentence:
        data['PosInSentence'] += [w for w in range(len(mat['sentences'][s][0][0]))]
        
        # Rejected:
        data['Rejected'] += [tuple((bool(x) for x in mat['reject'][s][0][w])) for w in range(len(mat['sentences'][s][0][0]))]
        
        # N400:
        data['N400'] += [tuple((mat['ERP'][s][0][w][p][2] for p in range(len(mat['ERP'][s][0][w])))) for w in range(len(mat['sentences'][s][0][0]))]

        # P600:
        data['P600'] += [tuple((mat['ERP'][s][0][w][p][4] for p in range(len(mat['ERP'][s][0][w])))) for w in range(len(mat['sentences'][s][0][0]))]
        
        # 4-gram surprisal (maximal, i.e. over full corpus):
        data['NgramSrp'] += [mat['surp_ngramfull'][s][0][w][2] for w in range(len(mat['sentences'][s][0][0]))]
        
        # PSG surprisal (maximal, i.e. over 1.1M training sentences):
        data['PsgSrp'] += [mat['surp_psg'][s][0][w][8] for w in range(len(mat['sentences'][s][0][0]))]

    df = pd.DataFrame(data)
    
    # Additional transformations
    df['IsStopword'] = [all((x in stopwords_en for x in strip_punctuation(token).lower().split("'"))) for token in df.Token]
    
    df['N400_NoRej'] = [tuple((n4 if not rej else None for n4, rej in zip(df.N400[i], df.Rejected[i]))) for i in range(len(df.Rejected))]
    
    df['P600_NoRej'] = [tuple((p6 if not rej else None for p6, rej in zip(df.P600[i], df.Rejected[i]))) for i in range(len(df.Rejected))]
    
    # Compute "semantic distances"
    glove_dists = []
    if glove_model:
        prev_cwords = []
        current_sent_id = -1
        for sent_id, token, is_sword in zip(df.SentenceID, df.Token, df.IsStopword):
            if sent_id != current_sent_id:
                prev_cwords = []
                current_sent_id = sent_id
            if not is_sword:
                cword = strip_punctuation(token)
                values = [cosim(cword, prev_cword, glove_model) for prev_cword in prev_cwords]
                values = [1 - x for x in values if x != None]
                if values:
                    glove_dists.append(np.mean(values))
                else:
                    glove_dists.append(None)
                prev_cwords.append(strip_punctuation(token))
            else:
                glove_dists.append(None)
        with open('glove_dists.pickle', 'wb') as f:
            pickle.dump(glove_dists, f)
    else:
        with open('glove_dists.pickle', 'rb') as f:
            glove_dists = pickle.load(f)
    df['GloVeDist'] = glove_dists
    
    # Remove rows with no available "semantic distance" measure or with only rejected ERP values (usually, sentence-initial/-final)
    rows_to_remove = []
    for i in range(len(df)):
        if df.GloVeDist[i] != df.GloVeDist[i]: # is NaN
            rows_to_remove.append(i)
        elif False not in df.Rejected[i]:
            rows_to_remove.append(i)
    df = df.drop(df.index[rows_to_remove])
    df = df.reset_index(drop=True)
    
    # Standardise variables
    df.WordLength = standardise_variable(df.WordLength)
    df.PosInSentence = standardise_variable(df.PosInSentence)
    df.NgramSrp = standardise_variable(df.NgramSrp)
    df.PsgSrp = standardise_variable(df.PsgSrp)
    df.GloVeDist = standardise_variable(df.GloVeDist)
    
    # Flatten data, remove redundant columns
    num_subjects = len(df.Rejected[0])
    data = {'SentenceID': [], 'Token': [], 'WordLength': [], 'PosInSentence': [], 'NgramSrp': [], 'PsgSrp': [], 'GloVeDist': [],
            'N400_NoRej': [], 'P600_NoRej': [], 'Subject': [], 'Item': []}
    for item in range(len(df)):
        for subject in range(num_subjects):
            data['SentenceID'].append(str(df.SentenceID[item]))
            data['Token'].append(df.Token[item])
            data['WordLength'].append(df.WordLength[item])
            data['PosInSentence'].append(df.PosInSentence[item])
            data['NgramSrp'].append(df.NgramSrp[item])
            data['PsgSrp'].append(df.PsgSrp[item])
            data['GloVeDist'].append(df.GloVeDist[item])
            data['N400_NoRej'].append(df.N400_NoRej[item][subject])
            data['P600_NoRej'].append(df.P600_NoRej[item][subject])
            data['Subject'].append(str(subject))
            data['Item'].append(str(item))
    df = pd.DataFrame(data)
    df.SentenceID = df.SentenceID.astype('category')
    df.Token = df.Token.astype('category')
    df.Subject = df.Subject.astype('category')
    df.Item = df.Item.astype('category')

    # Likelihood ratio tests:
    n4_reduced_model = fit_model(df, erp='N400', add_glove=False)
    n4_full_model = fit_model(df, erp='N400', add_glove=True)
    p6_reduced_model = fit_model(df, erp='P600', add_glove=False)
    p6_full_model = fit_model(df, erp='P600', add_glove=True)
    
    n4_LR_statistic = -2 * (n4_reduced_model.logLike - n4_full_model.logLike)
    p6_LR_statistic = -2 * (p6_reduced_model.logLike - p6_full_model.logLike)
    
    n4_p_val = scipy.stats.chi2.sf(n4_LR_statistic, 1)
    p6_p_val = scipy.stats.chi2.sf(p6_LR_statistic, 1)
    
    n4_holm_p_val, p6_holm_p_val = multitest.multipletests([n4_p_val, p6_p_val], alpha=0.05, method='h')[1]
    
    print('\n' + '-' * 22)
    print('Likelihood ratio tests')
    print('-' * 22 + '\n')
    
    print('LRT test statistic (chi^2, df=1) for N400:', round(n4_LR_statistic, 3))
    print('LRT test statistic (chi^2, df=1) for P600:', round(p6_LR_statistic, 3))
    print()
    print('LRT raw p-value for N400:', round(n4_p_val, 3))
    print('LRT raw p-value for P600:', round(p6_p_val, 3))
    print()
    print('LRT Holm-corrected p-value for N400:', round(n4_holm_p_val, 3), '( <  0.05 --> choose full model  )' if n4_holm_p_val < .05 else '( >= 0.05 --> keep reduced model )')
    print('LRT Holm-corrected p-value for P600:', round(p6_holm_p_val, 3), '( <  0.05 --> choose full model  )' if p6_holm_p_val < .05 else '( >= 0.05 --> keep reduced model )')
    print()
    
    # Return dataframe
    return df


if __name__ == '__main__':
    df = main()
