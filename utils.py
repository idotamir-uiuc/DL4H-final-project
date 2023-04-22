import recordlinkage as rl, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from recordlinkage.preprocessing import phonetic
from numpy.random import choice
import random
import os
import collections, numpy
from IPython.display import clear_output
from sklearn.model_selection import train_test_split, KFold


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_false_links(df, size):
    # A counterpart of generate_true_links(), with the purpose to generate random false pairs
    # for training. The number of false pairs in specified as "size".
    df["rec_id"] = df.index.values.tolist()
    indices_1 = []
    indices_2 = []
    unique_match_id = df["match_id"].unique()
    unique_match_id = unique_match_id[~np.isnan(unique_match_id)]  # remove nan values
    for j in range(size):
        false_pair_ids = choice(unique_match_id, 2)
        candidate_1_cluster = df.loc[df['match_id'] == false_pair_ids[0]]
        candidate_1 = candidate_1_cluster.iloc[choice(range(len(candidate_1_cluster)))]
        candidate_2_cluster = df.loc[df['match_id'] == false_pair_ids[1]]
        candidate_2 = candidate_2_cluster.iloc[choice(range(len(candidate_2_cluster)))]
        indices_1 = indices_1 + [candidate_1["rec_id"]]
        indices_2 = indices_2 + [candidate_2["rec_id"]]
    links = pd.MultiIndex.from_arrays([indices_1, indices_2])
    return links


def generate_true_links(df):
    # although the match_id column is included in the original df to imply the true links,
    # this function will create the true_link object identical to the true_links properties
    # of recordlinkage toolkit, in order to exploit "Compare.compute()" from that toolkit
    # in extract_function() for extracting features quicker.
    # This process should be deprecated in the future release of the UNSW toolkit.
    df["rec_id"] = df.index.values.tolist()
    indices_1 = []
    indices_2 = []
    processed = 0
    for match_id in df["match_id"].unique():
        if match_id != -1:
            processed = processed + 1
            # print("In routine generate_true_links(), count =", processed)
            # clear_output(wait=True)
            linkages = df.loc[df['match_id'] == match_id]
            for j in range(len(linkages) - 1):
                for k in range(j + 1, len(linkages)):
                    indices_1 = indices_1 + [linkages.iloc[j]["rec_id"]]
                    indices_2 = indices_2 + [linkages.iloc[k]["rec_id"]]
    links = pd.MultiIndex.from_arrays([indices_1, indices_2])
    return links


def generate_train_X_y(df, train_true_links, extract_features):
    # This routine is to generate the feature vector X and the corresponding labels y
    # with exactly equal number of samples for both classes to train the classifier.
    pos = extract_features(df, train_true_links)
    train_false_links = generate_false_links(df, len(train_true_links))
    neg = extract_features(df, train_false_links)
    X = pos.values.tolist() + neg.values.tolist()
    y = [1] * len(pos) + [0] * len(neg)
    X, y = shuffle(X, y, random_state=0)
    X = np.array(X)
    y = np.array(y)
    return X, y


# max_iter = 10000 for FEBRL and 30000 ePBRN
def train_model(modeltype, modelparam, train_vectors, train_labels, modeltype_2, max_iter):
    if modeltype == 'svm':  # Support Vector Machine
        model = svm.SVC(C=modelparam, kernel=modeltype_2, random_state=42)
        model.fit(train_vectors, train_labels)
    elif modeltype == 'lg':  # Logistic Regression
        model = LogisticRegression(C=modelparam, penalty=modeltype_2, class_weight=None, dual=False, fit_intercept=True,
                                   intercept_scaling=1, max_iter=5000, multi_class='ovr',
                                   n_jobs=1, random_state=42)
        model.fit(train_vectors, train_labels)
    elif modeltype == 'nb':  # Naive Bayes
        model = GaussianNB()
        model.fit(train_vectors, train_labels)
    elif modeltype == 'nn':  # Neural Network
        model = MLPClassifier(solver='lbfgs', alpha=modelparam, hidden_layer_sizes=(256,),
                              activation=modeltype_2, random_state=42, batch_size='auto',
                              learning_rate='constant', learning_rate_init=0.001,
                              power_t=0.5, max_iter=max_iter, shuffle=True,
                              tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                              nesterovs_momentum=True, early_stopping=False,
                              validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.fit(train_vectors, train_labels)
    elif modeltype == 'rf':
        model = RandomForestClassifier(criterion=modeltype_2, random_state=42)
        model.fit(train_vectors, train_labels)
    else:
        raise Exception()

    return model


def classify(model, test_vectors):
    result = model.predict(test_vectors)
    return result


def evaluation(test_labels, result):
    true_pos = np.logical_and(test_labels, result)
    count_true_pos = np.sum(true_pos)
    true_neg = np.logical_and(np.logical_not(test_labels), np.logical_not(result))
    count_true_neg = np.sum(true_neg)
    false_pos = np.logical_and(np.logical_not(test_labels), result)
    count_false_pos = np.sum(false_pos)
    false_neg = np.logical_and(test_labels, np.logical_not(result))
    count_false_neg = np.sum(false_neg)
    precision = count_true_pos / (count_true_pos + count_false_pos)
    sensitivity = count_true_pos / (count_true_pos + count_false_neg)  # sensitivity = recall
    confusion_matrix = [count_true_pos, count_false_pos, count_false_neg, count_true_neg]
    no_links_found = np.count_nonzero(result)
    no_false = count_false_pos + count_false_neg
    Fscore = 2 * precision * sensitivity / (precision + sensitivity)
    metrics_result = {'no_false': no_false, 'confusion_matrix': confusion_matrix, 'precision': precision,
                      'sensitivity': sensitivity, 'no_links': no_links_found, 'F-score': Fscore}
    return metrics_result


def blocking_performance(candidates, true_links, df):
    count = 0
    for candi in candidates:
        if df.loc[candi[0]]["match_id"] == df.loc[candi[1]]["match_id"]:
            count = count + 1
    return count


def argmax(a):
    np_a = np.array(a)
    i = np.argmax(np_a)
    return np_a[i], i


def argmin(a):
    np_a = np.array(a)
    i = np.argmin(np_a)
    return np_a[i], i
