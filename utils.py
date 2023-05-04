import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from numpy.random import choice
import random
import os
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


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
        if modeltype_2 == 'l2':
            model = LogisticRegression(C=modelparam, penalty=modeltype_2, class_weight=None, dual=False,
                                       fit_intercept=True,
                                       intercept_scaling=1, max_iter=5000, multi_class='ovr',
                                       n_jobs=1, random_state=42)
        else:
            model = LogisticRegression(C=modelparam, penalty=modeltype_2, class_weight=None, solver='liblinear',
                                       fit_intercept=True,
                                       intercept_scaling=1, max_iter=5000, multi_class='ovr',
                                       n_jobs=1, random_state=42)
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
        model = RandomForestClassifier(criterion=modeltype_2, n_estimators=400, max_features='sqrt', max_depth=10, bootstrap=True, random_state=42, class_weight='balanced', min_samples_leaf=2, ccp_alpha=modelparam)
        model.fit(train_vectors, train_labels)
    elif modeltype == 'dnn':
        model = MLPClassifier(solver='lbfgs', alpha=modelparam, hidden_layer_sizes=(256, 100, 50),
                              activation=modeltype_2, random_state=42, batch_size='auto',
                              learning_rate='constant', learning_rate_init=0.001,
                              power_t=0.5, max_iter=max_iter, shuffle=True,
                              tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                              nesterovs_momentum=True, early_stopping=False,
                              validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.fit(train_vectors, train_labels)
    else:
        raise Exception()

    return model


def classify(model, test_vectors):
    result = model.predict(test_vectors)
    return result


class EvaluationResults:
    def __init__(self, cm, precision, recall, f_score):
        self.confusion_matrix = cm
        self.precision = precision
        self.recall = recall
        self.f_score = f_score

    def __repr__(self):
        return f'confusion_matrx: {self.confusion_matrix.tolist()}, precision: {self.precision}, recall: {self.recall}, f_score: {self.f_score}'


def evaluate_results(test_labels, results):
    cm = confusion_matrix(test_labels, results)  # tn, fp, fn, tp
    f_score = f1_score(test_labels, results)
    precision = precision_score(test_labels, results, zero_division=0)
    recall = recall_score(test_labels, results)
    return EvaluationResults(cm, precision, recall, f_score)


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


# max_iter = 10000 for FEBRL and 30000 ePBRN
def calculate_f_scores_for_models(models, hyperparameter_values, X_train, y_train, X_test, y_test, max_iter):
    model_f_scores = {}
    for model, model_types in models.items():
        for model_type in model_types:
            print(f'Calculating F-scores for all variations of {model} {model_type}')
            f_scores = []
            model_f_scores[f'{model}, {model_type}'] = f_scores
            for param_value in hyperparameter_values:
                md = train_model(model, param_value, X_train, y_train, model_type, max_iter)
                final_result = classify(md, X_test)
                final_eval = evaluate_results(y_test, final_result)
                f_scores += [final_eval.f_score]

    return model_f_scores


def plot_f_scores(f_scores, hyperparameter_values, scheme_name):
    fig = plt.figure(figsize=(12, 6), dpi=100)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    for model, scores in f_scores.items():
        plt.plot([str(x) for x in hyperparameter_values], scores, label=model)
    plt.title(f'Hyperparameter Performance of Scheme {scheme_name}')
    plt.xlabel('Hyperparameters')
    plt.ylabel('F1-score')
    ax.legend(bbox_to_anchor=(1.2, 0.6), loc='center right', shadow=True, fontsize='large')
    fig.savefig(f'Scheme_{scheme_name}.png', bbox_inches="tight")


def plus_minus_list(middle, element_dist, plus_minus, min_value):
    if middle is None or middle == True or middle == False:
        return [middle]
    return [middle - plus_minus * (d + 1) for d in range(element_dist) if
            middle - plus_minus * (d + 1) > min_value] + [
        middle] + [middle + plus_minus * (d + 1) for
                   d in range(element_dist) if middle + plus_minus * (d + 1) > min_value]


def find_hyperparameters_for_random_forest(X_train, y_train, X_test, y_test):
    params = {'n_estimators': [int(x) for x in np.arange(100, 2000, 200)],
              'max_features': [1, 'log2', 'sqrt'],
              'max_depth': ([int(x) for x in np.linspace(10, 110, num=11)] + [None]),
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'bootstrap': [True, False],
              'random_state': [42]}
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=params, n_iter=100, cv=3, verbose=0,
                                   random_state=42, n_jobs=-1, scoring='f1')
    rf_random.fit(X_train, y_train)
    base_model = RandomForestClassifier(n_estimators=10, random_state=42)
    base_model.fit(X_train, y_train)
    print('Base results for Random Forest:')
    print(evaluate_results(y_test, base_model.predict(X_test)))
    print('Improved results for Random Forest:')
    best_random = rf_random.best_estimator_
    print(evaluate_results(y_test, best_random.predict(X_test)))

    params = {
        'bootstrap': [rf_random.best_params_['bootstrap']],
        'max_depth': plus_minus_list(rf_random.best_params_['max_depth'], 1, 5, 0),
        'max_features': [1, 'log2', 'sqrt'],
        'min_samples_leaf': plus_minus_list(rf_random.best_params_['min_samples_leaf'], 1, 1, 1),
        'min_samples_split': plus_minus_list(rf_random.best_params_['min_samples_split'], 1, 1, 1),
        'n_estimators': plus_minus_list(rf_random.best_params_['n_estimators'], 2, 50, 0),
        'random_state': [42]
    }

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=params,
                               cv=3, n_jobs=-1, verbose=0, scoring='f1')
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    best_rf.fit(X_train, y_train)
    print('Best results for Random Forest:')
    print(evaluate_results(y_test, best_rf.predict(X_test)))
    print('Parameters for these results:')
    print(grid_search.best_params_)