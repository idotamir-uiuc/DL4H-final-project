from sklearn.model_selection import KFold
import itertools
from utils import *
import numpy as np

set_seed(42)

print('Importing training data')
X_train = np.genfromtxt('data/ePBRN_X_train.csv', delimiter=',')
y_train = np.genfromtxt('data/ePBRN_y_train.csv', delimiter=',')

print('Import testing data')
X_test = np.genfromtxt('data/ePBRN_X_test.csv', delimiter=',')
y_test = np.genfromtxt('data/ePBRN_y_test.csv', delimiter=',')

print('Base model performance')
models = {
    'svm': ['rbf', 0.001],
    'nn': ['relu', 2000],
    'lg': ['l2', 0.005],
    'rf': ['gini', 0.15],
    'dnn': ['relu', 750],
}

for model, model_params in models.items():
    md = train_model(model, model_params[1], X_train, y_train, model_params[0], 30000)
    prediction = classify(md, X_test)
    eval_results = evaluate_results(y_test, prediction)
    print(model, eval_results)

print('Bagging performance')
number_models = len(models)
number_folds = 10
kf = KFold(n_splits=number_folds)
model_bagging_results = np.zeros((number_models, len(X_test)))
model_i = 0
model_names = np.array(['.  '] * number_models, dtype=object)
for model, model_params in models.items():
    model_names[model_i] = model
    fold_i = 0
    fold_results = [0] * number_folds
    print(f'{model} individual fold performance')
    for train_index, valid_index in kf.split(X_train):
        X_train_fold = X_train[train_index]
        y_train_fold = y_train[train_index]
        md = train_model(model, model_params[1], X_train_fold, y_train_fold, model_params[0], 30000)
        fold_results[fold_i] = classify(md, X_test)
        print(f'{model} fold {fold_i}: ', evaluate_results(y_test, fold_results[fold_i]))
        fold_i += 1
    bagging_results = np.average(fold_results, axis=0)
    bagging_predictions = np.copy(bagging_results)
    bagging_predictions[bagging_predictions > 0.5] = 1
    bagging_predictions[bagging_predictions <= 0.5] = 0
    bagging_eval = evaluate_results(y_test, bagging_predictions)
    print('##############################################################################')
    print(f'{model} bagging performance: ', bagging_eval)
    print('##############################################################################')
    model_bagging_results[model_i, :] = bagging_results
    model_i += 1

print('Stacking performance using complete consensus:')
agreement_threshold = 0.99
for comb in itertools.combinations(range(len(model_bagging_results)), 3):
    sub_models_bagging_results = model_bagging_results[list(comb)]
    sub_models_names = model_names[list(comb)]
    stack_results = np.average(sub_models_bagging_results, axis=0)
    stack_predictions = np.copy(stack_results)
    stack_predictions[stack_predictions > agreement_threshold] = 1
    stack_predictions[stack_predictions <= agreement_threshold] = 0
    stacking_eval = evaluate_results(y_test, stack_predictions)
    print(*sub_models_names, sep=', ')
    print(stacking_eval)

agreement_threshold = 0.66
model_decision_threshold = 0.99
print('Stacking performance using partial consensus:')
for comb in itertools.combinations(range(len(model_bagging_results)), 3):
    sub_models_bagging_results = np.copy(model_bagging_results[list(comb)])
    sub_models_names = model_names[list(comb)]
    sub_models_bagging_results[sub_models_bagging_results > model_decision_threshold] = 1
    sub_models_bagging_results[sub_models_bagging_results <= model_decision_threshold] = 0
    stack_results = np.average(sub_models_bagging_results, axis=0)
    stack_predictions = np.copy(stack_results)
    stack_predictions[stack_predictions > agreement_threshold] = 1
    stack_predictions[stack_predictions <= agreement_threshold] = 0
    stacking_eval = evaluate_results(y_test, stack_predictions)
    print(*sub_models_names, sep=', ')
    print(stacking_eval)
