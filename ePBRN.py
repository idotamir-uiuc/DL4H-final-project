from sklearn.model_selection import KFold
import itertools
from utils import *
from ePBRN_utils import *

training_set_file_name = 'ePBRN_train'
testing_set_file_name = 'ePBRN_test'
set_seed(42)

print('Importing training data')
X_train, y_train = preprocess_and_generate_train_data(training_set_file_name)

print('Import testing data')
X_test, y_test = preprocess_and_generate_test_data(testing_set_file_name)

print('Base model performance')
models = {
    'svm': ['rbf', 0.001],
    'nn': ['relu', 2000],
    'lg': ['l2', 0.005],
    'rf': ['gini', 250],
    'dnn': ['relu', 300],
}

for model, model_params in models.items():
    md = train_model(model, model_params[1], X_train, y_train, model_params[0], 10000)
    prediction = classify(md, X_test)
    eval_results = evaluate_results(y_test, prediction)
    print(model, eval_results)

print('Bagging performance')
number_models = len(models)
number_folds = 10
kf = KFold(n_splits=number_folds)
model_raw_score = np.zeros((number_models, len(X_test)))
model_binary_score = [0] * number_models
model_i = 0
model_names = np.array(['.  '] * number_models, dtype=object)
for model, model_params in models.items():
    model_names[model_i] = model
    iFold = 0
    result_fold = [0] * number_folds
    for train_index, valid_index in kf.split(X_train):
        X_train_fold = X_train[train_index]
        y_train_fold = y_train[train_index]
        md = train_model(model, model_params[1], X_train_fold, y_train_fold, model_params[0], 30000)
        result_fold[iFold] = classify(md, X_test)
        iFold = iFold + 1
    bagging_raw_score = np.average(result_fold, axis=0)
    bagging_binary_score = np.copy(bagging_raw_score)
    bagging_binary_score[bagging_binary_score > 0.5] = 1
    bagging_binary_score[bagging_binary_score <= 0.5] = 0
    bagging_eval = evaluate_results(y_test, bagging_binary_score)
    print(model, bagging_eval)
    model_raw_score[model_i, :] = bagging_raw_score
    model_binary_score[model_i] = bagging_binary_score
    model_i += 1

print('Stacking performance using complete consensus:')
agreement_threshold = 0.99
for comb in itertools.combinations(range(len(model_raw_score)), 3):
    selected_raw_scores = model_raw_score[list(comb)]
    selected_models = model_names[list(comb)]
    stack_raw_score = np.average(selected_raw_scores, axis=0)
    stack_binary_score = np.copy(stack_raw_score)
    stack_binary_score[stack_binary_score > agreement_threshold] = 1
    stack_binary_score[stack_binary_score <= agreement_threshold] = 0
    stacking_eval = evaluate_results(y_test, stack_binary_score)
    print(*selected_models, sep=", ")
    print(stacking_eval)

agreement_threshold = 0.6
model_decision_threshold = 0.5
print('Stacking performance using partial consensus:')
for comb in itertools.combinations(range(len(model_raw_score)), 3):
    selected_raw_scores = model_raw_score[list(comb)]
    selected_models = model_names[list(comb)]
    selected_raw_scores[selected_raw_scores > model_decision_threshold] = 1
    selected_raw_scores[selected_raw_scores <= model_decision_threshold] = 0
    stack_raw_score = np.average(selected_raw_scores, axis=0)
    stack_binary_score = np.copy(stack_raw_score)
    stack_binary_score[stack_binary_score > agreement_threshold] = 1
    stack_binary_score[stack_binary_score <= agreement_threshold] = 0
    stacking_eval = evaluate_results(y_test, stack_binary_score)
    print(*selected_models, sep=", ")
    print(stacking_eval)
