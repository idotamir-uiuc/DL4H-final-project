from ePBRN_utils import *
from utils import *

training_set_file_name = 'ePBRN_train'
testing_set_file_name = 'ePBRN_test'
set_seed(42)

print('Importing training data')
X_train, y_train = preprocess_and_generate_train_data(training_set_file_name)

print('Import testing data')
X_test, y_test = preprocess_and_generate_test_data(testing_set_file_name)

print("Base learners from original paper hyperparameter selection:")
models = {
    'dnn': ['relu', 'logistic'],
    'svm': ['linear', 'rbf'],
    'lg': ['l1', 'l2'],
    'nn': ['relu', 'logistic'],
    'rf': ['gini'],
}

hyperparameter_values = [.001, .002, .005, .01, .02, .05, .1, .2, .5, 1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
f_scores = calculate_f_scores_for_models(models, hyperparameter_values, X_train, y_train, X_test, y_test, 30000)
plot_f_scores(f_scores, hyperparameter_values, 'B')


# print("Random Forest from original paper hyperparameter selection:")
# find_hyperparameters_for_random_forest(X_train, y_train, X_test, y_test)
