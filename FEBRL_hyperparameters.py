from utils import *

set_seed(42)

print('Importing training data')
X_train = np.genfromtxt('data/FERBL_X_train.csv', delimiter=',')
y_train = np.genfromtxt('data/FERBL_y_train.csv', delimiter=',')

print('Import testing data')
X_test = np.genfromtxt('data/FERBL_X_test.csv', delimiter=',')
y_test = np.genfromtxt('data/FERBL_y_test.csv', delimiter=',')

print('Base learners from original paper hyperparameter selection:')
models = {
    'dnn': ['relu', 'logistic'],
    'svm': ['linear', 'rbf'],
    'lg': ['l1', 'l2'],
    'nn': ['relu', 'logistic'],
    'rf': ['gini'],
}

hyperparameter_values = [.001, .002, .005, .01, .02, .05, .1, .2, .5, 1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
f_scores = calculate_f_scores_for_models(models, hyperparameter_values, X_train, y_train, X_test, y_test, 10000)
plot_f_scores(f_scores, hyperparameter_values, 'A')
