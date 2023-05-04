from ePBRN_utils import *
from utils import *

training_set_file_name = 'data/ePBRN_train'
testing_set_file_name = 'data/ePBRN_test'
set_seed(42)

print('Importing training data')
X_train, y_train = preprocess_and_generate_train_data(training_set_file_name)

print('Import testing data')
X_test, y_test = preprocess_and_generate_test_data(testing_set_file_name)

np.savetxt("data/ePBRN_X_train.csv", X_train, delimiter=",")
np.savetxt("data/ePBRN_y_train.csv", y_train, delimiter=",")
np.savetxt("data/ePBRN_X_test.csv", X_test, delimiter=",")
np.savetxt("data/ePBRN_y_test.csv", y_test, delimiter=",")
