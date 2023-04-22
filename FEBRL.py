from FEBRL_utils import *
from utils import *

trainset = 'febrl3_UNSW'
testset = 'febrl4_UNSW'
set_seed(42);

# TRAIN SET CONSTRUCTION

# Import
print("Import train set...")
df_train = pd.read_csv(trainset + ".csv", index_col="rec_id")
train_true_links = generate_true_links(df_train)
print("Train set size:", len(df_train), ", number of matched pairs: ", str(len(train_true_links)))

# Preprocess train set
df_train['postcode'] = df_train['postcode'].astype(str)
df_train['given_name_soundex'] = phonetic(df_train['given_name'], method='soundex')
df_train['given_name_nysiis'] = phonetic(df_train['given_name'], method='nysiis')
df_train['surname_soundex'] = phonetic(df_train['surname'], method='soundex')
df_train['surname_nysiis'] = phonetic(df_train['surname'], method='nysiis')

# Final train feature vectors and labels
X_train, y_train = generate_train_X_y(df_train, train_true_links, extract_features)
print("Finished building X_train, y_train")

# Blocking Criteria: declare non-match of all the below fields disagree
# Import
print("Import test set...")
df_test = pd.read_csv(testset + ".csv", index_col="rec_id")
test_true_links = generate_true_links(df_test)
leng_test_true_links = len(test_true_links)
print("Test set size:", len(df_test), ", number of matched pairs: ", str(leng_test_true_links))

print("BLOCKING PERFORMANCE:")
blocking_fields = ["given_name", "surname", "postcode"]
all_candidate_pairs = []
for field in blocking_fields:
    block_indexer = rl.Index()
    block_indexer.block(on=field)
    candidates = block_indexer.index(df_test)
    detects = blocking_performance(candidates, test_true_links, df_test)
    all_candidate_pairs = candidates.union(all_candidate_pairs)
    print("Number of pairs of matched " + field + ": " + str(len(candidates)), ", detected ",
          detects, '/' + str(leng_test_true_links) + " true matched pairs, missed " +
          str(leng_test_true_links - detects))
detects = blocking_performance(all_candidate_pairs, test_true_links, df_test)
print("Number of pairs of at least 1 field matched: " + str(len(all_candidate_pairs)), ", detected ",
      detects, '/' + str(leng_test_true_links) + " true matched pairs, missed " +
      str(leng_test_true_links - detects))

# TEST SET CONSTRUCTION

# Preprocess test set
print("Processing test set...")
print("Preprocess...")
df_test['postcode'] = df_test['postcode'].astype(str)
df_test['given_name_soundex'] = phonetic(df_test['given_name'], method='soundex')
df_test['given_name_nysiis'] = phonetic(df_test['given_name'], method='nysiis')
df_test['surname_soundex'] = phonetic(df_test['surname'], method='soundex')
df_test['surname_nysiis'] = phonetic(df_test['surname'], method='nysiis')

# Test feature vectors and labels construction
print("Extract feature vectors...")
df_X_test = extract_features(df_test, all_candidate_pairs)
vectors = df_X_test.values.tolist()
labels = [0] * len(vectors)
feature_index = df_X_test.index
for i in range(0, len(feature_index)):
    if df_test.loc[feature_index[i][0]]["match_id"] == df_test.loc[feature_index[i][1]]["match_id"]:
        labels[i] = 1
X_test, y_test = shuffle(vectors, labels, random_state=0)
X_test = np.array(X_test)
y_test = np.array(y_test)
print("Count labels of y_test:", collections.Counter(y_test))
print("Finished building X_test, y_test")

# ENSEMBLE CLASSIFICATION AND EVALUATION

modeltypes = ['svm', 'nn', 'lg', 'rf']
modeltypes_2 = ['linear', 'relu', 'l2', 'gini']
modelparams = [0.005, 100, 0.2, 0]
number_models = len(modeltypes)
print('BASE MODEL RESULTS:')
for model_i in range(number_models):
    modeltype = modeltypes[model_i]
    modeltype_2 = modeltypes_2[model_i]
    modelparam = modelparams[model_i]
    md = train_model(modeltype, modelparam, X_train, y_train, modeltype_2, 10000)
    result = classify(md, X_test)
    eval = evaluation(y_test, result)
    print(modeltype, eval)

print("BAGGING PERFORMANCE:\n")
nFold = 10
kf = KFold(n_splits=nFold)
model_raw_score = [0] * number_models
model_binary_score = [0] * number_models
model_i = 0
for model_i in range(number_models):
    modeltype = modeltypes[model_i]
    modeltype_2 = modeltypes_2[model_i]
    modelparam = modelparams[model_i]
    print(modeltype, "per fold:")
    iFold = 0
    result_fold = [0] * nFold
    final_eval_fold = [0] * nFold
    for train_index, valid_index in kf.split(X_train):
        X_train_fold = X_train[train_index]
        y_train_fold = y_train[train_index]
        md = train_model(modeltype, modelparam, X_train_fold, y_train_fold, modeltype_2, 10000)
        result_fold[iFold] = classify(md, X_test)
        final_eval_fold[iFold] = evaluation(y_test, result_fold[iFold])
        print("Fold", str(iFold), final_eval_fold[iFold])
        iFold = iFold + 1
    bagging_raw_score = np.average(result_fold, axis=0)
    bagging_binary_score = np.copy(bagging_raw_score)
    bagging_binary_score[bagging_binary_score > 0.5] = 1
    bagging_binary_score[bagging_binary_score <= 0.5] = 0
    bagging_eval = evaluation(y_test, bagging_binary_score)
    print(modeltype, "bagging:", bagging_eval)
    print('')
    model_raw_score[model_i] = bagging_raw_score
    model_binary_score[model_i] = bagging_binary_score

thres = 0.99

print("STACKING PERFORMANCE:\n")
for i in range(number_models):
    selected_raw_scores = model_raw_score[:i] + model_raw_score[i+1:] if i < number_models - 1 else []
    assert len(selected_raw_scores) == number_models - 1
    stack_raw_score = np.average(selected_raw_scores, axis=0)
    stack_binary_score = np.copy(stack_raw_score)
    stack_binary_score[stack_binary_score > thres] = 1
    stack_binary_score[stack_binary_score <= thres] = 0
    stacking_eval = evaluation(y_test, stack_binary_score)
    print(stacking_eval)
