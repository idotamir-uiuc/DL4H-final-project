# Feature creation methods all taken directly from
# https://github.com/ePBRN/Medical-Record-Linkage-Ensemble/blob/master/ePBRN_UNSW_Linkage.ipynb
from utils import *
import recordlinkage as rl


def swap_fields_flag(f11, f12, f21, f22):
    return ((f11 == f22) & (f12 == f21)).astype(float)


def join_names_space(f11, f12, f21, f22):
    return ((f11 + ' ' + f12 == f21) | (f11 + ' ' + f12 == f22) | (f21 + ' ' + f22 == f11) | (
            f21 + ' ' + f22 == f12)).astype(float)


def join_names_dash(f11, f12, f21, f22):
    return ((f11 + '-' + f12 == f21) | (f11 + '-' + f12 == f22) | (f21 + '-' + f22 == f11) | (
            f21 + '-' + f22 == f12)).astype(float)


def abb_surname(f1, f2):
    return ((f1[0] == f2) | (f1 == f2[0])).astype(float)


def reset_day(f11, f12, f21, f22):
    return (((f11 == 1) & (f12 == 1)) | ((f21 == 1) & (f22 == 1))).astype(float)


def extract_features(df, links):
    c = rl.Compare()
    c.string('given_name', 'given_name', method='levenshtein', label='y_name_leven')
    c.string('surname', 'surname', method='levenshtein', label='y_surname_leven')
    c.string('given_name', 'given_name', method='jarowinkler', label='y_name_jaro')
    c.string('surname', 'surname', method='jarowinkler', label='y_surname_jaro')
    c.string('postcode', 'postcode', method='jarowinkler', label='y_postcode')
    exact_fields = ['postcode', 'address_1', 'address_2', 'street_number']
    for field in exact_fields:
        c.exact(field, field, label='y_' + field + '_exact')
    c.compare_vectorized(reset_day, ('day', 'month'), ('day', 'month'), label='reset_day_flag')
    c.compare_vectorized(swap_fields_flag, ('day', 'month'), ('day', 'month'), label='swap_day_month')
    c.compare_vectorized(swap_fields_flag, ('surname', 'given_name'), ('surname', 'given_name'), label='swap_names')
    c.compare_vectorized(join_names_space, ('surname', 'given_name'), ('surname', 'given_name'),
                         label='join_names_space')
    c.compare_vectorized(join_names_dash, ('surname', 'given_name'), ('surname', 'given_name'), label='join_names_dash')
    c.compare_vectorized(abb_surname, 'surname', 'surname', label='abb_surname')
    # Build features
    feature_vectors = c.compute(links, df, df)
    return feature_vectors


def preprocess_and_generate_train_data(training_set_file_name):
    df_train = pd.read_csv(training_set_file_name + '.csv', index_col="rec_id")
    train_true_links = generate_true_links(df_train)
    print("Train set size:", len(df_train), ", number of matched pairs: ", str(len(train_true_links)))
    # Preprocess train set
    df_train['postcode'] = df_train['postcode'].astype(str)
    # Final train feature vectors and labels
    return generate_train_X_y(df_train, train_true_links, extract_features)

def preprocess_and_generate_test_data(testing_set_file_name):
    df_test = pd.read_csv(testing_set_file_name + ".csv", index_col="rec_id")
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
        detects = blocking_performance(candidates, df_test)
        all_candidate_pairs = candidates.union(all_candidate_pairs)
        print("Number of pairs of matched " + field + ": " + str(len(candidates)), ", detected ",
              detects, '/' + str(leng_test_true_links) + " true matched pairs, missed " +
              str(leng_test_true_links - detects))
    detects = blocking_performance(all_candidate_pairs, df_test)
    print("Number of pairs of at least 1 field matched: " + str(len(all_candidate_pairs)), ", detected ",
          detects, '/' + str(leng_test_true_links) + " true matched pairs, missed " +
          str(leng_test_true_links - detects))
    ## TEST SET CONSTRUCTION
    # Preprocess test set
    print("Processing test set...")
    print("Preprocess...")
    df_test['postcode'] = df_test['postcode'].astype(str)
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
    return X_test, y_test
