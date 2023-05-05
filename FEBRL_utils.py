# Feature creation methods all taken directly from
# https://github.com/ePBRN/Medical-Record-Linkage-Ensemble/blob/master/FEBRL_UNSW_Linkage.ipynb
from utils import *
import recordlinkage as rl
from recordlinkage.preprocessing import phonetic


def extract_features(df, links):
    c = rl.Compare()
    c.string('given_name', 'given_name', method='jarowinkler', label='y_name')
    c.string('given_name_soundex', 'given_name_soundex', method='jarowinkler', label='y_name_soundex')
    c.string('given_name_nysiis', 'given_name_nysiis', method='jarowinkler', label='y_name_nysiis')
    c.string('surname', 'surname', method='jarowinkler', label='y_surname')
    c.string('surname_soundex', 'surname_soundex', method='jarowinkler', label='y_surname_soundex')
    c.string('surname_nysiis', 'surname_nysiis', method='jarowinkler', label='y_surname_nysiis')
    c.exact('street_number', 'street_number', label='y_street_number')
    c.string('address_1', 'address_1', method='levenshtein', threshold=0.7, label='y_address1')
    c.string('address_2', 'address_2', method='levenshtein', threshold=0.7, label='y_address2')
    c.exact('postcode', 'postcode', label='y_postcode')
    c.exact('day', 'day', label='y_day')
    c.exact('month', 'month', label='y_month')
    c.exact('year', 'year', label='y_year')

    # Build features
    feature_vectors = c.compute(links, df, df)
    return feature_vectors


def preprocess_and_generate_train_data(training_set_file_name):
    # Import
    df_train = pd.read_csv(training_set_file_name + '.csv', index_col='rec_id')
    train_true_links = generate_true_links(df_train)
    print('Train set size:', len(df_train), ', number of matched pairs: ', str(len(train_true_links)))
    # Preprocess train set
    df_train['postcode'] = df_train['postcode'].astype(str)
    df_train['given_name_soundex'] = phonetic(df_train['given_name'], method='soundex')
    df_train['given_name_nysiis'] = phonetic(df_train['given_name'], method='nysiis')
    df_train['surname_soundex'] = phonetic(df_train['surname'], method='soundex')
    df_train['surname_nysiis'] = phonetic(df_train['surname'], method='nysiis')
    # Final train feature vectors and labels
    return generate_train_X_y(df_train, train_true_links, extract_features)


def preprocess_and_generate_test_data(testing_set_file_name):
    df_test = pd.read_csv(testing_set_file_name + '.csv', index_col='rec_id')
    test_true_links = generate_true_links(df_test)
    leng_test_true_links = len(test_true_links)
    print('Test set size:', len(df_test), ', number of matched pairs: ', str(leng_test_true_links))
    print('BLOCKING PERFORMANCE:')
    blocking_fields = ['given_name', "surname", "postcode"]
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

    return X_test, y_test
