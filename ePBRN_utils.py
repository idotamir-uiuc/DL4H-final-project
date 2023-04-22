# Feature creation methods all taken directly from TODO: Insert link here
import recordlinkage as rl


def swap_fields_flag(f11, f12, f21, f22):
    return ((f11 == f22) & (f12 == f21)).astype(float)


def join_names_space(f11, f12, f21, f22):
    return ((f11 + " " + f12 == f21) | (f11 + " " + f12 == f22) | (f21 + " " + f22 == f11) | (
                f21 + " " + f22 == f12)).astype(float)


def join_names_dash(f11, f12, f21, f22):
    return ((f11 + "-" + f12 == f21) | (f11 + "-" + f12 == f22) | (f21 + "-" + f22 == f11) | (
                f21 + "-" + f22 == f12)).astype(float)


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
