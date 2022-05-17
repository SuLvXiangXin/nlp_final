import sys
from utils import printout_manager
import utils.estimator_definitions as esitmator_definitions
import argparse
import os
import utils.score_calculation as score_calculation
import numpy as np
from refs.utils.generate_test_splits import kfold_split, get_stances_for_folds
from refs.feature_engineering import refuting_features, polarity_features, hand_features, word_overlap_features
from refs.feature_engineering import gen_or_load_feats
from refs.feature_engineering import word_unigrams_5000_concat_tf_l2_holdout_unlbled_test, \
    NMF_cos_300_holdout_unlbled_test, \
    NMF_concat_300_holdout_unlbled_test, latent_dirichlet_allocation_25_holdout_unlbled_test, \
    latent_semantic_indexing_gensim_300_concat_holdout_unlbled_test, stanford_based_verb_noun_sim_1sent, \
    NMF_cos_50, latent_dirichlet_allocation_25, \
    latent_semantic_indexing_gensim_300_concat_holdout, NMF_concat_300_holdout, word_unigrams_5000_concat_tf_l2_holdout
from refs.feature_engineering_challenge import NMF_fit_all_incl_holdout_and_test, \
    latent_dirichlet_allocation_incl_holdout_and_test, latent_semantic_indexing_gensim_holdout_and_test, \
    NMF_fit_all_concat_300_and_test, word_ngrams_concat_tf5000_l2_w_holdout_and_test, NMF_fit_all, \
    latent_dirichlet_allocation, latent_semantic_indexing_gensim_test, NMF_fit_all_concat_300, \
    word_ngrams_concat_tf5000_l2_w_holdout
from refs.utils.score import LABELS, score_submission
from refs.utils.dataset import DataSet
from refs.utils.testDataset import TestDataSet
import csv

sys.path.append('../')


def get_args():
    ''' This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(description='Scorer pipeline')
    parser.add_argument('-p', '--pipeline_type', type=str, default='ftrain', nargs='+',
                        help='Pipeline Type (crossv,holdout,ftrain,ftest), e.g. -p crossv holdout', required=False)
    args = parser.parse_args()
    pipeline_type = args.pipeline_type
    return pipeline_type


def generate_features(stances, dataset, name, feature_list, features_dir):
    """
    Creates feature vectors out of the provided dataset
    """
    h, b, y, bodyId, headId = [], [], [], [], []

    feature_dict = {'overlap': word_overlap_features,
                    'refuting': refuting_features,
                    'polarity': polarity_features,
                    'hand': hand_features,
                    'stanford_wordsim_1sent': stanford_based_verb_noun_sim_1sent,
                    'word_unigrams_5000_concat_tf_l2_holdout_unlbled_test': word_unigrams_5000_concat_tf_l2_holdout_unlbled_test,
                    'NMF_cos_300_holdout_unlbled_test': NMF_cos_300_holdout_unlbled_test,
                    'NMF_concat_300_holdout_unlbled_test': NMF_concat_300_holdout_unlbled_test,
                    'latent_dirichlet_allocation_25_holdout_unlbled_test': latent_dirichlet_allocation_25_holdout_unlbled_test,
                    'latent_semantic_indexing_gensim_300_concat_holdout_unlbled_test': latent_semantic_indexing_gensim_300_concat_holdout_unlbled_test,
                    'NMF_fit_all_incl_holdout_and_test': NMF_fit_all_incl_holdout_and_test,
                    'latent_dirichlet_allocation_incl_holdout_and_test': latent_dirichlet_allocation_incl_holdout_and_test,
                    'latent_semantic_indexing_gensim_holdout_and_test': latent_semantic_indexing_gensim_holdout_and_test,
                    'NMF_fit_all_concat_300_and_test': NMF_fit_all_concat_300_and_test,
                    'word_ngrams_concat_tf5000_l2_w_holdout_and_test': word_ngrams_concat_tf5000_l2_w_holdout_and_test,
                    'NMF_fit_all': NMF_fit_all,
                    'word_ngrams_concat_tf5000_l2_w_holdout': word_ngrams_concat_tf5000_l2_w_holdout,
                    'latent_dirichlet_allocation': latent_dirichlet_allocation,
                    'latent_semantic_indexing_gensim_test': latent_semantic_indexing_gensim_test,
                    'NMF_fit_all_concat_300': NMF_fit_all_concat_300,
                    'NMF_cos_50': NMF_cos_50,
                    'latent_dirichlet_allocation_25': latent_dirichlet_allocation_25,
                    'latent_semantic_indexing_gensim_300_concat_holdout': latent_semantic_indexing_gensim_300_concat_holdout,
                    'NMF_concat_300_holdout': NMF_concat_300_holdout,
                    'word_unigrams_5000_concat_tf_l2_holdout': word_unigrams_5000_concat_tf_l2_holdout
                    }

    stanceCounter = 0
    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])
        bodyId.append(stance['Body ID'])
        headId.append(name + str(stanceCounter))
        stanceCounter += 1

    X_feat = []
    feat_list = []
    last_index = 0
    for feature in feature_list:
        print(feature)
        feat = gen_or_load_feats(feature_dict[feature], h, b, features_dir + "/" + feature + "." + name + '.npy',
                                 bodyId, feature, headId)
        feat_list.append((last_index, last_index + len(feat[0]), str(feature)))
        last_index += len(feat[0])
        X_feat.append(feat)
    X = np.concatenate(X_feat, axis=1)

    return X, y, feat_list


def generate_features_test(stances, dataset, name, feature_list, features_dir):
    """
    Equal to generate_features(), but creates features for the unlabeled test data
    """
    h, b, bodyId, headId = [], [], [], []

    feature_dict = {'overlap': word_overlap_features,
                    'refuting': refuting_features,
                    'polarity': polarity_features,
                    'hand': hand_features,
                    # 'stanford_wordsim_1sent': stanford_based_verb_noun_sim_1sent,
                    'word_unigrams_5000_concat_tf_l2_holdout_unlbled_test': word_unigrams_5000_concat_tf_l2_holdout_unlbled_test,
                    'NMF_cos_300_holdout_unlbled_test': NMF_cos_300_holdout_unlbled_test,
                    'NMF_concat_300_holdout_unlbled_test': NMF_concat_300_holdout_unlbled_test,
                    'latent_dirichlet_allocation_25_holdout_unlbled_test': latent_dirichlet_allocation_25_holdout_unlbled_test,
                    'latent_semantic_indexing_gensim_300_concat_holdout_unlbled_test': latent_semantic_indexing_gensim_300_concat_holdout_unlbled_test,
                    'NMF_fit_all_incl_holdout_and_test': NMF_fit_all_incl_holdout_and_test,
                    'latent_dirichlet_allocation_incl_holdout_and_test': latent_dirichlet_allocation_incl_holdout_and_test,
                    'latent_semantic_indexing_gensim_holdout_and_test': latent_semantic_indexing_gensim_holdout_and_test,
                    'NMF_fit_all_concat_300_and_test': NMF_fit_all_concat_300_and_test,
                    'word_ngrams_concat_tf5000_l2_w_holdout_and_test': word_ngrams_concat_tf5000_l2_w_holdout_and_test,
                    'NMF_fit_all': NMF_fit_all,
                    'word_ngrams_concat_tf5000_l2_w_holdout': word_ngrams_concat_tf5000_l2_w_holdout,
                    'latent_dirichlet_allocation': latent_dirichlet_allocation,
                    'latent_semantic_indexing_gensim_test': latent_semantic_indexing_gensim_test,
                    'NMF_fit_all_concat_300': NMF_fit_all_concat_300,
                    'NMF_cos_50': NMF_cos_50,
                    'latent_dirichlet_allocation_25': latent_dirichlet_allocation_25,
                    'latent_semantic_indexing_gensim_300_concat_holdout': latent_semantic_indexing_gensim_300_concat_holdout,
                    'NMF_concat_300_holdout': NMF_concat_300_holdout,
                    'word_unigrams_5000_concat_tf_l2_holdout': word_unigrams_5000_concat_tf_l2_holdout
                    }

    stanceCounter = 0
    for stance in stances:
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])
        bodyId.append(stance['Body ID'])
        headId.append(name + str(stanceCounter))
        stanceCounter += 1

    X_feat = []
    for feature in feature_list:
        print("calculate feature: " + str(feature))
        feat = gen_or_load_feats(feature_dict[feature], h, b, features_dir + "/" + feature + "_test." + name + '.npy',
                                 bodyId, feature, headId)
        X_feat.append(feat)
        print(len(feat))
    X = np.concatenate(X_feat, axis=1)
    return X


def save_model(clf, save_folder, filename):
    """
    Dumps a given classifier to the specific folder with the given name
    """
    import pickle
    path = os.path.join(save_folder, filename)
    with open(path, 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(save_folder, filename):
    """
    Loads and returns a classifier at the given folder with the given name
    """
    print("Warning: Make sure older models with this name have been trained on the same features! Otherwise,"
          "if the lengths of the features the model has been trained on, differ, an error will occur!")
    import pickle
    path = save_folder + filename
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def final_clf_training(Xs, ys, X_holdout, y_holdout, scorer_type, sanity_check=False):
    """
    Train final classifier on all of the data to prepare it for the prediction of the FNC-1's unlabeled data
    :param Xs: All the training data's feature vectors, split in their folds
    :param ys: All the training data's labels, split in their folds
    :param X_holdout: The holdout feature vectors
    :param y_holdout: The holdout labels
    :param scorer_type: the scorer type, e.g. MLB_base (see estimator_definitions.py in utils folder)
    :param sanity_check: If true, the trained classifier predicts the labels of the data it was trained on and prints out the score
    :return: the final classifier
    """

    # stack all the feature vectors of all the folds
    X_train = np.vstack(tuple([Xs[i] for i in range(10)]))
    y_train = np.hstack(tuple([ys[i] for i in range(10)]))

    # stack the holdout feature vectors on the feature vectors of all folds
    X_all = np.concatenate([X_train, X_holdout], axis=0)
    y_all = np.concatenate([y_train, y_holdout], axis=0)

    # define and create parent folder to save all trained classifiers into
    parent_folder = "mlp_models/"

    # get classifier and only pass a save folder if the classifier should be saved
    clf = esitmator_definitions.get_estimator(scorer_type, in_channels=X_all.shape[1])

    # fit the final classifier
    clf.fit(X_all, y_all)

    # save the model
    save_folder = 'mlp_models'
    filename = 'vote.pth'
    save_model(clf, save_folder, filename)  # save model with filename to specific folder

    # predict on the data the classifier was trained on => should give near perfect score
    if sanity_check == True:
        # get predicted and actual labels
        y_predicted = clf.predict(X_all)
        predicted = [LABELS[int(a)] for a in y_predicted]
        actual = [LABELS[int(a)] for a in y_all]

        # calc FNC score
        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)
        score = fold_score / max_fold_score

        # calc accuracy, f1 macro
        accuracy_stance = score_calculation.get_accuracy(y_predicted, y_all, stance=True)
        accuracy_related = score_calculation.get_accuracy(y_predicted, y_all, stance=False)
        f1_stance = score_calculation.get_f1score(y_predicted, y_all, stance=True)
        f1_related = score_calculation.get_f1score(y_predicted, y_all, stance=False)

        # printout results
        printout = printout_manager.get_holdout_printout(save_folder, accuracy_related, accuracy_stance, f1_related,
                                                         f1_stance, score)
        print("SANITY CHECK (predict on train data):")
        print(printout)
    return clf


def final_clf_prediction(data_path, features, features_dir, scorer_type, run_final_train, final_clf):
    """
    Run the prediction on the final model. In order to do that, the features vectors of the unlabeled FNC-1 data are
    generated first.
    :param data_path: data_path to the unlabeled stances and the corresponding bodies
    :param features: The feature list
    :param features_dir: The directory where the features are stored
    :param scorer_type: the scorer type, e.g. MLB_base (see estimator_definitions.py in utils folder)
    :param run_final_train: Sanity check: if the final classifier has been trained in this run, check if the prediction of it
    compared to the classifier that is being loaded in this method, are the same. If yes, they represent the same model.
    :param final_clf: The classifier that was trained in this run (IF a classifier was trained)
    :return:
    """

    d = TestDataSet(data_path)

    # generate features for the unlabeled testing set
    X_final_test = generate_features_test(d.stances, d, str("final_test"), features, features_dir)

    # define and create parent folder to save all trained classifiers into
    parent_folder = "mlp_models/"

    # load model [scorer_type]_final_2 classifier
    filename = 'vote.pth'
    load_clf = load_model(parent_folder, filename)  # TODO set the correct path to the classifier here

    print("Load model for final prediction of test set: " + parent_folder + scorer_type + "_final_2/" + filename)

    # predict classes and turn into labels
    y_predicted = load_clf.predict(X_final_test)
    predicted = [LABELS[int(a)] for a in y_predicted]

    # save the submission file, including the prediction for the labels
    with open("out.csv", 'w') as csvfile:
        fieldnames = ["Headline", "Body ID", "Stance"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        i = 0
        for stance in d.stances:
            writer.writerow(
                {'Headline': stance['Headline'], 'Body ID': stance['Body ID'], 'Stance': predicted[i]})
            i += 1


def pipeline():
    # define data paths
    data_path = '../fnc-1'
    splits_dir = "splits"
    features_dir = "features"
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    # get arguments for pipeline call
    pipeline_type = get_args()

    run_final_train = False
    if "ftrain" in pipeline_type:
        run_final_train = True  # train classifier on all the data available

    run_final_prediction = False
    if "ftest" in pipeline_type:
        run_final_prediction = True  # run prediction on test data provided by FNC-1 challenge

    # train the model / predict on basis of the model
    if True in [run_final_train, run_final_prediction]:
        d = DataSet(data_path)
        folds, hold_out = kfold_split(d, n_folds=10, base_dir=splits_dir)
        fold_stances, hold_out_stances = get_stances_for_folds(d, folds, hold_out)

        Xs = dict()
        ys = dict()

        feature_list = [
            # ORIGINAL FEATURES OF FNC-1 BEST SUBMISSION 3)
            ('voting_mlps_hard',
             ['overlap', 'refuting', 'polarity', 'hand', 'NMF_fit_all_incl_holdout_and_test',
              'latent_dirichlet_allocation_incl_holdout_and_test', 'latent_semantic_indexing_gensim_holdout_and_test',
              'NMF_fit_all_concat_300_and_test', 'word_ngrams_concat_tf5000_l2_w_holdout_and_test'],
             [])
        ]

        for scorer_type, features, non_bleeding_features in feature_list:

            # print classifier and features for this loop
            print(scorer_type)
            print(features)
            print(non_bleeding_features)

            # Load/Precompute all features now
            X_holdout, y_holdout, feat_indices = generate_features(hold_out_stances, d, "holdout", features,
                                                                   features_dir)
            for fold in fold_stances:
                Xs[fold], ys[fold], _ = generate_features(fold_stances[fold], d, str(fold), features, features_dir)
            print('done!!!!!!!!!!!!!!!!!!!!!!!')

            # Train the final classifer
            if run_final_train == True:
                final_clf = final_clf_training(Xs, ys, X_holdout, y_holdout, scorer_type, sanity_check=True)

            # Run the final classifier on the test data
            if run_final_prediction == True:
                if run_final_train == True:
                    final_clf_prediction(data_path, features, features_dir, scorer_type, run_final_train, final_clf)
                else:
                    final_clf_prediction(data_path, features, features_dir, scorer_type, run_final_train, None)


if __name__ == '__main__':
    pipeline()
