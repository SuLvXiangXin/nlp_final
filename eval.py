from score import *
import argparse
import os

# get args
parser = argparse.ArgumentParser()
parser.add_argument('-n', help='team name', type=str, default='ours')
args = parser.parse_args()

# specify team name
team_name = args.n
if team_name == 'base':
    team_name = 'fnc-1-baseline'

# predict csv
src_dir = os.path.join(team_name, 'out.csv')

# target csv
tgt_dir = 'fnc-1/competition_test_stances.csv'
gold_labels = load_dataset(tgt_dir)
test_labels = load_dataset(src_dir)

# compute scores
test_score, cm = score_submission(gold_labels, test_labels)
null_score, max_score = score_defaults(gold_labels)

# print confusion matrix
print_confusion_matrix(cm)
print(SCORE_REPORT.format(max_score, null_score, test_score))
print('score: {:.2f}/{:.2f}={:.4f}'.format(test_score, max_score, test_score / max_score))
