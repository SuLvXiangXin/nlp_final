from score import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-n', help='team name', type=str, default='ours')
args = parser.parse_args()
team_name = args.n
team_name = 'base'
if team_name=='base':
    team_name = 'fnc-1-baseline'
src_dir = '/SSD_DISK/users/guchun/nerf/nlp_final/others/fnc-1/deep_learning_model/deepoutput.csv'
# src_dir = os.path.join(team_name, 'out.csv')
# src_dir = '/SSD_DISK/users/guchun/nerf/nlp_final/team1/tree_model/tree_pred_cor2.csv'
tgt_dir = 'fnc-1/competition_test_stances.csv'
import pandas as pd
tst=pd.read_csv(src_dir)
gold_labels = load_dataset(tgt_dir)
test_labels = load_dataset(src_dir)
test_score, cm = score_submission(gold_labels, test_labels)
null_score, max_score = score_defaults(gold_labels)
print_confusion_matrix(cm)
print(SCORE_REPORT.format(max_score, null_score, test_score))
print('score: {:.2f}/{:.2f}={:.4f}'.format(test_score, max_score, test_score/max_score))

