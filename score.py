import csv
import matplotlib.pyplot as plt
import torch
import pandas as pd

FIELDNAMES = ['Headline', 'Body ID', 'Stance']
LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
RELATED = LABELS[0:3]
SCORE_REPORT = """
MAX  - the best possible score (100% accuracy)
NULL - score as if all predicted stances were unrelated
TEST - score based on the provided predictions

||    MAX    ||    NULL   ||    TEST   ||\n||{:^11}||{:^11}||{:^11}||
"""


def score_submission(gold_labels, test_labels):
    score = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g['Stance'], t['Stance']
        if g_stance == t_stance:
            score += 0.25
            if g_stance != 'unrelated':
                score += 0.50
        if g_stance in RELATED and t_stance in RELATED:
            score += 0.25

        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

    return score, cm


def score_defaults(gold_labels):
    """
    Compute the "all false" baseline (all labels as unrelated) and the max
    possible score
    :param gold_labels: list containing the true labels
    :return: (null_score, best_score)
    """
    unrelated = [g for g in gold_labels if g['Stance'] == 'unrelated']
    null_score = 0.25 * len(unrelated)
    max_score = null_score + (len(gold_labels) - len(unrelated))
    return null_score, max_score


def load_dataset(filename):
    with open(filename) as fh:
        reader = csv.DictReader(fh)
        data = list(reader)
    return data


def print_confusion_matrix(cm):
    lines = ['CONFUSION MATRIX:']
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-" * line_len)
    lines.append(header)
    lines.append("-" * line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                   *row))
        lines.append("-" * line_len)
    lines.append("ACCURACY: {}/{}={:.3f}".format(hit, total, hit / total, ))
    print('\n'.join(lines))


def plot(path='ours/run/weight_0.25_1_rand/best.pth'):
    """ Plot the training procedure """
    losses, accs = torch.load(path, )['train']
    losses = pd.Series(losses).rolling(100).mean()
    accs = pd.Series(accs).rolling(100).mean()
    x = list(range(1, 1 + len(losses)))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, losses)
    ax1.set_ylabel('Loss')
    ax2 = ax1.twinx()
    ax2.plot(x, accs, 'r')
    ax2.set_xlim([0, len(losses)])
    ax2.set_ylabel('Acc')
    ax2.set_xlabel('Iteration')
    plt.savefig('training process')


if __name__ == '__main__':
    plot()
