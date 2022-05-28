import pandas as pd
import os


def merge_data(stances_file, bodies_file, merged_file, train=True, path='../fnc-1'):
    """
    以名字为”Body ID“的列合并两个csv文件，并调整列的顺序为 ['Headline', 'articleBody', 'Stance']
    :param stances_file: 路径
    :param bodies_file: 路径
    :param merged_file: 合并文件的保存路径
    :param train: bool 是否是train data，若为否，则添加一列全为None的名称为”Stance“的列
    :return: None
    """
    stances_file_ = os.path.join(path, stances_file)
    bodies_file_ = os.path.join(path, bodies_file)
    stances = pd.read_csv(stances_file_).reset_index()
    bodies = pd.read_csv(bodies_file_)
    merged = pd.merge(stances, bodies, on='Body ID').sort_values('index').drop(['index', 'Body ID'], axis=1)
    merged = merged.loc[:, ['Headline', 'articleBody', 'Stance']]
    merged.to_csv(merged_file, index=0, encoding='utf_8')
    print("Saving is finished.", merged.shape)


if __name__ == '__main__':
    merge_data('train_stances.csv', 'train_bodies.csv', 'train_merged.csv', train=True)
    merge_data('competition_test_stances.csv', 'competition_test_bodies.csv', 'test_merged.csv', train=False)
