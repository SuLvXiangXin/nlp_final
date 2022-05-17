import os

from models.MultiThreadingFeedForwardMLP import MultiThreadingFeedForwardMLP
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

def get_estimator(scorer_type, in_channels=11246):
    clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
    os.makedirs('mlp_models', exist_ok=True)
    if scorer_type == 'voting_mlps_hard':
        import sys
        seed = np.random.randint(1, sys.maxsize)
        mlp1 = MultiThreadingFeedForwardMLP(in_channels, n_classes=4, batch_size=188, hm_epochs=70, learning_rate=0.001,
                                            hidden_layers=(362, 942, 1071, 870, 318, 912, 247), seed=seed, name=1)

        seed = np.random.randint(1, sys.maxsize)
        mlp2 = MultiThreadingFeedForwardMLP(in_channels, n_classes=4, batch_size=188, hm_epochs=70, learning_rate=0.001,
                                            hidden_layers=(362, 942, 1071, 870, 318, 912, 247), seed=seed, name=2)


        seed = np.random.randint(1, sys.maxsize)
        mlp3 = MultiThreadingFeedForwardMLP(in_channels, n_classes=4, batch_size=188, hm_epochs=70, learning_rate=0.001,
                                            hidden_layers=(362, 942, 1071, 870, 318, 912, 247), seed=seed, name=3)

        seed = np.random.randint(1, sys.maxsize)
        mlp4 = MultiThreadingFeedForwardMLP(in_channels, n_classes=4, batch_size=188, hm_epochs=70, learning_rate=0.001,
                                            hidden_layers=(362, 942, 1071, 870, 318, 912, 247), seed=seed, name=4)

        seed = np.random.randint(1, sys.maxsize)
        mlp5 = MultiThreadingFeedForwardMLP(in_channels, n_classes=4, batch_size=188, hm_epochs=70, learning_rate=0.001,
                                            hidden_layers=(362, 942, 1071, 870, 318, 912, 247), seed=seed, name=5)

        clf = VotingClassifier(estimators=[
            ('mlp1', mlp1),
            ('mlp2', mlp2),
            ('mlp3', mlp3),
            ('mlp4', mlp4),
            ('mlp5', mlp5),
        ],  n_jobs=1,
            voting='hard')



    if scorer_type == 'MLP_base':
        clf = MultiThreadingFeedForwardMLP(in_channels, n_classes=4, batch_size=188, hm_epochs=70, learning_rate=0.001,
                                            hidden_layers=(362, 942, 1071, 870, 318, 912, 247), seed=seed, name=1)

    return clf