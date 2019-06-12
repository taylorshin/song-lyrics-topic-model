import os

OUT_DIR = 'out'

DATAFRAME_FNAME = 'dataframe.pkl'
TOKENS_TRAIN_FNAME = 'tokens_train.pkl'
TOKENS_TEST_FNAME = 'tokens_test.pkl'
FEATMAT_TRAIN_FNAME = 'feat_mat_train.npy'
FEATMAT_TEST_FNAME = 'feat_mat_test.npy'
VERIFY_LEARN_PLOT_FILE = os.path.join(OUT_DIR, 'perp_vs_iter.png')

STOP_WORDS = ['can', 'don', 're', 'll', 've', 'em', 'got', 'get']
