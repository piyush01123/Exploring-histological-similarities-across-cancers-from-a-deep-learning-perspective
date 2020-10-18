
import numpy as np
from sklearn.manifold import TSNE
import h5py
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os


def main():
    parser = argparse.ArgumentParser(description='Process args for t-SNE plot')
    parser.add_argument("--h5py_files_root", nargs='+', type=str, required=True)
    parser.add_argument("--test_sets", nargs='+', type=str, required=True)
    parser.add_argument("--model_organ", type=str, required=True)
    parser.add_argument("--points_to_use", type=int, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    args = parser.parse_args()

    features, labels, organs = [], [], []
    for test_set in args.test_sets:
        fp = os.path.join(args.h5py_files_root, "{}.h5".format(test_set))
        assert os.path.isfile(fp)
        h5data = h5py.File(fp, 'r')
        feat_all, labels = h5data.get("embeddings").value, h5data.get("labels").value
        feat_C = feat_all[labels==0]
        feat_N = feat_all[labels==1]
        idx_C = np.random.randint(len(feat_C), size=(args.points_to_use,))
        idx_N = np.random.randint(len(feat_N), size=(args.points_to_use,))
        features.append(feat_C[idx_C])
        features.append(feat_N[idx_N])
        labels.append(np.array(["cancer"]*args.points_to_use))
        labels.append(np.array(["normal"]*args.points_to_use))
        organs.append(np.array([test_set]*(2*args.points_to_use)))

    features,labels,organs = np.concatenate(features),np.concatenate(labels),np.concatenate(organs)

    tsne = TSNE(n_components=2, verbose=1, perplexity=500, n_iter=5000)
    points = tsne.fit_transform(features)

    plt.figure(figsize=(10,8))
    sns.scatterplot(x=points[:,0], y=points[:,1], hue=organs, style=labels, cmap='tab20')
    plt.savefig(args.outfile)


if __name__=="__main__":
    main()
