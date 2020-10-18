
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS, SpectralEmbedding, Isomap
import h5py
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os


def main():
    parser = argparse.ArgumentParser(description='Process args for t-SNE plot')
    parser.add_argument("--h5py_files_root", type=str, required=True)
    parser.add_argument("--test_sets", nargs='+', type=str, required=True)
    parser.add_argument("--model_organ", type=str, required=True)
    parser.add_argument("--points_to_use", type=int, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()
    print(args,flush=True)

    features, labels, organs = [], [], []
    for test_set in args.test_sets:
        fp = os.path.join(args.h5py_files_root, "{}.h5".format(test_set))
        assert os.path.isfile(fp)
        h5data = h5py.File(fp, 'r')
        feat_all, Y = h5data.get("embeddings").value, h5data.get("labels").value
        feat_C = feat_all[Y==0]
        feat_N = feat_all[Y==1]
        idx_C = np.random.randint(len(feat_C), size=(args.points_to_use,))
        idx_N = np.random.randint(len(feat_N), size=(args.points_to_use,))
        features.append(feat_C[idx_C])
        features.append(feat_N[idx_N])
        labels.append(np.array(["cancer"]*args.points_to_use))
        labels.append(np.array(["normal"]*args.points_to_use))
        organs.append(np.array([test_set]*(2*args.points_to_use)))

    features,labels,organs = np.concatenate(features),np.concatenate(labels),np.concatenate(organs)

    if args.method=="PCA":
        pca = PCA(n_components=2)
        points = pca.fit_transform(features)
    elif args.method=="KernelPCA":
        kpca = KernelPCA(n_components=2, kernel="rbf")
        points = kpca.fit_transform(features)
    elif args.method=="LDA":
        lda = LinearDiscriminantAnalysis(n_components=2)
        points = lda.fit(features, labels).tranform(features)
    elif args.method=="TSNE":
        tsne = TSNE(n_components=2, verbose=1, perplexity=500, n_iter=5000)
        points = tsne.fit_transform(features)
    elif args.method=="MDS":
        mds = MDS(n_components=2)
        points = mds.fit_transform(features)
    elif args.method=="Isomap":
        isomap = Isomap(n_neighbors=10, n_components=2)
        points = isomap.fit_transform(features)
    elif args.method=="SpectralEmbedding":
        emb = SpectralEmbedding(n_neighbors=10, n_components=2)
        points = emb.fit_transform(features)
    elif args.method=="LLE":
        emb = LocallyLinearEmbedding(n_neighbors=10, n_components=2, method="standard")
        points = emb.fit_transform(features)
    elif args.method=="LTSA":
        emb = LocallyLinearEmbedding(n_neighbors=10, n_components=2, method="ltsa")
        points = emb.fit_transform(features)
    elif args.method=="HessianLLE":
        emb = LocallyLinearEmbedding(n_neighbors=10, n_components=2, method="hessian")
        points = emb.fit_transform(features)
    elif args.method=="ModifiedLLE":
        emb = LocallyLinearEmbedding(n_neighbors=10, n_components=2, method="modified")
        points = emb.fit_transform(features)
    else:
        raise ValueError("Method Invalid.")


    plt.figure(figsize=(10,8))
    sns.scatterplot(x=points[:,0], y=points[:,1], hue=organs, style=labels, cmap='tab20')
    plt.savefig(args.outfile)


if __name__=="__main__":
    main()
