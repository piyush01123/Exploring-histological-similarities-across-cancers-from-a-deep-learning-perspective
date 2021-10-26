import numpy as np, sklearn.metrics as metrics, matplotlib.pyplot as plt, pandas as pd, os, glob
subtypes = set([f[:4] for f in os.listdir("records")])
subtypes = sorted(subtypes)
for data in subtypes:
    fig = plt.figure(figsize=(10,6))
    plots = []
    for model in subtypes:
        f = "records/{}_Model_{}_record.csv".format(model, data)
        df = pd.read_csv(f)
        gt, pred = df.targets, df.p_normal
        fpr, tpr, thresholds = metrics.roc_curve(gt, pred)
        plot=plt.plot(fpr, tpr, label="{} model".format(model))
        plots.append(plot)
    plt.grid(linestyle="dotted")
    plt.legend(loc='upper right')
    plt.xlabel("False Positive Rate (Positive Label: Normal)")
    plt.ylabel("True Positive Rate (Positive Label: Normal)")
    plt.title("ROC Curve for {} patches as predicted by all models".format(data))
    fig.savefig("roc_curves/{}_patches_roc_curve.png".format(data))
    plt.clf()
    plt.cla()
    print( data, "done")


