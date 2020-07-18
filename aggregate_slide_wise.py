
import pandas as pd
import json

RECORD_FILE = "record.csv"


def main():
    df = pd.read_csv(RECORD_FILE)

    result_df = df.groupby("slide_ids", as_index=False).sum()[["slide_ids", "targets", "preds"]]
    result_df = result_df.rename(columns={'preds': 'num_tiles_pred_normal'})
    result_df["num_tiles"] = df.groupby("slide_ids", as_index=False).count()['targets'].tolist()
    result_df["targets"] = (result_df.targets>0).astype(int)
    result_df["num_tiles_pred_cancer"] = result_df.num_tiles-result_df.num_tiles_pred_normal
    result_df["tumor_fraction"] = result_df.num_tiles_pred_cancer/result_df.num_tiles
    result_df["pred_vote"] = (result_df.tumor_fraction<0.5).astype(int)
    result_df["pred_mil"] = (result_df.tumor_fraction==0).astype(int)

    def get_conf_mat(pred_col):
        tp = ((result_df[pred_col]==1) & (result_df.targets==1)).sum()
        fn = ((result_df[pred_col]==0) & (result_df.targets==1)).sum()
        fp = ((result_df[pred_col]==1) & (result_df.targets==0)).sum()
        tn = ((result_df[pred_col]==0) & (result_df.targets==0)).sum()
        return tp,fn,fp,tn

    def get_performance(tp,fn,fp,tn):
        precision_cancer = tn/(tn+fn)
        recall_cancer = tn/(tn+fp)
        precision_normal = tp/(tp+fp)
        recall_normal = tp/(tp+fn)
        acc = (tp+tn)/(tp+tn+fp+fn)
        return {"accuracy": acc,
                "precision_cancer": precision_cancer,
                "recall_cancer": recall_cancer,
                "precision_normal": precision_normal,
                "recall_normal": recall_normal
               }

    performances = {"VOTING": get_performance(*get_conf_mat("pred_vote")), "MIL": get_performance(*get_conf_mat("pred_mil"))}

    result_df.to_csv('slide_wise/slide_summary.csv', index=False)
    with open("slide_wise/conf_mats.json", 'w') as fh:
        tp,fn,fp,tn = get_conf_mat("pred_vote")
        conf_mat_voting = {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn)}
        tp,fn,fp,tn = get_conf_mat("pred_mil")
        conf_mat_MIL = {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn)}
        conf_mats = {"conf_mat_voting": conf_mat_voting, "conf_mat_MIL": conf_mat_MIL}
        json.dump(conf_mats, fh)
    with open("slide_wise/performances.json", 'w') as fh:
        json.dump(performances, fh)

    y_points_tumor, bins = np.histogram(result_df[result_df.targets==0].tumor_fraction,bins=10)
    x_points_tumor = (bins[:-1]+bins[1:])/2
    y_points_normal, bins = np.histogram(result_df[result_df.targets==1].tumor_fraction,bins=10)
    x_points_normal = (bins[:-1]+bins[1:])/2
    plt.plot(x_points_tumor,y_points_tumor,c='r', label='tumor')
    plt.plot(x_points_normal,y_points_normal,c='b', label='normal')
    plt.legend(); plt.grid(); plt.xlim(0,1); plt.ylim(0,1)
    plt.savefig('tumor_fraction.jpg')


if __name__=="__main__":
    main()
