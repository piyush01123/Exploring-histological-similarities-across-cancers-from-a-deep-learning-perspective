
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
        TP = ((result_df[pred_col]==1) & (result_df.targets==1)).sum()
        FN = ((result_df[pred_col]==0) & (result_df.targets==1)).sum()
        FP = ((result_df[pred_col]==1) & (result_df.targets==0)).sum()
        TN = ((result_df[pred_col]==0) & (result_df.targets==0)).sum()
        return TP,FN,FP,TN

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
    performances = pd.DataFrame({"Method": ["Voting", "MIL"], })
    with open("slide_wise/performances.json", 'w') as fp:
        json.dump(performances, fp)

    with open("slide_wise/conf_mat_voting.json", 'w') as fp:
        json.dump(get_conf_mat("pred_vote"), fp)

    with open("slide_wise/conf_mat_MIL.json", 'w') as fp:
        json.dump(get_conf_mat("pred_mil"), fp)

    result.to_csv('slide_wise/summary.csv', index=False)


if __name__=="__main__":
    main()
