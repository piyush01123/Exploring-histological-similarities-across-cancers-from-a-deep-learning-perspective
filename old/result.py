
import pandas as pd
import glob
import re

log_file = "kirc_inference_log.txt"

txt = open(log_file, 'r').read().strip().split('\n')[3:-2]

# test_files = glob.glob("/ssd_scratch/cvit/medical_data/TCGA_KIRC/test/*/*/*.png")
# assert len(txt)==len(test_files)

df = pd.DataFrame(columns=["slide_id", "img_id", "gt", "py"])
paths = [re.search("\".*\"", line).group(0).strip("\"") for line in txt]
slide_ids = [re.search("TCGA-.*/", item).group(0).strip("/") for item in paths]
img_ids = [item.split('/')[-1] for item in paths]
gts = [int(tx[-1]) for tx in txt]
pys = [float(tx.split(', ')[-2].strip("prob=")) for tx in txt]

df = pd.DataFrame({"slide_id": slide_ids, "img_id": img_ids, "gt": gts, "py": pys})

df["pred_y"] = (df["py"]>0.5).astype(int)
res = df.groupby("slide_id", as_index=False).sum()[["slide_id", "gt", "pred_y"]]
count = df.groupby("slide_id", as_index=False).count()[["slide_id", "gt"]]
count = count.rename(columns={"gt": "count"})
res = pd.merge(res, count, how='inner', on='slide_id')
res["gt"] = (res["gt"]>0).astype(int)
res = res.rename(columns={"pred_y": "pred_1"})
res["pred_0"] = res["count"] - res["pred_1"]

# Heuristic 1: Voting
res["pred_vote"] = (res["pred_1"] > res["pred_0"]).astype(int)

# Heuristic 2: >=1 Positive instance in bag (Positive=Cancer)
res["pred_mil"] = (res["pred_0"] == 0 ).astype(int)

acc_vote = sum(res["pred_vote"]==res["gt"])/res.shape[0]*100
acc_mil = sum(res["pred_mil"]==res["gt"])/res.shape[0]*100

print("Acc Vote = ", acc_vote)
print("Acc MIL = ", acc_mil)

res[['slide_id','gt','count','pred_0','pred_1','pred_vote','pred_mil']].to_csv("verdict.csv", index=False)
