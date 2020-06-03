import glob, pandas as pd

df = pd.DataFrame({"Manifest": glob.glob("Manifests/*.txt")})

df["Body_Organ"] = df.Manifest.apply(lambda x: ' '.join(x[:x.rindex('_')].split('/')[1].split('_')))
df["Disease"] = df.Manifest.apply(lambda x: x[x.rindex('_')+1:][:-4])

df["Num_Cases"] = df.Manifest.apply(lambda x: pd.read_csv(x,sep='\t').filename.apply(lambda y:y[:12]).unique().__len__())
df["Num_Slides"] = df.Manifest.apply(lambda x: pd.read_csv(x,sep='\t').shape[0])
df["Cancer_Slides"] = df.Manifest.apply(lambda x: sum(pd.read_csv(x,sep='\t').filename.apply(lambda y:int(y.split('.')[0].split('-')[3][:2]))<=9))
df["Normal_Slides"] = df.Manifest.apply(lambda x: sum(pd.read_csv(x,sep='\t').filename.apply(lambda y:int(y.split('.')[0].split('-')[3][:2]))>9))
df["CAN_Ratio"] = df.Cancer_Slides/df.Num_Slides

df["Size_GB"] = df.Manifest.apply(lambda x: pd.read_csv(x,sep='\t')["size"].sum()*(2**-30))

df = df.sort_values(by="Body_Organ")
df.to_csv("summary.csv",index=False)
