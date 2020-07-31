
import os, glob, pandas as pd
done = open("done.txt", 'r').read().split('\n')
done = [i.split('/')[-1] for i in done]

organs = ["Breast", "Colorectal", "Kidney", "Liver", "Lung", "Prostate", "Stomach"]
manifests = [glob.glob(os.path.join('Manifests/', organ+"*.txt")) for organ in organs]
manifests = [j for i in manifests for j in i]
print(manifests)

df = [pd.read_csv(fp, sep='\t') for fp in manifests]
df = pd.concat(df)
print(df.shape)

finished = df[df.filename.isin(done)]
unfinished = df[~df.filename.isin(done)]
unfinished = unfinished.reset_index(drop=True)
print(finished.shape, unfinished.shape)

for i in range(len(unfinished)//100+1):
    print("Partition", i)
    subset = unfinished.iloc[100*i:100*(i+1)]
    subset.to_csv("Partition_{}".format(i),sep='\t',index=False)
