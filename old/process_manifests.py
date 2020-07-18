
import pandas as pd

df = pd.read_csv("kidney_diagnostic_slide_or_FFPE_slide_SARC.txt", sep='\t')
fun = lambda x: x[:12]
df["case_id"] = df.filename.apply(fun)
