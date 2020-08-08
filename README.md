# Task T1

## Organs selected for T1 task in our Abstract

<table style="text-align: center">
<thead>
  <tr>
    <th>Primary Site</th>
    <th> Subtype</th>
    <th>Disease Medical Name</th>
    <th>Size (GB)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Breast</td>
    <td>BRCA</td>
    <td>Breast Invasive Carcinoma</td>
    <td>448.6</td>
  </tr>
  <tr>
    <td rowspan="2">Colorectal </td>
    <td>READ</td>
    <td>Rectum Adenocarcinoma</td>
    <td>55.5</td>
  </tr>
  <tr>
    <td>COAD</td>
    <td>Colon Adenocarcinoma</td>
    <td>149.8</td>
  </tr>
  <tr>
    <td  rowspan="3">Kidney </td>
    <td>KIRC</td>
    <td>Kidney Renal Clear Cell Carcinoma</td>
    <td>311.6</td>
  </tr>
  <tr>
    <td>KIRP</td>
    <td>Kidney Renal Papillary Cell Carcinoma</td>
    <td>95.5</td>
  </tr>
  <tr>
    <td>KICH</td>
    <td>Kidney Chromophobe</td>
    <td>180.2</td>
  </tr>
  <tr>
    <td>Liver</td>
    <td>LIHC</td>
    <td>Liver Hepatocellular Carcinoma</td>
    <td>101.2</td>
  </tr>
  <tr>
    <td  rowspan="2">Lung</td>
    <td>LUAD</td>
    <td>Lung Adenocarcinoma</td>
    <td>199.4</td>
  </tr>
  <tr>
    <td>LUSC</td>
    <td>Lung Squamous Cell Carcinoma</td>
    <td>184.2</td>
  </tr>
  <tr>
    <td>Prostate</td>
    <td>PRAD</td>
    <td>Prostate Adenocarcinoma</td>
    <td>143.4</td>
  </tr>
  <tr>
    <td>Stomach</td>
    <td>STAD</td>
    <td>Stomach Adenocarcinoma</td>
    <td>292.3</td>
  </tr>
  <tr>
    <td colspan="3">Total</td>
    <td>2161.7</td>
  </tr>
</tbody>
</table>


## T1.1: Patch CNN training and inference for any subtype

> Note: See `sbatch_scripts`  for running end-to-end.

Step 0: Install required libraries:

If you use `pip`:
```
pip install virtualenv
python3.5 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you use `anaconda`:
```
conda create -n venv python=3.5 anaconda
source activate yourenvname
pip install -r requirements.txt
```

Step 1: Move SVS files from storage server

Step 2: Patch extraction

Step 3: Divide into train,val,test

Step 4: Patch extraction

Step 5: Training and validation

Step 6: Inference

Step 7: MIL pooling (Useful for T2/T3 tasks)

## T1.2: t-SNE plot of embeddings colored by subtype and correlation matrix

Step 1: Save embeddings to HDF5 file

Step 2: Generate plots

## T1.3: Cross-subtype inference
TODO

## T1.4: Inference on other datasets
TODO


---

Link to old code for Patch-CNN which I wrote earlier to match Sairam's results:

<https://github.com/piyush-kgp/RCC-classifcation-and-survival-prediction-from-histopathology-images-using-deep-learning>
