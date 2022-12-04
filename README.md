# Exploring Histological Similarities Across Cancers From a Deep Learning Perspective

This code is for our paper titled *Exploring Histological Similarities Across Cancers From a Deep Learning Perspective* published at Frontiers in Oncology 2021.


|   📝 Paper   |   📑 Demo Page  |    📑 Code for demo  | 
|-----------|-------------------|-------------------|
| [paper](https://www.frontiersin.org/articles/10.3389/fonc.2022.842759/full) | [website](https://bhasha.iiit.ac.in/tcga_cross_organ_project) | [demo code](https://github.com/piyush01123/tcga_app)



**Summary:** In this work, we trained 11 patch classifiers for the Cancer vs Normal task on 11 different cancer types. Then we perform cross inference. Further we use each of these classifiers to generate RoIs using GradCAM and measure overlap with respct to the model trained on each cancer type. Furthermore, we study the similarities in the histograms of geometric features within these RoIs to enhance this understanding even more.


**Abstract:** Histopathology image analysis is widely accepted as a gold standard for cancer diagnosis. The Cancer Genome Atlas (TCGA) contains large repositories of histopathology whole slide images spanning several organs and subtypes. However, not much work has gone into analyzing all the organs and subtypes and their similarities. Our work attempts to bridge this gap by training deep learning models to classify cancer vs. normal patches for 11 subtypes spanning seven organs (9,792 tissue slides) to achieve high classification performance. We used these models to investigate their performances in the test set of other organs (cross-organ inference). We found that every model had a good cross-organ inference accuracy when tested on breast, colorectal, and liver cancers. Further, high accuracy is observed between models trained on the cancer subtypes originating from the same organ (kidney and lung). We also validated these performances by showing the separability of cancer and normal samples in a high-dimensional feature space. We further hypothesized that the high cross-organ inferences are due to shared tumor morphologies among organs. We validated the hypothesis by showing the overlap in the Gradient-weighted Class Activation Mapping (GradCAM) visualizations and similarities in the distributions of nuclei features present within the high-attention regions.

## Results




## Summary of dataset used in project

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

