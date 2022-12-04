# Exploring Histological Similarities Across Cancers From a Deep Learning Perspective

This code is for our paper titled *Exploring Histological Similarities Across Cancers From a Deep Learning Perspective* published at Frontiers in Oncology 2021.


|   📝 Paper   |   📑 Demo Page  |    📑 Code for demo  | 
|-----------|-------------------|-------------------|
| [paper](https://www.frontiersin.org/articles/10.3389/fonc.2022.842759/full) | [website](https://bhasha.iiit.ac.in/tcga_cross_organ_project) | [demo code](https://github.com/piyush01123/tcga_app)



**Summary:** In this work, we trained 11 patch classifiers for the Cancer vs Normal task on 11 different cancer types. Then we perform cross inference. Further we use each of these classifiers to generate RoIs using GradCAM and measure overlap with respct to the model trained on each cancer type. Furthermore, we study the similarities in the histograms of geometric features within these RoIs to enhance this understanding even more.

## Architecture
### Patch CNN architecture
![arch_v2_lite(1)](https://user-images.githubusercontent.com/19518507/205519069-35e04a92-8058-4366-8169-9154dbf9624a.jpg)
### Nucleus geometry analysis workflow
![nuc_seg_wflow](https://user-images.githubusercontent.com/19518507/205519080-971f3fa8-bcc7-44ab-ab50-c7de419d97ec.jpg)

## Results
### Cross Inference Results
![grid_v2](https://user-images.githubusercontent.com/19518507/205519152-bf67eab2-9da3-450c-8f1a-a177366c7840.jpg)
### GradCAM Overlap
![gradcam_v2(1)](https://user-images.githubusercontent.com/19518507/205519432-1d137e08-d33e-4ec3-9d89-2047d9a9fce5.jpg)
### Nucleus geometry analysis
![BRCA_KICH_COAD_combined(1)](https://user-images.githubusercontent.com/19518507/205519359-ccc5d09e-c86e-423d-b9e0-a67aac92236a.jpg)

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

