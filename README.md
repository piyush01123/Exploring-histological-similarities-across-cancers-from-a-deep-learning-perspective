# Exploring Histological Similarities Across Cancers From a Deep Learning Perspective

This code is for our paper titled *Exploring Histological Similarities Across Cancers From a Deep Learning Perspective* published at Frontiers in Oncology, Vol 12, 2022.

**Authors:** Piyush Singh*, Ashish Menon*, C. V. Jawahar, P. K. Vinod

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

## License and Citation
The software is licensed under the MIT License. Please cite the following paper if you have used this code:
```
@article{Menon2022ExploringHS,
  title={Exploring Histological Similarities Across Cancers From a Deep Learning Perspective},
  author={A Vipin Menon and Piyush Singh and P. K. Vinod and C.V. Jawahar},
  journal={Frontiers in Oncology},
  year={2022},
  volume={12}
}
```
