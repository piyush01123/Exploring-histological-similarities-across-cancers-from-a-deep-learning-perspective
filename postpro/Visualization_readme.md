Follow the vis.sh to run the grad_Cam visualization.

Used the pytorch-cnn-visulaization repo for creating grad cam.

Looped across the dataset class (1000 samples of cancer and 1000 samples of normal) and run the grad cam on these images.

We get 4 results after grad cam for each image (Bounding box representation of heatmap, gray scale cam heat map , cam heatmap and cam on image heatmap).

The images are saved in a folder by name "model_organ".

I have saved cam visualization of all models inferred on best_organ(LIHC), worst organs(KIRC,KICH,KIRP) and the same organ as that of the model.