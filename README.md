# Reaching Nirvana: Maximizing the Margin in Both Euclidean and Angular Spaces
**Abstract:** The classification loss functions used in deep neural network classifiers can be grouped into two categories based on maximizing the margin in either Euclidean or angular spaces. Euclidean distances between sample vectors are used during classification for the methods maximizing the margin in Euclidean spaces whereas the Cosine similarity distance is used during the testing stage for the methods maximizing the margin in the angular spaces. This paper introduces a novel classification loss that maximizes the margin in both the Euclidean and angular spaces at the same time. This way, the Euclidean and Cosine distances will produce similar and consistent results and complement each other, which will in turn improve the accuracies. The proposed loss function enforces the samples of classes to cluster around the centers that represent them. The centers approximating classes are chosen from the boundary of a hypersphere, and the pair-wise distances between class centers are always equivalent. This restriction corresponds to choosing centers from the vertices of a regular simplex inscribed in a hypersphere. There is not any hyperparameter that must be set by the user in the proposed loss function, therefore the use of the proposed method is extremely easy for classical classification problems. Moreover, since the class samples are compactly clustered around their corresponding means, the proposed classifier is also very suitable for open set recognition problems where test samples can come from the unknown classes that are not seen in the training phase. Experimental studies show that the proposed method achieves the state-of-the-art accuracies on open set recognition despite its simplicity.

![Nirvana](https://user-images.githubusercontent.com/67793643/217524225-82240880-27c7-4918-ab12-2e9b1235f701.png)
In the proposed method, class samples are enforced to lie closer to the class-specific centers representing them, and the class centers are located on the boundary of a hypersphere. All the distances between the class centers are equivalent, thus there is no need to tune any margin term. The class centers form the vertices of a regular simplex inscribed in a hypersphere. Therefore, to separate $C$ different classes, the dimensionality of the feature space must be at least $C-1$. The figure on the left shows the separation of 2 classes in 1-D space, the middle figure depicts the separation of 3 classes in 2-D space, and the figure on the right illustrates the separation of 4 classes in 3-D space. For all cases, the centers are chosen from a regular $C-$ simplex.

![PDAM](https://user-images.githubusercontent.com/67793643/217527332-b7962b96-d864-4a0a-bd81-fb8002d7e3d8.png)
The plug and play module that will be used for increasing feature dimension. It maps $d-$dimensional feature vectors onto a much higher $(C-1)-$ dimensional space.
# 1. Requirements
## Environments
Following packages are required for this repo.

    - python 3.8+
    - torch 1.9+
    - torchvision 0.10+
    - CUDA 10.2+
    - scikit-learn 0.24+
    - catalyst 21.10+
    - mlxtend 0.19+
 ## Datasets
 Cifar datasets and synthetic datasets are under the directory **'data'**. To train the network on face verification you need to download **MS1MV3** dataset. We have used a subset of this dataset including 12K individuals. We can provide this dataset for the interested users. Please send an email to hakan.cevikalp@gmail.com if you want this specific dataset.
# 2. Training & Evaluation
## Synthetic Experiments
- For synthetic experiments, simply run **'main_dsc_cifar_synthetic.py'** for our deafult loss function, **'main_dsc_cifar_synthetic_hinge.py'** for the hinge loss, and **'main_dsc_cifar_synthetic_softmax.py'** for softmax loss function. They will produce the figures used in Fig. 3 given in the paper.
## Open Set Recognition
- For open set recognition, use the files under the **'NiravanaOpenSet'** directory. For opne set recognition, we use some other libararies we deveoped for open set recognition. For training simply run **xx.py** file and use **xx.py** file for testing. If you have any question, email to **hasansaribas48@gmail.com**.
## Closed Set Recognition
- Just use **'main_cifar.py'** to train the network for Cifar100 dataset. The dataset is already under **data** folder. For other datasets, just downlaod the datasets and revise the main file to use the specific dataset.
## Experiments by Using DAM
- 
# 3. Results
### The learned feature embeddings:
![embeddings](https://user-images.githubusercontent.com/67793643/217549694-0c4deabe-ed97-480f-8f29-2263534b0dda.png)
Learned feature representations of image samples: (a) the embeddings returned by the proposed method trained with the default loss function, (b) the embeddings returned by the proposed method trained with the hinge loss, (c) the embeddings returned by the proposed method trained with the softmax loss function.
### Semantically related feature embeddings:

![NirvanaHeatmap](https://user-images.githubusercontent.com/67793643/217550692-f3b65c68-9723-4fb5-ac1b-46d4fc3e32bf.png)
The distance matrix is computed by using the centers of the testing classes. The four classes that are not used in training are closer to their semantically related classes in the learned embedding space.




