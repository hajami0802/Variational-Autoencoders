# Variational-Autoencoders with a Classifier Head

What are Variational-Autoencoders?

VAEs leverage probabilistic modeling to uncover the hidden representation of data. The encoder learns this hidden
representation, which is the compressed version of the data or latent space, by formulating the likelihood, which represents the probability of observing a specific data point given the VAE model. While VAEs don't directly calculate the exact likelihood due to its complexity, they employ a variational inference framework based on the concept of the latent space to approximate it.

<img width="433" alt="Screen Shot 2024-05-17 at 12 53 24 PM" src="https://github.com/hajami0802/Variational-Autoencoders-Classifier/assets/169827483/30060413-4770-48d0-920d-bc9f438c4fcf">

The fundamental structure of a variational autoencoder

What is the goal of this study?

This research compares the results of PCAEClassifier, and BetaVAEClassifier for automatically analyzing brain MRI scans from multiple sclerosis (MS) patients. The goal is to identify these WMLs and classify them into two classes, the class that contains the MRIs with white matter lesions, and the other class is for MRIs that do not have these lesions, ultimately helping diagnose and assess this disease.
The image dataset was modified by applying data augmentation, preprocessing methods, compression, and normalization. The performance of each model was then evaluated using the metrics accuracy, precision, recall, and F-score. 
The PCAEClassifier and BetaVAEClassifier with different sparsity regularization and beta values respectively, it's evident that the choice of hyperparameters significantly impacts the model's performance.
For the BetaVAEClassifier, the model's performance varied significantly with different beta values. With a beta value of 5, the accuracy stood at 83.56%, accompanied by the precision, recall, and F1-score of 84.13%, 82.52%, and 83.32%, respectively. Increasing the beta to 10 resulted in an accuracy of 85.56%, with precision peaking at 90.78%, albeit at the expense of lower recall (78.59%) and F1- score (84.25%). Finally, with a beta of 20, the model achieved an accuracy of 87.61%, maintaining a balanced performance with precision, recall, and F1-score hovering around 87%.
Similarly, for the PCAEClassifier, the model's performance varied with different sparsity parameters (S). With S equals to 5, the accuracy was 70.82%, with precision, recall, and F1-score at 78.11%, 57.57%, and 66.29%, respectively. Increasing the sparsity parameter to 10 led to a notable improvement in accuracy to 84.08%, with precision, recall, and F1-score showing similar increments. Finally, with S equals to 20, the model achieved the highest accuracy of 88.82%, accompanied by precision, recall, and F1-score of 90.38%, 86.40%, and 88.34%, respectively. 

These models' sensitivity to hyperparameters highlights the intricate balance required in model configuration, especially in tasks where precision and recall are crucial, such as anomaly detection. Particularly in medical diagnostics like multiple sclerosis (MS) classification, where accurately distinguishing between affected and unaffected individuals is paramount, the performance of these models holds significant promise.

Some more information and the results:

Deterministic encoder (PCAEClassifier) vs Probabilistic encoder (BetaVAEClassifier):

PCAEClassifier employs a fixed set of convolutional layers with learned filters, ensuring that the encoding process remains consistent across different input images, resulting in deterministic outputs. This deterministic approach streamlines the training process and may facilitate quicker convergence. On the other hand, Beta-VAE, a type of Variational Autoencoder (VAE), employs a Probabilistic Encoder. Unlike deterministic encoders, probabilistic encoders introduce stochastic elements into the encoding process, learning a distribution—often Gaussian—to represent encoded data. By capturing both mean features and variability, this probabilistic approach enriches the latent space, enabling it to have a broader spectrum of information about the input data.

Deterministic encoder vs Probabilistic encoder:

<img width="528" alt="Screen Shot 2024-05-17 at 12 50 26 PM" src="https://github.com/hajami0802/Variational-Autoencoders-Classifier/assets/169827483/ad00b7a7-8757-4924-9fa3-053fe4a3ea4b">


Principal convolutional autoencoder classifier architecture in a glance:

<img width="559" alt="Screen Shot 2024-05-17 at 12 38 56 PM" src="https://github.com/hajami0802/Variational-Autoencoders-Classifier/assets/169827483/fc9dcef6-f024-4120-ad9b-f4591cddad63">

PCAEClassifier architecture:

The architecture consists of three main components:
1. Encoder: A sequence of convolutional layers that extracts features from the input image and progressively reduces its dimensionality, resulting in the encoded representation. While the code snippet doesn't explicitly show spatial reduction, it can be achieved by using convolutional layers with strides greater than 1.
2. Decoder (Optional for Classification): A sequence of transposed convolutional layers that attempts to reconstruct the original image from the encoded representation. While not directly influencing classification, it can aid in regularization and potentially improve generalization.
3. Classifier: A set of fully connected layers that process the encoded representation and produce class probabilities for each predefined category.
In the PCAEClassifier architecture, the information used for classification comes from the encoded representation (encoded) generated by the encoder network. This encoded representation is a compressed version of the original input image, capturing the essential features relevant for distinguishing between the defined classes.

Beta Variational Autoencoder:

<img width="531" alt="Screen Shot 2024-05-17 at 12 41 42 PM" src="https://github.com/hajami0802/Variational-Autoencoders-Classifier/assets/169827483/e965766a-615d-44f0-8a0c-e30dcba9f63a">

The BetaVAEClassifier structure can be separated into three key components:

Encoder: This component takes the input image and processes it through convolutional and linear layers. Its objective is to extract the latent representation (z) that captures the essential features of the image.
Classifier Head: This additional layer receives the latent representation (z) from the encoder. It then utilizes a fully connected neural network to predict the class label (MS or noMS). The classifier capitalizes on the disentangled nature of the latent space, allowing it to focus on discriminative features for accurate classification.

Decoder (Optional):While not directly involved in classification, the decoder in a BetaVAEClassifier is often present. It takes the latent representation (z) and attempts to reconstruct the original image. 
This reconstruction serves two purposes:

• Regularization: It helps the model learn a more meaningful latent space by encouraging it to capture the essential information needed for reconstruction.

• Visualization: The reconstructed image can be used to visualize the latent space and understand what kind of information different dimensions encode.


The results:

<img width="498" alt="Screen Shot 2024-05-17 at 12 44 39 PM" src="https://github.com/hajami0802/Variational-Autoencoders-Classifier/assets/169827483/30053cb2-3469-47dc-a1b4-c606b2f29451">



References:

 Kingma, Diederik P., and Max Welling. "An introduction to variational autoencoders." Foundations and Trends® in Machine Learning 12.4 (2019): 307-392.
 
 Heaton, Jeff. "Ian Goodfellow, Yoshua Bengio, and Aaron Courville: Deep learning: The MIT press, 2016, 800 pp, ISBN: 0262035618." Genetic programming and evolvable machines 19.1
 (2018): 305-307.

 Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep learning. MIT press, 2016.

 Muslim, Ali M., et al. "Brain MRI dataset of multiple sclerosis with consensus manual lesion segmentation and patient meta information." Data in Brief 42 (2022): 108139.
 
 Carass, Aaron, et al. "Longitudinal multiple sclerosis lesion segmentation: resource and challenge." NeuroImage 148 (2017): 77-102.
