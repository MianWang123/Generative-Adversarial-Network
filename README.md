## Project Title

Generative Adversarial Network


### Goal and Process

The goal is to find similar pairs of reviews in the AmazonReviews dataset(108 Mb), also given any new review, this code will find the most similar one in the database.  

The process involves data preprocess (get rid of stopwords and punctuations), K-shingling of reviews (K=5), locally sensitive hashing (m times permutation & min-hashing & divided by b bands), compute Jaccard distance (1 - Jaccard similarity), and find similar pairs of reviews.


### Introduction

The Generative Adversarial Network (GAN) is established and trained here for for the UT Zappos50K.  

The training process:  
1.   Create the latent vector z  
2.   Zero out the gradient for generator  
3.   Generate the fake image, and calculate generator loss  
4.   Backpropagate generator loss and step the generator optimizer  
5.   Zero out the gradient for discriminator  
6.   Calculate discriminator loss  
7.   Backpropagate discriminator loss and step the discriminator optimizer  

The generator loss function:  
<div align=center><img src="http://chart.googleapis.com/chart?cht=tx&chl= L_G = \frac{1}{n}\sum_{i=1}^{n}L_{CE}(D(G(z)), 1) " style="border:none;"></div>  
The discriminator loss function:  
<div align=center><img src="http://chart.googleapis.com/chart?cht=tx&chl= L_D = \frac{1}{2n}\sum_{i=1}^{n}L_{CE}(D(X_i), 1) %2B L_{CE}(D(G(z)), 0) " style="border:none;"></div>
 
For Least Square GAN, the loss functions should look like:
<div align=center><img src="http://chart.googleapis.com/chart?cht=tx&chl= \min _{D} V_{\mathrm{LSGAN}}(D)=\frac{1}{2} \mathbb{E}_{\boldsymbol{x} \sim p_{\mathrm{dnta}}(\boldsymbol{x})}\left[(D(\boldsymbol{x})-1)^{2}\right]%2B\frac{1}{2} \mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}\left[(D(G(\boldsymbol{z})))^{2}\right] " style="border:none;"></div>  
<div align=center><img src="http://chart.googleapis.com/chart?cht=tx&chl= \min _{G} V_{\mathrm{LSGAN}}(G)=\frac{1}{2} \mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}\left[(D(G(\boldsymbol{z}))-1)^{2}\right] " style="border:none;"></div>


### Prerequisites

Put 'amazonReviews.json' in the same docment with 'lsh.py'.  
The dataset can be found here https://drive.google.com/file/d/1UMAL2OULAEpdhlSUSgtUMy7ErxVuYNdR/view?usp=sharing


### Data Visualization



### Acknowledge  
Special thanks to CIS522 course's TA and professor, for providing the data set and guidance of the training process

