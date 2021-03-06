## Project Title

Generative Adversarial Network


### Task and Model

The task is to generate as similar images(fake images) as the given dataset(real images).  
The Generative Adversarial Network(GAN) is created here, to generate images for the UT Zappos50K Dataset.


### Prerequisites

I uploaded the zipped UT Zappos50K dataset(625 Mb) to Google drive, and used Google Colab to load & unzip the data, the dataset can be found here https://drive.google.com/file/d/1nYEgytPOkFyUjDQfBGzwCQbszf6OE143/view?usp=sharing. You can also directly upload the unzipped dataset to your Colab, just remember to change the path in Step1 -'DATASETS/UTZappos50K'.


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


### Data Visualization
For GAN, the generator loss looks like:   
<div align=center><img src="https://github.com/MianWang123/Generative-Adversarial-Network/blob/master/pics/g_loss.png" width='400'/></div>   
the discriminator loss looks like:   
<div align=center><img src="https://github.com/MianWang123/Generative-Adversarial-Network/blob/master/pics/d_loss.png" width='400'/></div>   

After training for only 10 epochs, the GAN model can generate images below:  
<div align=center><img src="https://github.com/MianWang123/Generative-Adversarial-Network/blob/master/pics/generator%20image.png" width='400'/></div>


### Acknowledge  
Special thanks to CIS522 course's TA and professor, for providing the data set and guidance of the training process

