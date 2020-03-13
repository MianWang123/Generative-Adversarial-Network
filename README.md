## Project Title

Generative Adversarial Network

### Goal and Process

The goal is to find similar pairs of reviews in the AmazonReviews dataset(108 Mb), also given any new review, this code will find the most similar one in the database.  

The process involves data preprocess (get rid of stopwords and punctuations), K-shingling of reviews (K=5), locally sensitive hashing (m times permutation & min-hashing & divided by b bands), compute Jaccard distance (1 - Jaccard similarity), and find similar pairs of reviews.

### Introduction

The Generative Adversarial Network (GAN) is established and trained here for for the UT Zappos50K.

### Prerequisites

Put 'amazonReviews.json' in the same docment with 'lsh.py'.  
The dataset can be found here https://drive.google.com/file/d/1UMAL2OULAEpdhlSUSgtUMy7ErxVuYNdR/view?usp=sharing

### Data Visualization

For Step 4, I randomly picked 10000 pairs of reviews, and draw the distribution of their Jaccard distance, from which we can have a glimpse at how the whole AmazonReviews distinguish from each other.
![Image](https://github.com/MianWang123/Information-Retrieval/blob/master/pics/Jaccard%20distance%20of%2010000%20pairs.png)  
For Step 5, The graph of probability of hit vs similarity with different parameters is plotted here so as to choose appropriate parameters, i.e. m permutations & b bands.  
![Image](https://github.com/MianWang123/Information-Retrieval/blob/master/pics/probability%20of%20hit.png)  
For Step 9, I draw the distribution of Jaccard similarity in neareast duplicates, we can see that their Jaccard Similarity are very high since they are all similar pairs.  
![Image](https://github.com/MianWang123/Information-Retrieval/blob/master/pics/Jaccard%20similarity%20distribution%20of%20nearest%20duplicates.png)  


