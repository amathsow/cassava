# Cassava Disease Detection (Kaggle competition)

The solutions I submitted for the Cassava Disease Detection competition.

I first hit the accuracy ceiling at 81% with MobileNetV2 trained for classification on the training set with heavy artificial data augmentation. 
Then I tried a few pretext learning/unsupervised learning techniques to take advantage of the extraimages dataset. Specifically, I tried Auto-colorization [1], Jigsaw puzzling [2] and Egomotion [3]. The first two did not really converge: my guesses are, respectively, that the dataset is basically composed on mostly green images so there wasn't much colorization to learn, and the images do not have strong structural properties (moslty just foliage) to help with jigsaw puzzling. The third approach did converge, and after transfer learning on to the training set it increased accuracy to about 85%.

I recognize a few issues in my solution, for instance I crop images to a square the size of the shorter edge of the picture, and then resize to 224x224: a better cropping can probably increase accuracy. I had some computational constraints, but I should've tried to use ResNets or DenseNets, at the cost of consistently reducing the batch size.
Competition winners used SE-ResNets and DenseNets on the labelled dataset, generated pseudo-labels for the unlabelled dataset, and trained again on the resulting merge.

Overall, my increase in accuracy from 81% to 85% based on just pretraining on an auxiliary ask shows to be promising

Datasets available at https://www.kaggle.com/c/cassava-disease/data

[1] *Learning Representations for Automatic Colorization.* Larsson, Gustav and Maire, Michael and Shakhnarovich, Gregory. ECCV 2016

[2] *Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles.* Noroozi, Mehdi and Favaro, Paolo. ECCV 2016

[3] *Learning to See by Moving.* Agrawal, Pulkit and Carreira, Joao and Malik, Jitendra. ICCV 2015