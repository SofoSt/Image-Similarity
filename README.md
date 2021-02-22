# Image-Similarity

The goal of this project is to find ways for comparing images in order to determine their similarity. This is done mainly for high-dimension images, thus in order to speed-up the process, we represent the images as vectors reduced in a lower dimension. For us to compare images, we have implemented 
 - A __bottleneck Neural Network__ in order to to autoencode the images, and represent them in a lower dimension
 - The __LSH algorithm__ in that space, as well as the original, in order to find the closest neighbours to the image
 - The __Earth Mover's Distance__ metric, in order to compare the images
 - A __clustering__ algorithm, based on Lloyd's, to categorize the images depending on a feature

The auto-encoder model consists of two different types of layers: the encoding and the decoding layers. The program created is a handy interface in order for the user to insert different values of several hyper-parameters and see the behavior of each model, with ultimate goal to chose one model that best handles the dataset given.

The classification model aims to classifying images in a category. To do so, it uses a pre-trained auto-encoder model, by taking advantage of its encoding layers, which are then connected to a fully connected layer, and then to an output one, aiming to the best possible classification of the images. Once again, multiple models can be trained, in order for the user to decide the best for the training set needs, and then, the best one will be used in order to predict the test dataset.


## Compiling and Running

### Autoencoder N.N.

In order to run the Autoencoder model and create the reduced space datasets, you should navigate to the directory Autoencoder, and run the file autoencoder.py, as following:
```
python3 reduce.py −d <inpdataset> −q <inpqueryset> −od <out dataset> −oq <outqueryset>
```

### LSH algorithm

In order to run the Similarity Search algorithm, you should navigate to the directory `Similarity Search `and run the following command:
```
make run lsh
```
By default, the datasets used are of reduced size, in order for the programs to run quickly. Shall you want to change to the original files, you should run the command:
```
make run lsh_big
```
**Note**: The files (reduced and regular) should be of the same size, otherwise then program is going to fail.

**Note:** Our lower space dimension files are produced with the pixels as little endians, for our convenience. Shall you want to run the program, you should provide the files created by us, or others with little endianness.

### Similarity Search with EMD metric

In order to run the Similarity Search with EMD, you should navigate to the directory `NearestNeighbour_EMD`, and then run the following command:
```
python3 search.py −d <trainingset> −q <testset> −l1 <training labels> −l2 <test labels>
```

**Note1:** The files should be in the original space

**Note2:** You should first install pulp

**Note3:** Because the datasets are big, we use 1000 features and 10 queries since the EMD takes a lot of time. We can change it again in lines 25-30.

**Note4:** We have results for 10 queries-1000 features and 100 queries-1000 features for 2x2 ie 4 clusters, 4x4 ie 16 clusters and 7x7 ie 49 clusters.

### Clustering on Reduced and Original Spaces

In order to create the clustering file from the classifier, you should navigate to the `Autoencoder` directory, and then run the following command:
```
python3 classifier.p -d <training set> −dl <training labels> −t <testset> − tl <test labels> −model <autoencoder h5>
```

A file produced from our best classifying model is stored in misc, and is used by our makefile in the clustering program.

In order to run the Clustering algorithm, you should navigate to the directory `Similarity Search` and run the following command:
```
make run cluster lloyds
```

By default, the datasets used are of reduced size, in order for the programs to run quickly. Shall you want to change to the original files, you should run the command:
```
make run cluster lloyds big
```

**Note:** The files (reduced, regular, NN) should be of the same size, otherwise the program is going to fail.

## More Implementation details and Results
They can be found in the Proejct [README](/README.pdf)

## License
This project is licensed under the MIT License - see the [LICENSE](/LICENCE) file for details

## Contributors

[Nikos Galanis](https://github.com/nikosgalanis) \
[Sofoklis Strompolas](https://github.com/SofoSt/)
