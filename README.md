# Leaf-classifier
A simple CNN network built by Keras was used to classify the leaves. There are six categories in the dataset, each with 50 images.

## required
python ≥ 3.5  
keras 2.4.3  
tensorflow 2.3.0  
numpy  
matplotlib  

## dataset
The original data is saved in the dataset_origin folder. There are six kinds of plant pictures: *Lotus magnolia tree*, *Redrlowered Loropetalum, maple*, *Camphora officinarum*, *cypress*, *ginkgo*, which are stored in six folders. There are two sizes of pictures: 369x800 and 450x800.  

Run gen_train_test_val.py to generate training set, verification set and test set. The default ratio is 8:1:1, which can be set by modifying parameters.

## train
Run train.py to train the model. The model and trained weight parameters will be saved in folder work_dirs.

## test
Run test.py to test the model.
