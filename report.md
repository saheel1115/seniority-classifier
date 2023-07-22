# Running the code
1. Create a conda environment with Python 3.10 using `conda create -n "eightfold" Python=3.10`. Doesn't have to be conda, of course. Any Python 3.10 environment should work.
2. Run `pip install -r requirements.txt` in your environment to install the dependencies.
3. Install CUDA 11.8 following these instructions: https://developer.nvidia.com/cuda-11-8-0-download-archive
    - Test the installation by opening a terminal and running `nvcc -V`
4. Install Pytorch with CUDA support by running: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`. 
5. Copy the train and test files into `data/`
6. Run the cells in the `SeniorityClassifier.ipynb` notebook


# Notes
- I have noted various assumptions and thoughts in the notebook as I worked on the problem and explored the data
- I decided to use XGBoost model because it works better with sparse features as compared to Random Forest
- As expected, due to lack of data points in the VP and CXO seniorities, we see pretty poor performance for these class in the confusion matrix
- There are a few optimizations and refactoring that can be done before productionizing this code:
    - Use PyTorch Datasets/Dataloaders library with the appropriate Data Transforms to streamline the data processing pipeline. This will allow us to scale beyond memory limitations.
    - We can maintain a list of special data processing rules -- as we include more and more training/test data, we can extend this list and use it easily in one of the transforms mentioned above.
    - Speaking of data processing, the feature extraction method is embarrassingly parallel. We can easily improve our feature extraction time by 5-10x by processing the data in parallel.
    - An alternative optimization during feature extraction would be to collect all the chunks of texts and process them in one go. And later, we can collect the embeddings/vectors and average them across all the chunks for a given data sample.
    - Perform a more thorough hyper-paramater search, perhaps using Optuna.
    - Currently, I am using Pickle files to save the features. In production, we would like to have a feature store that maps each candidate metadata to the appropriate features for easy retrieval.
    - For Model training, in case we are running into memory issues, we can distribute the training process across multiple CPUs/GPUs; for XGBoost, we can add on-disk support using something like Dask
    - Instead of hard-coding the usage of GPU (device 0) for inference, we can enable more dynamic and distributed inference capabilities, if needed.
