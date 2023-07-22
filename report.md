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
