# MachineLearning

Machine Learning Model to Predict Income

## Required dependencies:

```bash
brew lightgbm
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.


```bash
pip install numpy, sklearn, pandas, seaborn, category_encoders
```

## Steps to Predict Output
In file FinalIncome.py, uncomment the model that you want to use for prediction (By default, it runs lightgbm)


Run the command python FinalIncome.py

ALet the model train & the predicted result in generated in the file New_tcd-ml-1920-group-income-submission.csv

## Project Flow

1. Reads the dataset provided to train the model \tcd-ml-1920-group-income-train.csv
2. Reads the dataset on which income output is to be predictedtcd-ml-1920-group-income-test.csv
3. Merge both the set and pre-process data (remove outlier/target mean encoding)
4. Split the dataset into train and test dataset
5. Performs training of the model on the training dataset
6. Returns CSV for the prediction of income on the test dataset : New_tcd-ml-1920-group-income-submission.csv

