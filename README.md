# 🚗Predicting the Sales Price of  Cars using Machine learning🚙
# Project Description
In this notebook, I attempted predicting the sales price of cars with machine learning while using certain features of the cars
## 1. Problem defintion
How well can a machine learning model predict the future sale price of a car, given its characteristics and previous examples of how much similar cars have been sold for?

## 2. Data
The data used for this project is available on Kaggle at https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho

## 3. Evaluation
The evaluation metric for this competion is the RMSE (root mean squared error) between the actuall and the predicted auction prices

## 4. Features
Kaggle provides a data dictionary detailing of all the features of the dataset. You can view this data dictionary at https://www.kaggle.com/nehalbiral/vehicle-dataset-from-cardekho

# Steps 
This project was carried out using the jupyter notebook in the miniconda environment. link: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
The steps involved included:
### Basic Importations
* This is importing all the tools need for the project including regular exploratory data analysis tools such as `numpy`, `pandas`, `matplotlib.pyplot`
* Importing model evaluation tools such as `train_test_split`, `RandomizedSearchCV`, `mean_absolute_error`
### Loading the Dataset
 * This loads your work dataset to jupyter notebook using `pd.read_csv('dataset_name.csv')`
 * Note your dataset should be in `.csv` format
 ### Data Exploration (exploratory data analysis or EDA)
 The purpose of EDA is to understand and know more about the dataset we are working on. Things to consider:

 1. The problem(s) / question(s) to be solved.
 2. Findout the kind of data we have and treat them accordingly.
 3. Check if there are missing data and discover the best way to deal with them.
 4. Findout the outliers and consider their significance.
 5. Consider if adding, changing, removing features would help us make optimum use of the data.
 ### Basic Visualization
 This presents features in our data in various visual forms such as `barchat`, `linegraph`
 `scattere plot`
 This help us to have a better understanding of trends and pattern in a given dataset
 ### Modeling 
 This involves training a machine learning model such as `Randomforestregressor` and `RandomizedSearchCV`, in this case to be able to make good prediction base on pattern the model was able to recognise from our dataset.
 * This include choosing a model;  https://scikit-learn.org/stable/tutorial/machine_learning_map/index.htm
 * Spliting the data in to `feature` variables (X) and `target` variable (y).
 * Spliting the data into `training dataset` and `test dataset` and also `validation dataset` sometimes
 * Instantiating the model example `RandomForestRegressor()`
 * Fitting the model to the training data set
 * Evaluating our trained model and making predicitons with the model
 * Comparing various model to find the best model for the project.
 ### Finding the features of importance
 The features of importance are the features that had significant contribution to the pattern the trained model was able to recognise in a given data.
 It is equally important to identify how various features correlate with each other. As this helps to know the extent of relationship between variables, which then helps us to make better decision for future course of actions.
 ### The Results
 * This shows us the predictions our model made, and how close or far  it is to the actual values.
 * The regression model evaluation metrics are equally very import as it shows us to what extent our machine learning model performed.
 * Three most common regression
  #### R^2 (pronounced r-squared) or coefficient of determination.
       What R-squared does: Compares your models predicitons to the mean of the targets. Values can range from negative infinity (a very poor model) to 1. For example, if all your model does is predict the mean of the targets, it's R^2 value would be 0. And if your mode perfectly predicts a range of numbers it's R^2 value would be 1.
  #### Mean absolute error (MAE)
      MAE is the average of the absolute differences between predictions and actual values. It gives you an idea of how wrong your models predicitons are.
  ####  Mean squared error (MSE)
      MSE is the square of the mean absolute differences between predicted values and actual values.
 
 ### Saving and loading the trained model
 Once a model is trained, it can be saved for future use. There two common ways of doing this.
 They are saving and loading machine learning models: 1. With Python's `pickle module` 2. With the `joblib` module
 # Conclusion
 Once the trained model is able to meet the evaluation metric for the project, it is then considered acceptable and can then be deployed or shared. 
 On the other hand if you did not hit the evaluation metric, you can do the following;
 * Collect more data
 * Try a better model
 * Try improve the current model
 
 # Thanks for reading! 🙏
 
