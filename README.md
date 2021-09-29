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
 <img width="1680" alt="Screen Shot 2021-09-29 at 9 18 22 PM" src="https://user-images.githubusercontent.com/79149164/135343693-afd3432a-abca-47d7-add9-b5e232d51953.png">

<img width="1680" alt="Screen Shot 2021-09-29 at 9 18 34 PM" src="https://user-images.githubusercontent.com/79149164/135343707-e359ce83-4620-4352-bf27-52de4faa4a0d.png">
<img width="1680" alt="Screen Shot 2021-09-29 at 9 18 45 PM" src="https://user-images.githubusercontent.com/79149164/135343747-2f241c78-97ed-46a4-8125-e155b85727a0.png">
<img width="1680" alt="Screen Shot 2021-09-29 at 9 19 00 PM" src="https://user-images.githubusercontent.com/79149164/135343765-66df2d85-3712-4e23-a5eb-610cb19b38bb.png">
<img width="1680" alt="Screen Shot 2021-09-29 at 9 19 09 PM" src="https://user-images.githubusercontent.com/79149164/135343778-69874d22-96ae-4056-9d20-8fb4d1ed9cd1.png">
<img width="1680" alt="Screen Shot 2021-09-29 at 9 19 20 PM" src="https://user-images.githubusercontent.com/79149164/135343795-4921eb90-f654-41f2-b0c0-95475cb53c3c.png">
<img width="1680" alt="Screen Shot 2021-09-29 at 9 19 28 PM" src="https://user-images.githubusercontent.com/79149164/135343802-d0f7ceed-3faf-4da4-b05e-63bd6199c82e.png">
<img width="1680" alt="Screen Shot 2021-09-29 at 9 19 34 PM" src="https://user-images.githubusercontent.com/79149164/135343854-f5030d7a-55c1-4ecc-975f-bc78c817fe9f.png">
<img width="1680" alt="Screen Shot 2021-09-29 at 9 19 40 PM" src="https://user-images.githubusercontent.com/79149164/135343867-fea81a60-0086-483c-a630-19ce18cec306.png">
<img width="1680" alt="Screen Shot 2021-09-29 at 9 19 46 PM" src="https://user-images.githubusercontent.com/79149164/135343890-0790205e-e38b-439d-95e1-53cab114f274.png">
<img width="1680" alt="Screen Shot 2021-09-29 at 9 19 54 PM" src="https://user-images.githubusercontent.com/79149164/135343901-6232e2b2-7fd4-470b-a5a8-f8144908448d.png">
<img width="1680" alt="Screen Shot 2021-09-29 at 9 20 03 PM" src="https://user-images.githubusercontent.com/79149164/135343916-4c286894-c22e-4db8-80c5-ec21a32b1784.png">
