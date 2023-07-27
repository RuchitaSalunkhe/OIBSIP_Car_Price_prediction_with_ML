# OIBSIP_Car_Price_prediction_with_ML
# Car Price Prediction 

## Introduction
This project aims to predict the price of a car based on various factors like brand goodwill, car features, horsepower, and mileage. The dataset used for this project is named "CarPrice" and contains several features related to cars along with their corresponding prices.

## Project Structure

The project contains the following files:

1. `CarPrice.csv`: The dataset containing car features and prices.

2. `Task_3_Car_price_prediction_with_ML.py`: The Python script that performs data preprocessing, model training, evaluation, and visualization.

3. `README.md`: This file, providing information about the project.

## Requirements

To run this project, you need the following libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using the following command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
````

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib

## Usage

1. Clone this repository to your local machine.

2. Ensure you have the required packages installed by running:

```bash
pip install pandas scikit-learn
```
4. Place the "CarPrice.csv" dataset in the same directory as the code.

5. Run the "data_preprocessing.py" script to preprocess the data.

6. The preprocessed data will be split into training and testing sets, which can be used for machine learning modeling.

## Code Explanation

- Step 1: Load the dataset using `pandas`.

- Step 2: Handle missing values (if any) using `data.dropna()`.

- Step 3: Split the dataset into features (X) and the target variable (y).

- Step 4: Encode categorical variables and scale numerical features using `OneHotEncoder` and `StandardScaler` from `scikit-learn`, respectively. The categorical variables are one-hot encoded, and the numerical features are scaled to have zero mean and unit variance.

- Step 5: Split the preprocessed data into training and testing sets using `train_test_split` from `scikit-learn`.


The code will perform data preprocessing, train a Linear Regression model, evaluate its performance, and provide the prediction output (optional: it will also perform hyperparameter tuning and training a Random Forest model).

## Output

The code will display various outputs, including:

- Information about the dataset, including the first few rows, missing values, and statistics of numerical features.

- Correlation heatmap between features and the target variable (price).

- Visualization of the distribution of car prices based on fuel type and car body type.

- Scatter plots showing the relationships between engine size, horsepower, city mpg, and the car price.

- Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) values for the Linear Regression model.

- (Optional) Best hyperparameters and evaluation metrics for the Random Forest model.

## Customize the Project

Feel free to customize the project as per your requirements. You can try different machine learning algorithms, feature engineering techniques, or visualize additional insights from the data.


