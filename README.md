# Fiat 500 Used Car Sales Analysis

## Project Description
The aim of this project is to analyze the sales data of used Fiat 500 cars in Italy. The analysis includes data preparation, exploration, linear regression modeling, the application of Principal Component Analysis (PCA) combined with linear regression, and the implementation of a Random Forest model.

## Data
The dataset contains information about various features of the cars, including model, engine power, transmission type, age, mileage, number of previous owners, location, and price.

### Data Columns:
- `model`: Car model
- `engine_power`: Engine power in horsepower
- `transmission`: Type of gearbox
- `age_in_days`: Car's age in days
- `km`: Mileage in kilometers
- `previous_owners`: Number of previous owners
- `lat`: Latitude coordinate of the car's location
- `lon`: Longitude coordinate of the car's location
- `price`: Car price in euros

## Data Preprocessing
- Replace categorical variables with numerical values:
  - Models: `{'pop': 4, 'lounge': 3, 'sport': 2, 'star': 1}`
  - Transmissions: `{'manual': 0, 'automatic': 1}`
- Convert age from days to years.
- Fill missing values with the mean of the column.

## Exploratory Data Analysis
- Visualize the distribution of numerical features using histograms.
- Display the correlation matrix to understand relationships between features.

## Linear Regression Model
- Prepare data by selecting relevant features.
- Split data into training and testing sets.
- Train the linear regression model.
- Evaluate the model using Mean Squared Error (MSE) and R-squared (R²) score.

## PCA and Linear Regression
- Preprocess the data using scaling and one-hot encoding.
- Apply PCA to reduce dimensionality.
- Train a linear regression model using the transformed data.

## Random Forest Model
- Prepare data by selecting relevant features.
- Split data into training and testing sets.
- Train the Random Forest model with the training data.
- Evaluate the model using Mean Squared Error (MSE) and R-squared (R²) score.
- Analyze feature importance to understand which features have the most impact on the car prices.
