# Linear regression
The dataset we will use in the project contains information about the sales of used Fiat 500 cars in Italy, with the following columns:

***model***:  Car model

**Engine_power**: Engine power in horsepower

**transmission**: Type of gearbox

**age_in_days**: Car's age in days

**km**: Mileage in kilometers

**previous_owners**: Number of previous owners

**lat**: Latitude coordinate of the car's location

**lon**: Longitude coordinate of the car's location (since cars are more expensive further south)

**price**: Car price in euros
df = pd.read_csv("fiat.csv")
df.head()
Preliminary data analysis:
df.info()
We have two columns of type "object," which are the columns "model" and "transmission."
print(df['model'].unique())
print(df['transmission'].unique())
To transform the categorical variables "model" and "transmission" into numerical variables, we can use the following mappings:

For the "model" column:

'pop' = 4

'lounge' = 3

'sport' = 2

'star' = 1

For the "transmission" column:

'manual' = 0

'automatic' = 1
model_dict = {'pop':4, 'lounge':3, 'sport':2, 'star':1}
df['model'].replace(model_dict, inplace=True)
trans_dict = {'manual':0, 'automatic':1}
df['transmission'].replace(trans_dict, inplace=True)
df.info()

'age_in_days' as a characteristic of the car's age counted in days is not very readable, we will convert it to years.
df['age_in_years'] = np.round(df['age_in_days'] / 365)
df.drop(columns=['age_in_days'], inplace=True)
df['age_in_years'].head()

Missing values:
for column in df:
    print("\n" + column + ":" + str(df[column].isnull().sum()))
Since there aren't many observations, we decided to fill in the missing values with the mean.
for column in df:
    if df[column].isnull().sum() > 0:  
        mean_value = df[column].mean().round()  
        df[column].fillna(mean_value, inplace=True)
        print("\n" + column + ": " + str(mean_value))


for column in df:
    print("\n" + column + ":" + str(df[column].isnull().sum()))
We remove the ID column because it is a unique number that does not carry useful information for prediction. It might lead to overfitting as it does not provide any additional information.
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

We will check the statistics.
statistics = df.describe()
print("\nBasic statistics:\n", statistics)
Based on the provided statistics:

The average age of cars is around 9 years, with an average mileage of 102,196 kilometers.

Most cars had one or two previous owners.

The average engine power is about 73 horsepower.

The average price of cars in the dataset is 5,855 euros, with a wide range from 2,890 to 12,900 euros.


**Price Analysis:**
Feature visualization
num_columns = ['engine_power', 'age_in_years', 'km', 'previous_owners', 'price']

plt.figure(figsize=(15, 10))
for i, col in enumerate(num_columns):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()
In the dataset, cars with an engine power of around 70 horsepower dominate, typically having one or two previous owners. Additionally, the majority of cars are around 12 years old with mileage ranging between 50,000 and 150,000 kilometers, and car prices are mainly concentrated in the range of 4,000 to 6,000 euros.
### The correlation matrix:
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

print("\nCorrelation matrix:\n", correlation_matrix)
The price is strongly negatively correlated with the age of the car (-0.911764) and its mileage (-0.787404). In simple terms, older and more used cars tend to have lower prices.
plt.figure(figsize=(15, 10))
for i, col in enumerate([ 'age_in_years', 'km']):
    plt.subplot(2, 2, i + 1)
    sns.scatterplot(x=df[col], y=df['price'])
    plt.title(f'Price vs {col}')
plt.tight_layout()
plt.show()

As we can see, older cars have a lower price compared to newer ones, and cars with higher mileage have a lower price.
### The regression model
X = df[['engine_power', 'age_in_years', 'km']] #zmienne wejściowe
y = df['price'].copy() #zmienna wyjściowa
Splitting the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error (MSE):", mse)
print("R-squared (R²) score:", r2)

Model predicts the car price correctly in 87% of cases. Let's try to find something that will improve our model.
# Regression model with PCA
First, we preprocess the data
df_pca = pd.read_csv("fiat.csv")
df_pca['age_in_years'] = np.round(df_pca['age_in_days'] / 365)

df_pca.drop(columns=['age_in_days'], inplace=True)

for column in df_pca:
    if df_pca[column].isnull().sum() > 0:  
        mean_value = df_pca[column].mean().round()  
        df_pca[column].fillna(mean_value, inplace=True)
        print("\n" + column + ": " + str(mean_value))


if 'id' in df_pca.columns:
    df_pca.drop('id', axis=1, inplace=True)
X = df_pca.drop(columns=['price'])
y = df_pca['price']

categorical_cols = ['model', 'transmission']
numerical_cols = ['engine_power', 'age_in_years', 'km', 'previous_owners', 'lat', 'lon']

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95)),  
    ('regressor', LinearRegression())
])

Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'r2:{r2}')

We obtain 88%.

X_test_transformed = model.named_steps['preprocessor'].transform(X_test)
pca = model.named_steps['pca']
X_test_pca = pca.transform(X_test_transformed)

plt.figure(figsize=(10, 6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.colorbar(label='Predykcje cen')
plt.xlabel('Główna składowa 1')
plt.ylabel('Główna składowa 2')
plt.title('Wizualizacja PCA')
plt.show()

Principal Component 1 and 2 represent linear combinations of the original features (engine_power, age_in_years, km, previous_owners, lat, lon).
# RandomForest Regression

In this case, we do not preprocess the data; we use it as is from our first model.
X = df.drop(columns=['price'])
y = df['price']
Next, we scale the features.







scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Splitting the data into training and testing sets, then creating a Random Forest Regressor model.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
We make predictions and evaluate the model.
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R^2: {r2}')
As we can see, in the case of our three models, the Random Forest model is the most fitted, with an accuracy of 92%.
estimator = model.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(estimator, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.show()
