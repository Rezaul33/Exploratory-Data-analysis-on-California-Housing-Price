# ABP-E2E-PROJECT

1. Tools
    - [Anaconda](https://www.anaconda.com/)
    - [VS Code](https://code.visualstudio.com/)
        - Python Extension Pack
        - Jupyter Notebook
        - Data Wrangler 

    

2. Steps
    - Install vs code extensions and dependencies
    - Create folders and files
        - Readme.md
        - Notebook
        - Data
    
3. Data Collection and Loading
    - Downloaded [California housing price dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices) from [Kaggle](https://www.kaggle.com/)
    - Load the dataset into [Pandas](https://pandas.pydata.org/docs/index.html) Dataframe
4. [Ydata Profile](https://docs.profiling.ydata.ai/latest/) generation
    - Generating a Data Profiling Report: Using ProfileReport from ydata_profiling (formerly pandas_profiling) to create a comprehensive report of the dataset. This report includes various statistics, distributions, correlations, and potential issues within the data.
    - Saving the Profiling Report: Using profile.to_file() to save the generated profiling report as an HTML file. This allows for a detailed and interactive exploration of the dataset.

5. Data cleaning & pre-Processing
    - Inspecting the first few rows: Displaying the first few rows of the dataset using df.head() to check its structure.
    - Inspecting dataset information: Using df.info() to check the data types and identify null values in the dataset.
    - Generating summary statistics: Using df.describe() to view statistical measures (mean, min, max, etc.) for numerical columns, helping to understand the dataset's distribution.
    - dentifying null values: Using df.isnull().any(axis=1) to find rows that contain null values.
    - Removing rows with null values: Using df.dropna() to remove all rows that contain any missing values.
    - Re-inspecting dataset after cleaning: Using df.info() again after removing null values to verify changes in the dataset structure.
    - Feature Selection: Using df[selected_columns] to filter and retain only specific columns from the dataset that are essential for analysis.
    - Visualizing outliers with box plots: Creating box plots for each selected column using plt.boxplot() to visually identify potential outliers in the dataset. Box plots help in detecting anomalies in the data distribution.
    - Detecting outliers using the IQR method:
        - Calculating the Interquartile Range (IQR) for each column using the first quartile (Q1) and third quartile (Q3).
        - Defining the lower bound as Q1 - 1.5 * IQR and the upper bound as Q3 + 1.5 * IQR.
        - Identifying outliers as values falling below the lower bound or above the upper bound.
        - Counting outliers: Using logical conditions to count the number of outliers in each column.
        - Removing rows with outliers: Using df.drop(index=outlier_rows) to create a new dataset without the rows that contain outliers.
        - Handling potential non-numeric columns: Attempting to convert columns to numeric format using pd.to_numeric(), and handling cases where conversion fails by skipping non-numeric columns.
        - Verifying the cleaned dataset: Displaying the shape of the new dataset without outliers to confirm the number of rows and columns remaining after outlier removal.
        - Created a new dataset without outliers named housing_no_outliers.csv
    - Selecting categorical columns for encoding: Identifying the ocean_proximity column as a categorical feature to be processed.
        - Applying one-hot encoding: Using pd.get_dummies() to convert the categorical values in the ocean_proximity column into binary indicator variables. The drop_first=True parameter is used to avoid multicollinearity by dropping the first category.
        - Saving the cleaned dataset: Storing the one-hot encoded DataFrame as a CSV file using df_encoded.to_csv(), ensuring the cleaned and processed dataset is saved for future use
    - Calculating the correlation matrix: Using df_encoded.corr() to compute the correlation matrix, which shows the pairwise correlations between numerical features in the dataset.
        - Visualizing correlations with a heatmap:
        - Using sns.heatmap() from Seaborn to create a heatmap for visualizing the correlation matrix.
        - The heatmap highlights correlations, where positive and negative relationships between variables can be quickly identified.
        - Setting a correlation threshold: Defining a threshold of 0.8 to identify highly correlated variables that should be considered for removal.
        - Identifying highly correlated variables: Iterating through the correlation matrix and selecting columns where the absolute value of the correlation exceeds the threshold, indicating high multicollinearity.
        - Dropping highly correlated variables: Using df_encoded.drop() to remove columns that are strongly correlated with others, reducing multicollinearity in the dataset.
        - Reordering columns: Moving the target variable, median_house_value, to the rightmost position in the dataset to separate it from the feature columns.
        - Visualizing the filtered correlation matrix: Creating a heatmap using Seaborn to visualize the correlations between the remaining variables after dropping highly correlated columns.
    
6. Primary Model Training and performance Evaluation
    - Defining features and target variable: Separating the dataset into features (X) and the target variable (y), with median_house_value as the target.
    - Splitting the data: Using train_test_split() to divide the data into training and testing sets (80% train, 20% test) for model evaluation.
    - Imputing missing values: Applying SimpleImputer with the 'mean' strategy to fill in any missing values in the features.
    - Standardizing features: Using StandardScaler to normalize the feature values, ensuring that each feature has a mean of 0 and a standard deviation of 1, which improves model performance.
    - Training a Linear Regression model: Fitting a LinearRegression model to the scaled training data.
    - Making predictions: Generating predictions on the test set using the trained linear regression model.
    - Evaluating the model: 
        - Calculating and printing the Mean Squared Error (MSE) to assess the model's performance on the test set.
        - Calculating Root Mean Squared Error (RMSE): Using np.sqrt() to compute the square root of the Mean Squared Error (MSE) to provide an error metric in the same units as the target variable. This helps to interpret the magnitude of prediction errors.
        - Calculating R-squared (R2) score: Using r2_score() to measure the proportion of variance in the target variable that is predictable from the features. This indicates the goodness-of-fit of the model.
    - Model Saving:

        - Joblib: Saves the trained Random Forest model to a file in the "../Model" directory using joblib.


7. Build Streamlit App
    - Imports:

        - [Streamlit](https://docs.streamlit.io/) for the app interface
        - [Joblib](https://joblib.readthedocs.io/en/stable/) for loading the model
        - pandas for data manipulation
    - Model Loading:

        - Load the Random Forest model from model/random_forest_model.pkl
    - Category Mapping:

        - Define a dictionary to map ocean_proximity categories to numerical values
    - Caching:

        - Use @st.cache decorator for the predict_price function to cache results and improve performance
    - Streamlit Interface:

    - Set up the title and header of the app
        - Create sliders and a select box in the sidebar for user input
    - Data Preparation:

        - Collect user inputs and prepare a DataFrame with these inputs
        - Convert ocean_proximity to its numerical category using the defined mapping
    - Feature Handling:

        - Ensure the features DataFrame includes all expected columns, adding any missing columns with default values
        - Reorder the columns to match the training data's column order
    - Prediction:

        - Call the predict_price function with the prepared features
        - Display the predicted housing price on the app