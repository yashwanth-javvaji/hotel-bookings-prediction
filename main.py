# File Handling
import os

# Data manipulation
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Resampling
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

# K-Fold Cross Validation
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, cross_val_score

# Custom implementations
from custom_implementations.logistic_regression import LogisticRegression as CustomLogisticRegression
from custom_implementations.naive_bayes import NaiveBayes as CustomNaiveBayes
from custom_implementations.k_nearest_neighbors import KNearestNeighbors as CustomKNearestNeighbors
from custom_implementations.decision_tree import DecisionTree as CustomDecisionTree
from custom_implementations.bagging_decision_tree import BaggingDecisionTree as CustomBaggingDecisionTree
from custom_implementations.boosting_decision_tree import BoostingDecisionTree as CustomBoostingDecisionTree

# SciKit-Learn implementations
from sklearn.linear_model import LogisticRegression as SciKitLearnLogisticRegression
from sklearn.naive_bayes import MultinomialNB as SciKitLearnNaiveBayes
from sklearn.neighbors import KNeighborsClassifier as SciKitLearnKNearestNeighbors
from sklearn.tree import DecisionTreeClassifier as SciKitLearnDecisionTree
from sklearn.ensemble import RandomForestClassifier as SciKitLearnRandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier as SciKitLearnAdaBoostClassifier

# Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


def load_data(num_samples=None):
    # read the data
    file_path = os.path.join(".", "data", "hotel_bookings.csv")
    df = pd.read_csv(file_path)

    # keep only the random number of samples
    if num_samples is not None:
        df = df.sample(num_samples)

    # view the first 5 rows
    print("The first 5 rows of the data are: ")
    print(df.head())
    print("-" * 100)

    # view the last 5 rows
    print("The last 5 rows of the data are: ")
    print(df.tail())
    print("-" * 100)

    # describe the data
    print("The data is described as: ")
    print(df.describe())
    print("-" * 100)

    # view the information about the data
    print("The data information is: ")
    print(df.info())
    print("-" * 100)

    # view the number of rows and columns
    print("The number of rows and columns are: ")
    print(df.shape)
    print("-" * 100)

    # view the columns of the data
    print("The columns of the data are: ")
    print(df.columns)
    print("-" * 100)

    # view the data types of the columns
    print("The data types of the columns are: ")
    print(df.dtypes)
    print("-" * 100)

    # view the unique values of the columns
    print("The unique values of the columns are: ")
    print(df.nunique())
    print("-" * 100)

    # view the missing values of the columns
    print("The missing values of the columns are: ")
    print(df.isnull().sum())
    print("-" * 100)

    # view the number of missing values of the columns
    print("The number of missing values of the columns are: ")
    print(df.isnull().sum().sum())
    print("-" * 100)

    return df


def clean_data(df):
    # removing anomalous data

    # Case 1
    # Duplicate rows: Duplicate rows with identical values in all columns could indicate data duplication or recording errors, and may need to be addressed during data cleaning.
    # Print the number of duplicate rows
    print("The number of duplicate rows are: ")
    print(df.duplicated().sum())
    print("-" * 100)
    # Drop the duplicate rows
    df.drop_duplicates(inplace=True)

    # Case 2
    # Missing or null values: Missing or null values in any of the columns could indicate incomplete or inconsistent data, and may require imputation or removal during data cleaning.
    # For children column, replace missing values with 0
    df['children'].fillna(0, inplace=True)
    # For country column, replace missing values with 'Unknown'
    df['country'].fillna('Unknown', inplace=True)
    # For agent and company columns, replace missing values with 0
    df['agent'].fillna(0, inplace=True)
    df['company'].fillna(0, inplace=True)
    # For all remaining rows with missing values, drop the row
    df.dropna(inplace=True)

    # Case 3
    # Adult, children, and babies all having a value of 0: This could indicate that a reservation or booking record is missing information or has incorrect data, as it is unlikely for a reservation to have no guests (adults, children, or babies).
    zero_guests_mask = (df['adults'] == 0) & (
        df['children'] == 0) & (df['babies'] == 0)
    zero_guests_rows = df[zero_guests_mask]
    # Print the number of rows where all guests are 0
    print("The number of rows where all guests are 0 are: ")
    print(zero_guests_rows.shape[0])
    print("-" * 100)
    # Drop rows where all guests are 0
    df.drop(zero_guests_rows.index, inplace=True)

    # Return the cleaned data
    return df


def exploratory_data_analysis(df):
    # Univariate Analysis

    # Distribution of numerical variables
    # Get numerical columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    # Loop through numerical columns
    for col in numerical_columns:
        # Get the range of data in the column
        data_range = df[col].max() - df[col].min()
        # Set the number of bins as the range of data
        num_bins = int(data_range) + 1
        # Create histogram
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], bins=num_bins, kde=True)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {col}')
        plt.show()

    # Distribution of categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x=col)
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.title(f'Count of {col}')
        plt.show()

    # Bivariate Analysis

    # Display the number of guests by country among not cancelled bookings in a choropleth map
    # Filter the data to only include rows where is_canceled is 0
    not_canceled_mask = df['is_canceled'] == 0
    not_canceled_df = df[not_canceled_mask]
    # Group the data by country and sum the number of guests
    guests_by_country = not_canceled_df.groupby(
        'country')[['adults', 'children', 'babies']].sum()
    # Reset the index
    guests_by_country.reset_index(inplace=True)
    # Rename the columns
    guests_by_country.rename(columns={
                             'adults': 'total_adults', 'children': 'total_children', 'babies': 'total_babies'}, inplace=True)
    # Create a column for total number of guests
    guests_by_country['total_guests'] = guests_by_country['total_adults'] + \
        guests_by_country['total_children'] + guests_by_country['total_babies']
    # Sort the data by total number of guests
    guests_by_country.sort_values(
        by='total_guests', ascending=False, inplace=True)
    # Create a choropleth map
    guests_map = px.choropleth(guests_by_country, locations='country', color='total_guests',
                               hover_name='country', color_continuous_scale=px.colors.sequential.Plasma)
    # Add a title
    guests_map.update_layout(
        title_text='Number of Guests by Country among Not Cancelled Bookings', title_x=0.5)
    # Display the map
    guests_map.show()

    # Display the number of guests by country among cancelled bookings in a choropleth map
    # Filter the data to only include rows where is_canceled is 1
    canceled_mask = df['is_canceled'] == 1
    canceled_df = df[canceled_mask]
    # Group the data by country and sum the number of guests
    guests_by_country = canceled_df.groupby(
        'country')[['adults', 'children', 'babies']].sum()
    # Reset the index
    guests_by_country.reset_index(inplace=True)
    # Rename the columns
    guests_by_country.rename(columns={
                             'adults': 'total_adults', 'children': 'total_children', 'babies': 'total_babies'}, inplace=True)
    # Create a column for total number of guests
    guests_by_country['total_guests'] = guests_by_country['total_adults'] + \
        guests_by_country['total_children'] + guests_by_country['total_babies']
    # Sort the data by total number of guests
    guests_by_country.sort_values(
        by='total_guests', ascending=False, inplace=True)
    # Create a choropleth map
    guests_map = px.choropleth(guests_by_country, locations='country', color='total_guests',
                               hover_name='country', color_continuous_scale=px.colors.sequential.Plasma)
    # Add a title
    guests_map.update_layout(
        title_text='Number of Guests by Country among Cancelled Bookings', title_x=0.5)
    # Display the map
    guests_map.show()

    # Price variation over time (trend) by hotel type among not cancelled bookings in a line plot
    # Filter the data to only include rows where is_canceled is 0
    not_canceled_mask = df['is_canceled'] == 0
    not_canceled_df = df[not_canceled_mask]
    # Group the data by hotel and arrival_date_year and get the average price
    price_by_year = not_canceled_df.groupby(
        ['hotel', 'arrival_date_year'])[['adr']].mean()
    # Reset the index
    price_by_year.reset_index(inplace=True)
    # Rename the columns
    price_by_year.rename(columns={'adr': 'average_price'}, inplace=True)
    # Create a line plot of price by year and hotel type
    price_by_year_line_plot = px.line(price_by_year, x='arrival_date_year',
                                      y='average_price', color='hotel', title='Average Price by Year and Hotel Type')
    # Add a title
    price_by_year_line_plot.update_layout(
        title_text='Average Price by Year and Hotel Type among Not Cancelled Bookings', title_x=0.5)
    # Display the plot
    price_by_year_line_plot.show()

    # Price variation over time (trend) by hotel type among cancelled bookings in a line plot
    # Filter the data to only include rows where is_canceled is 1
    canceled_mask = df['is_canceled'] == 1
    canceled_df = df[canceled_mask]
    # Group the data by hotel and arrival_date_year and get the average price
    price_by_year = canceled_df.groupby(
        ['hotel', 'arrival_date_year'])[['adr']].mean()
    # Reset the index
    price_by_year.reset_index(inplace=True)
    # Rename the columns
    price_by_year.rename(columns={'adr': 'average_price'}, inplace=True)
    # Create a line plot of price by year and hotel type
    price_by_year_line_plot = px.line(price_by_year, x='arrival_date_year',
                                      y='average_price', color='hotel', title='Average Price by Year and Hotel Type')
    # Add a title
    price_by_year_line_plot.update_layout(
        title_text='Average Price by Year and Hotel Type among Cancelled Bookings', title_x=0.5)
    # Display the plot
    price_by_year_line_plot.show()

    # Price variation over time (seasonality) by hotel type among not cancelled bookings in a line plot
    # Filter the data to only include rows where is_canceled is 0
    not_canceled_mask = df['is_canceled'] == 0
    not_canceled_df = df[not_canceled_mask]
    # Group the data by hotel and arrival_date_month and get the average price
    price_by_month = not_canceled_df.groupby(
        ['hotel', 'arrival_date_month'])[['adr']].mean()
    # Reset the index
    price_by_month.reset_index(inplace=True)
    # Rename the columns
    price_by_month.rename(columns={'adr': 'average_price'}, inplace=True)
    # Define the order of the months
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    # Sort the data by arrival_date_month
    price_by_month['arrival_date_month'] = pd.Categorical(
        price_by_month['arrival_date_month'], categories=month_order, ordered=True)
    price_by_month.sort_values(by='arrival_date_month', inplace=True)
    # Create a line plot of average price vs month and hotel type
    price_by_month_line_plot = px.line(price_by_month, x='arrival_date_month',
                                       y='average_price', color='hotel', title='Average Price by Month and Hotel Type')
    # Add a title
    price_by_month_line_plot.update_layout(
        title_text='Average Price by Month and Hotel Type among Not Cancelled Bookings', title_x=0.5)
    # Display the plot
    price_by_month_line_plot.show()

    # Price variation over time (seasonality) by hotel type among cancelled bookings in a line plot
    # Filter the data to only include rows where is_canceled is 1
    canceled_mask = df['is_canceled'] == 1
    canceled_df = df[canceled_mask]
    # Group the data by hotel and arrival_date_month and get the average price
    price_by_month = canceled_df.groupby(
        ['hotel', 'arrival_date_month'])[['adr']].mean()
    # Reset the index
    price_by_month.reset_index(inplace=True)
    # Rename the columns
    price_by_month.rename(columns={'adr': 'average_price'}, inplace=True)
    # Define the order of the months
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    # Sort the data by arrival_date_month
    price_by_month['arrival_date_month'] = pd.Categorical(
        price_by_month['arrival_date_month'], categories=month_order, ordered=True)
    price_by_month.sort_values(by='arrival_date_month', inplace=True)
    # Create a line plot of average price vs month and hotel type
    price_by_month_line_plot = px.line(price_by_month, x='arrival_date_month',
                                       y='average_price', color='hotel', title='Average Price by Month and Hotel Type')
    # Add a title
    price_by_month_line_plot.update_layout(
        title_text='Average Price by Month and Hotel Type among Cancelled Bookings', title_x=0.5)
    # Display the plot
    price_by_month_line_plot.show()

    # Price variation by year and hotel type among not cancelled bookings in a box plot
    # Filter the data to only include rows where is_canceled is 0
    not_canceled_mask = df['is_canceled'] == 0
    not_canceled_df = df[not_canceled_mask]
    # Create a box plot of price vs hotel type
    price_by_hotel_box_plot = px.box(
        not_canceled_df, x='arrival_date_year', y='adr', color='hotel', title='Price by Hotel Type')
    # Add a title
    price_by_hotel_box_plot.update_layout(
        title_text='Price by Hotel Type among Not Cancelled Bookings', title_x=0.5)
    # Display the plot
    price_by_hotel_box_plot.show()

    # Price variation by year and hotel type among cancelled bookings in a box plot
    # Filter the data to only include rows where is_canceled is 1
    canceled_mask = df['is_canceled'] == 1
    canceled_df = df[canceled_mask]
    # Create a box plot of price vs hotel type
    price_by_hotel_box_plot = px.box(
        canceled_df, x='arrival_date_year', y='adr', color='hotel', title='Price by Hotel Type')
    # Add a title
    price_by_hotel_box_plot.update_layout(
        title_text='Price by Hotel Type among Cancelled Bookings', title_x=0.5)
    # Display the plot
    price_by_hotel_box_plot.show()

    # Number of bookings variation over time (trend) by hotel type among not cancelled bookings in a line plot
    # Filter the data to only include rows where is_canceled is 0
    not_canceled_mask = df['is_canceled'] == 0
    not_canceled_df = df[not_canceled_mask]
    # Group the data by hotel and arrival_date_year and get the number of bookings
    bookings_by_year = not_canceled_df.groupby(['hotel', 'arrival_date_year'])[
        ['is_canceled']].count()
    # Reset the index
    bookings_by_year.reset_index(inplace=True)
    # Rename the columns
    bookings_by_year.rename(
        columns={'is_canceled': 'number_of_bookings'}, inplace=True)
    # Create a line plot of number of bookings by year and hotel type
    bookings_by_year_line_plot = px.line(bookings_by_year, x='arrival_date_year',
                                         y='number_of_bookings', color='hotel', title='Number of Bookings by Year and Hotel Type')
    # Add a title
    bookings_by_year_line_plot.update_layout(
        title_text='Number of Bookings by Year and Hotel Type among Not Cancelled Bookings', title_x=0.5)
    # Display the plot
    bookings_by_year_line_plot.show()

    # Number of bookings variation over time (trend) by hotel type among cancelled bookings in a line plot
    # Filter the data to only include rows where is_canceled is 1
    canceled_mask = df['is_canceled'] == 1
    canceled_df = df[canceled_mask]
    # Group the data by hotel and arrival_date_year and get the number of bookings
    bookings_by_year = canceled_df.groupby(['hotel', 'arrival_date_year'])[
        ['is_canceled']].count()
    # Reset the index
    bookings_by_year.reset_index(inplace=True)
    # Rename the columns
    bookings_by_year.rename(
        columns={'is_canceled': 'number_of_bookings'}, inplace=True)
    # Create a line plot of number of bookings by year and hotel type
    bookings_by_year_line_plot = px.line(bookings_by_year, x='arrival_date_year',
                                         y='number_of_bookings', color='hotel', title='Number of Bookings by Year and Hotel Type')
    # Add a title
    bookings_by_year_line_plot.update_layout(
        title_text='Number of Bookings by Year and Hotel Type among Cancelled Bookings', title_x=0.5)
    # Display the plot
    bookings_by_year_line_plot.show()

    # Number of bookings variation over time (seasonality) by hotel type among not cancelled bookings in a line plot
    # Filter the data to only include rows where is_canceled is 0
    not_canceled_mask = df['is_canceled'] == 0
    not_canceled_df = df[not_canceled_mask]
    # Group the data by hotel and arrival_date_month and get the number of bookings
    bookings_by_month = not_canceled_df.groupby(['hotel', 'arrival_date_month'])[
        ['is_canceled']].count()
    # Reset the index
    bookings_by_month.reset_index(inplace=True)
    # Rename the columns
    bookings_by_month.rename(
        columns={'is_canceled': 'number_of_bookings'}, inplace=True)
    # Define the order of the months
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    # Sort the data by arrival_date_month
    bookings_by_month['arrival_date_month'] = pd.Categorical(
        bookings_by_month['arrival_date_month'], categories=month_order, ordered=True)
    bookings_by_month.sort_values(by='arrival_date_month', inplace=True)
    # Create a line plot of number of bookings vs month and hotel type
    bookings_by_month_line_plot = px.line(bookings_by_month, x='arrival_date_month',
                                          y='number_of_bookings', color='hotel', title='Number of Bookings by Month and Hotel Type')
    # Add a title
    bookings_by_month_line_plot.update_layout(
        title_text='Number of Bookings by Month and Hotel Type among Not Cancelled Bookings', title_x=0.5)
    # Display the plot
    bookings_by_month_line_plot.show()

    # Number of bookings variation over time (seasonality) by hotel type among cancelled bookings in a line plot
    # Filter the data to only include rows where is_canceled is 1
    canceled_mask = df['is_canceled'] == 1
    canceled_df = df[canceled_mask]
    # Group the data by hotel and arrival_date_month and get the number of bookings
    bookings_by_month = canceled_df.groupby(['hotel', 'arrival_date_month'])[
        ['is_canceled']].count()
    # Reset the index
    bookings_by_month.reset_index(inplace=True)
    # Rename the columns
    bookings_by_month.rename(
        columns={'is_canceled': 'number_of_bookings'}, inplace=True)
    # Define the order of the months
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    # Sort the data by arrival_date_month
    bookings_by_month['arrival_date_month'] = pd.Categorical(
        bookings_by_month['arrival_date_month'], categories=month_order, ordered=True)
    bookings_by_month.sort_values(by='arrival_date_month', inplace=True)
    # Create a line plot of number of bookings vs month and hotel type
    bookings_by_month_line_plot = px.line(bookings_by_month, x='arrival_date_month',
                                          y='number_of_bookings', color='hotel', title='Number of Bookings by Month and Hotel Type')
    # Add a title
    bookings_by_month_line_plot.update_layout(
        title_text='Number of Bookings by Month and Hotel Type among Cancelled Bookings', title_x=0.5)
    # Display the plot
    bookings_by_month_line_plot.show()

    # Price variation by reserved room type among not cancelled bookings in a box plot
    # Filter the data to only include rows where is_canceled is 0
    not_canceled_mask = df['is_canceled'] == 0
    not_canceled_df = df[not_canceled_mask]
    # Create a box plot of price vs reserved room type
    price_by_room_type_box_plot = px.box(
        not_canceled_df, x='reserved_room_type', y='adr', color='hotel', title='Price by Reserved Room Type')
    # Add a title
    price_by_room_type_box_plot.update_layout(
        title_text='Price by Reserved Room Type among Not Cancelled Bookings', title_x=0.5)
    # Display the plot
    price_by_room_type_box_plot.show()

    # Price variation by reserved room type among cancelled bookings in a box plot
    # Filter the data to only include rows where is_canceled is 1
    canceled_mask = df['is_canceled'] == 1
    canceled_df = df[canceled_mask]
    # Create a box plot of price vs reserved room type
    price_by_room_type_box_plot = px.box(
        canceled_df, x='reserved_room_type', y='adr', color='hotel', title='Price by Reserved Room Type')
    # Add a title
    price_by_room_type_box_plot.update_layout(
        title_text='Price by Reserved Room Type among Cancelled Bookings', title_x=0.5)
    # Display the plot
    price_by_room_type_box_plot.show()


def feature_engineering(df):
    # Convert "reservation_status_date" to separate columns for year, month, and day
    df['reservation_status_date'] = pd.to_datetime(
        df['reservation_status_date'])
    df['reservation_status_date_year'] = df['reservation_status_date'].dt.year
    df['reservation_status_date_month'] = df['reservation_status_date'].dt.month
    df['reservation_status_date_day'] = df['reservation_status_date'].dt.day
    # Drop the original "reservation_status_date" column
    df.drop('reservation_status_date', axis=1, inplace=True)

    # Perform correlation analysis on the numerical variables
    numerical_variables = df.select_dtypes(
        include=['int64', 'float64']).columns
    # Display the correlation matrix in a heatmap with the correlation values in the cells
    correlation_matrix_heatmap = px.imshow(df[numerical_variables].corr())
    # Display the correlation values in the heatmap cells
    for i in range(len(numerical_variables)):
        for j in range(len(numerical_variables)):
            text = correlation_matrix_heatmap.data[0].z[i][j]
            correlation_matrix_heatmap.add_annotation(
                x=j, y=i, text=round(text, 2), showarrow=False)
    # Add a title
    correlation_matrix_heatmap.update_layout(
        title_text='Correlation Matrix Heatmap', title_x=0.5)
    # Display the plot
    correlation_matrix_heatmap.show()

    # Displaying the correlation values with the target variable
    # Get the correlation values with the target variable
    correlation_values = df[numerical_variables].corr(
    )['is_canceled'].sort_values(ascending=False)
    # Display the correlation values in a bar plot
    correlation_values_bar_plot = px.bar(correlation_values, x=correlation_values.values,
                                         y=correlation_values.index, orientation='h', title='Correlation Values with the Target Variable')
    # Add a title
    correlation_values_bar_plot.update_layout(
        title_text='Correlation Values with the Target Variable', title_x=0.5)
    # Display the plot
    correlation_values_bar_plot.show()

    # Store the useless variables in a list to drop them later
    useless_variables = []

    # Dropping the highly correlated variables
    # Drop the variable "reservation_status" as it is very highly correlated with the target variable
    useless_variables.append('reservation_status')
    # Drop the variable "reservation_status_date_year" as it is highly correlated with "arrival_date_year"
    useless_variables.append('reservation_status_date_year')
    # Drop the variable "reservation_status_date_month" as it is highly correlated with "arrival_date_week_number"
    useless_variables.append('reservation_status_date_month')
    # Drop the variable "reservation_status_date_day" as it is highly correlated with "arrival_date_day_of_month"
    useless_variables.append('reservation_status_date_day')

    # Dropping the variables with low correlation with the target variable
    # Get the variables with low correlation with the target variable
    low_correlation_variables = correlation_values[(
        correlation_values < 0.05) & (correlation_values > -0.05)]
    # Print the variables with low correlation with the target variable
    print('Variables with low correlation with the target variable:')
    print(low_correlation_variables)
    print('-' * 100)
    # Drop the variables with low correlation with the target variable
    useless_variables.extend(low_correlation_variables.index)

    # Dropping the useless variables
    df.drop(useless_variables, axis=1, inplace=True)

    # Converting categorical variables to numerical variables
    # Get the categorical variables
    categorical_variables = df.select_dtypes(include=['object']).columns
    # Print the unique values of each categorical variable
    for variable in categorical_variables:
        print(variable, df[variable].unique(), sep=': ')
    # Encode the categorical variables
    df = pd.get_dummies(df, columns=categorical_variables, drop_first=True)

    # Normalize the numerical variables except the target variable
    numerical_variables = df.select_dtypes(
        include=['int64', 'float64']).columns.drop('is_canceled')
    for variable in numerical_variables:
        # Get the minimum and maximum values
        minimum, maximum = df[variable].min(), df[variable].max()
        # Normalize the variable
        if minimum != maximum:
            df[variable] = (df[variable] - minimum) / (maximum - minimum)
        else:
            df[variable] = 0
    # Print the first 5 rows of the data
    print('First 5 rows of the data:')
    print(df.head())
    print('-' * 100)
    # Print the variance of each variable
    print('Variance of each variable:')
    print(df.var())
    print('-' * 100)

    # Return the preprocessed data
    return df


def grid_search(model, hyper_parameters, folds, X, y):
    # Create a grid search object
    grid_search = GridSearchCV(
        estimator=model, param_grid=hyper_parameters, scoring='accuracy', cv=folds)

    # Fit the grid search
    grid_search.fit(X, y)

    # Extract the best hyper parameters
    best_hyper_parameters = grid_search.best_params_

    # Extract the best model
    best_model = grid_search.best_estimator_

    # Return the best model and the best hyper parameters
    return best_model, best_hyper_parameters


def evaluate_model(model, X, y, resampling_strategy_name, class_ratio, folds, model_name):
    # Get the predictions
    y_pred = model.predict(X)

    # Get the confusion matrix
    _confusion_matrix = pd.crosstab(
        y, y_pred, rownames=['Actual'], colnames=['Predicted'])

    # Get the accuracy score
    _accuracy_score = accuracy_score(y, y_pred)

    # Get the precision score
    _precision_score = precision_score(y, y_pred, zero_division=1)

    # Get the recall score
    _recall_score = recall_score(y, y_pred, zero_division=1)

    # Get the f1 score
    _f1_score = f1_score(y, y_pred)

    # Get the roc auc score
    _roc_auc_score = roc_auc_score(y, y_pred)

    # Display the roc curve
    _roc_curve = roc_curve(y, y_pred)
    roc_curve_plot = px.line(x=_roc_curve[0], y=_roc_curve[1])
    # Add a title
    roc_curve_plot.update_layout(
        title_text=f'ROC Curve for {model_name} with {resampling_strategy_name} Resampling and {class_ratio} Class Ratio, {folds}-Fold Cross Validation', title_x=0.5) 
    # Add x-axis and y-axis labels
    roc_curve_plot.update_xaxes(title_text='False Positive Rate')
    roc_curve_plot.update_yaxes(title_text='True Positive Rate')
    # Display the plot
    roc_curve_plot.show()

    # Return the confusion matrix, accuracy score, precision score, recall score, f1 score, and roc auc score
    return _confusion_matrix, _accuracy_score, _precision_score, _recall_score, _f1_score, _roc_auc_score


if __name__ == "__main__":
    # Load the data
    print("Loading the data...")
    df = load_data(num_samples=20000)
    print("Completed loading the data")
    print("-" * 100)

    # Clean the data
    print("Cleaning the data...")
    df = clean_data(df)
    print("Completed cleaning the data")
    print("-" * 100)

    # Perform exploratory data analysis
    print("Performing exploratory data analysis...")
    # exploratory_data_analysis(df)
    print("Completed exploratory data analysis")
    print("-" * 100)

    # Perform feature engineering
    print("Performing feature engineering...")
    df = feature_engineering(df)
    print("Completed feature engineering")
    print("-" * 100)

    # Split the data into features and target
    X, y = df.drop('is_canceled', axis=1), df['is_canceled']

    # Resample the data for different class ratios
    # Resampling strategies
    resampling_strategies = {
        # 'No Resampling': None,
        # 'Random Oversampling': RandomOverSampler(random_state=42),
        'SMOTE': SMOTE(random_state=42),
        'ADASYN': ADASYN(random_state=42),
    }

    # Class ratios to resample the data for each resampling strategy
    class_ratios = [1.0, 0.5, 0.33, 0.25, 0.1]

    # Different number of folds for k-fold cross-validation
    num_folds_list = [3, 5, 10]

    # Defining the models
    models = {
        'Logistic Regression': {
            # 'Custom': CustomLogisticRegression(),
            'Sklearn': SciKitLearnLogisticRegression(),
        },
        'Naive Bayes': {
            # 'Custom': CustomNaiveBayes(),
            'Sklearn': SciKitLearnNaiveBayes(),
        },
        'K Nearest Neighbors': {
            # 'Custom': CustomKNearestNeighbors(),
            'Sklearn': SciKitLearnKNearestNeighbors(),
        },
        'Decision Tree': {
            # 'Custom': CustomDecisionTree(),
            'Sklearn': SciKitLearnDecisionTree(),
        },
        # 'Bagging Decision Tree (Random Forest)': {
        #     # 'Custom': CustomBaggingDecisionTree(),
        #     'Sklearn': SciKitLearnRandomForestClassifier(),
        # },
    }

    # Defining the hyperparameters
    hyperparameters = {
        'Logistic Regression': {
            'Custom': {
                'lr': [0.01, 0.05, 0.1, 0.5, 1.0],
                'n_iters': [100, 500, 1000, 5000],
            },
            'Sklearn': {
                'C': [0.01, 0.05, 0.1, 0.5, 1.0],
                'max_iter': [100, 500, 1000, 5000],
            },
        },
        'Naive Bayes': {
            'Custom': {
                'alpha': [0.1, 0.5, 1.0, 5.0, 10.0],
            },
            'Sklearn': {
                'alpha': [0.1, 0.5, 1.0, 5.0, 10.0],
            },
        },
        'K Nearest Neighbors': {
            'Custom': {
                'k': [1, 5, 10, 15, 20],
                'distance_function': ['manhattan', 'euclidean'],
                'weights': ['uniform', 'distance'],
            },
            'Sklearn': {
                'n_neighbors': [1, 5, 10, 15, 20],
                'metric': ['manhattan', 'euclidean'],
                'weights': ['uniform', 'distance'],
            },
        },
        'Decision Tree': {
            'Custom': {
                'max_depth': [1, 5, 10, 15, 20],
            },
            'Sklearn': {
                'max_depth': [1, 5, 10, 15, 20],
            },
        },
        'Bagging Decision Tree (Random Forest)': {
            'Custom': {
                'max_depth': [1, 5, 10, 15, 20],
                'n_estimators': [1, 5, 10, 15, 20],
            },
            'Sklearn': {
                'n_estimators': [1, 5, 10, 15, 20],
            },
        },
    }
    

    # Split data into train, validation, and test sets
    train_val_split_ratio = 0.8  # 80% for train/validation
    test_split_ratio = 0.2  # 20% for test

    # Store the results in a dictionary
    results = {}

    # Loop over the resampling strategies
    for resampling_strategy_name, resampling_strategy in resampling_strategies.items():
        # Store the results for each resampling strategy in a dictionary
        results[resampling_strategy_name] = {}

        # Loop over the class ratios
        for class_ratio in class_ratios:
            # Store the results for each class ratio in a dictionary
            results[resampling_strategy_name][class_ratio] = {}

            # Resample the data
            if resampling_strategy is not None:
                X_resampled, y_resampled = resampling_strategy.fit_resample(
                    X, y)
            else:
                X_resampled, y_resampled = X, y

            # Split into train and test sets
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X_resampled, y_resampled, test_size=test_split_ratio, random_state=42)

            # Looping over the folds on the training set
            for folds in num_folds_list:
                # Store the results for each fold in a dictionary
                results[resampling_strategy_name][class_ratio][folds] = {}

                # Loop over the models
                for model_name, model in models.items():
                    # Store the results for each model in a dictionary
                    results[resampling_strategy_name][class_ratio][folds][model_name] = {}

                    # Loop over the implementations
                    for implementation_name, implementation in model.items():
                        # Perform Grid Search
                        print('Performing Grid Search for {} with {} implementation on {} folds using {} resampling strategy for {} class ratio'.format(
                            model_name, implementation_name, folds, resampling_strategy_name, class_ratio))
                        best_model, best_hyperparameters = grid_search(
                            implementation,
                            hyperparameters[model_name][implementation_name],
                            folds,
                            X_train_val,
                            y_train_val
                        )

                        # Evaluate the model on the test set
                        _confusion_matrix, _accuracy_score, _precision_score, _recall_score, _f1_score, _roc_auc_score = evaluate_model(
                            best_model, X_test, y_test, resampling_strategy_name, class_ratio, folds, model_name)

                        # Store the results in a dictionary
                        results[resampling_strategy_name][class_ratio][folds][model_name][implementation_name] = {
                            'Best Hyperparameters': best_hyperparameters,
                            'Confusion Matrix': _confusion_matrix,
                            'Accuracy': _accuracy_score,
                            'Precision': _precision_score,
                            'Recall': _recall_score,
                            'F1 Score': _f1_score,
                            'ROC AUC Score': _roc_auc_score,
                        }

    # Print the results
    for resampling_strategy_name in resampling_strategies:
        for class_ratio in class_ratios:
            for folds in num_folds_list:
                for model_name, model in models.items():
                    for implementation_name, implementation in model.items():
                        print('Results for {} with {} implementation on {} folds using {} resampling strategy for {} class ratio:'.format(
                            model_name, implementation_name, folds, resampling_strategy_name, class_ratio))
                        print(results[resampling_strategy_name]
                              [class_ratio][folds][model_name][implementation_name])
                        print('-' * 100)

    # Best parameters for logistic regression
    n_iters = 500

    # Initialize a variable to store the results
    results = {}

    # Iterate over resampling techniques
    for resampling_strategy_name, resampling_strategy in resampling_strategies.items():
        # Store the results for each resampling strategy in a dictionary
        results[resampling_strategy_name] = {}
        # Iterate over class ratios
        for class_ratio in class_ratios:
            # Store the results for each class ratio in a dictionary
            results[resampling_strategy_name][class_ratio] = {}
            # Resample the data
            if resampling_strategy is not None:
                X_resampled, y_resampled = resampling_strategy.fit_resample(
                    X, y)
            else:
                X_resampled, y_resampled = X, y

            # Split into train and test sets
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X_resampled, y_resampled, test_size=test_split_ratio, random_state=42)

            # Create the model
            model = CustomLogisticRegression(n_iters=n_iters)

            # Train the model
            model.fit(X_train_val, y_train_val)

            # Evaluate the model on the test set
            _confusion_matrix, _accuracy_score, _precision_score, _recall_score, _f1_score, _roc_auc_score = evaluate_model(
                model, X_test, y_test, resampling_strategy_name, class_ratio, 1, 'Custom Logistic Regression')
            
            # Store the results in a dictionary
            results[resampling_strategy_name][class_ratio] = {
                'Confusion Matrix': _confusion_matrix,
                'Accuracy': _accuracy_score,
                'Precision': _precision_score,
                'Recall': _recall_score,
                'F1 Score': _f1_score,
                'ROC AUC Score': _roc_auc_score,
            }
    
    # Print the results
    for resampling_strategy_name in resampling_strategies:
        for class_ratio in class_ratios:
            print('Results for {} using {} class ratio:'.format(
                resampling_strategy_name, class_ratio))
            print(results[resampling_strategy_name][class_ratio])
            print('-' * 100)

    # Best parameters for Naive Bayes
    alpha = 0.1

    # Initialize a variable to store the results
    results = {}

    # Iterate over resampling techniques
    for resampling_strategy_name, resampling_strategy in resampling_strategies.items():
        # Store the results for each resampling strategy in a dictionary
        results[resampling_strategy_name] = {}
        # Iterate over class ratios
        for class_ratio in class_ratios:
            # Store the results for each class ratio in a dictionary
            results[resampling_strategy_name][class_ratio] = {}
            # Resample the data
            if resampling_strategy is not None:
                X_resampled, y_resampled = resampling_strategy.fit_resample(
                    X, y)
            else:
                X_resampled, y_resampled = X, y

            # Split into train and test sets
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X_resampled, y_resampled, test_size=test_split_ratio, random_state=42)

            # Create the model
            model = CustomNaiveBayes(alpha=alpha)

            # Train the model
            model.fit(X_train_val, y_train_val)

            # Evaluate the model on the test set
            _confusion_matrix, _accuracy_score, _precision_score, _recall_score, _f1_score, _roc_auc_score = evaluate_model(
                model, X_test, y_test, resampling_strategy_name, class_ratio, 1, 'Custom Naive Bayes')
            
            # Store the results in a dictionary
            results[resampling_strategy_name][class_ratio] = {
                'Confusion Matrix': _confusion_matrix,
                'Accuracy': _accuracy_score,
                'Precision': _precision_score,
                'Recall': _recall_score,
                'F1 Score': _f1_score,
                'ROC AUC Score': _roc_auc_score,
            }

    # Print the results
    for resampling_strategy_name in resampling_strategies:
        for class_ratio in class_ratios:
            print('Results for {} using {} class ratio:'.format(
                resampling_strategy_name, class_ratio))
            print(results[resampling_strategy_name][class_ratio])
            print('-' * 100)

    # Best parameters for K Nearest Neighbors
    k = 1
    distance_function = 'manhattan'
    weights = 'uniform'

    # Initialize a variable to store the results
    results = {}

    # Iterate over resampling techniques
    for resampling_strategy_name, resampling_strategy in resampling_strategies.items():
        # Store the results for each resampling strategy in a dictionary
        results[resampling_strategy_name] = {}
        # Iterate over class ratios
        for class_ratio in class_ratios:
            # Store the results for each class ratio in a dictionary
            results[resampling_strategy_name][class_ratio] = {}
            # Resample the data
            if resampling_strategy is not None:
                X_resampled, y_resampled = resampling_strategy.fit_resample(
                    X, y)
            else:
                X_resampled, y_resampled = X, y

            # Split into train and test sets
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X_resampled, y_resampled, test_size=test_split_ratio, random_state=42)

            # Create the model
            model = CustomKNearestNeighbors(
                k=k, distance_function=distance_function, weights=weights)

            # Train the model
            model.fit(X_train_val, y_train_val)

            # Evaluate the model on the test set
            _confusion_matrix, _accuracy_score, _precision_score, _recall_score, _f1_score, _roc_auc_score = evaluate_model(
                model, X_test, y_test, resampling_strategy_name, class_ratio, 1, 'Custom K Nearest Neighbors')
            
            # Store the results in a dictionary
            results[resampling_strategy_name][class_ratio] = {
                'Confusion Matrix': _confusion_matrix,
                'Accuracy': _accuracy_score,
                'Precision': _precision_score,
                'Recall': _recall_score,
                'F1 Score': _f1_score,
                'ROC AUC Score': _roc_auc_score,
            }

    # Print the results
    for resampling_strategy_name in resampling_strategies:
        for class_ratio in class_ratios:
            print('Results for {} using {} class ratio:'.format(
                resampling_strategy_name, class_ratio))
            print(results[resampling_strategy_name][class_ratio])
            print('-' * 100)

    # Best parameters for Decision Tree
    max_depth = 15

    # Initialize a variable to store the results
    results = {}

    # Iterate over resampling techniques
    for resampling_strategy_name, resampling_strategy in resampling_strategies.items():
        # Store the results for each resampling strategy in a dictionary
        results[resampling_strategy_name] = {}
        # Iterate over class ratios
        for class_ratio in class_ratios:
            # Store the results for each class ratio in a dictionary
            results[resampling_strategy_name][class_ratio] = {}
            # Resample the data
            if resampling_strategy is not None:
                X_resampled, y_resampled = resampling_strategy.fit_resample(
                    X, y)
            else:
                X_resampled, y_resampled = X, y

            # Split into train and test sets
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X_resampled, y_resampled, test_size=test_split_ratio, random_state=42)

            # Create the model
            model = CustomDecisionTree(max_depth=max_depth)

            # Train the model
            model.fit(X_train_val, y_train_val)

            # Evaluate the model on the test set
            _confusion_matrix, _accuracy_score, _precision_score, _recall_score, _f1_score, _roc_auc_score = evaluate_model(
                model, X_test, y_test, resampling_strategy_name, class_ratio, 1, 'Custom K Nearest Neighbors')
            
            # Store the results in a dictionary
            results[resampling_strategy_name][class_ratio] = {
                'Confusion Matrix': _confusion_matrix,
                'Accuracy': _accuracy_score,
                'Precision': _precision_score,
                'Recall': _recall_score,
                'F1 Score': _f1_score,
                'ROC AUC Score': _roc_auc_score,
            }

    # Print the results
    for resampling_strategy_name in resampling_strategies:
        for class_ratio in class_ratios:
            print('Results for {} using {} class ratio:'.format(
                resampling_strategy_name, class_ratio))
            print(results[resampling_strategy_name][class_ratio])
            print('-' * 100)