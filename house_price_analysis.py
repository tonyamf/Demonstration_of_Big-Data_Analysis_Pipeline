# -*- coding: utf-8 -*-
"""
House Price Prediction - Analysis and Modeling

This script performs data analysis and builds a linear regression model
to predict house prices in India based on the Kaggle dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels import api as sm

def load_and_prepare_data(filepath):
    """
    Loads data from a CSV file and performs initial type conversions.
    
    Args:
        filepath (str): The path to the train.csv file.
        
    Returns:
        pandas.DataFrame: The prepared DataFrame.
    """
    try:
        prices_df = pd.read_csv(filepath)
        
        # Correct column names if they have trailing spaces or tabs
        prices_df.columns = prices_df.columns.str.strip()

        # Convert boolean-like columns to boolean type
        bool_cols = ['UNDER_CONSTRUCTION', 'RERA', 'READY_TO_MOVE', 'RESALE']
        for col in bool_cols:
            prices_df[col] = prices_df[col].astype(bool)

        # Convert numeric columns
        prices_df['BHK_NO.'] = prices_df['BHK_NO.'].astype(int)
        prices_df['SQUARE_FT'] = prices_df['SQUARE_FT'].astype(float)
        prices_df['LONGITUDE'] = prices_df['LONGITUDE'].astype(float)
        prices_df['LATITUDE'] = prices_df['LATITUDE'].astype(float)
        
        # Convert target variable and handle potential formatting issues
        prices_df['TARGET(PRICE_IN_LACS)'] = prices_df['TARGET(PRICE_IN_LACS)'].astype(float)
        
        # --- Data Cleaning and Feature Engineering from the Paper ---
        # 1. Convert Square Feet to Square Meters
        prices_df['SQUARE_M'] = prices_df['SQUARE_FT'] / 10.764
        
        # 2. Convert Price from INR Lacs to GBP
        # Assuming an exchange rate of ~95 INR to 1 GBP (as used in the paper)
        # 1 Lac = 100,000
        prices_df['TARGET(PRICE_IN_BGP)'] = (prices_df['TARGET(PRICE_IN_LACS)'] * 100000) / 94.94

        # 3. Drop original and irrelevant columns
        # BHK_OR_RK is constant, and original columns are now redundant.
        prices_df = prices_df.drop(columns=['SQUARE_FT', 'BHK_OR_RK', 'TARGET(PRICE_IN_LACS)'])
        
        return prices_df
        
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None


def exploratory_data_analysis(df):
    """
    Performs and displays exploratory data analysis plots.
    
    Args:
        df (pandas.DataFrame): The dataframe to analyze.
    """
    print("--- Exploratory Data Analysis ---")
    
    # Correlation Matrix
    print("\nCorrelation Matrix:")
    corr = df.corr(numeric_only=True)
    print(corr['TARGET(PRICE_IN_BGP)'].sort_values(ascending=False))
    
    # Plotting Correlation Matrix
    plt.figure(figsize=(12, 10))
    sm.graphics.plot_corr(corr, xnames=list(corr.columns))
    plt.title('Feature Correlation Matrix')
    plt.show()

    # Scatter plot of the most correlated feature vs. price
    plt.figure(figsize=(10, 6))
    plt.scatter(df['SQUARE_M'], df['TARGET(PRICE_IN_BGP)'], alpha=0.5)
    plt.title('House Price vs. Area (Square Meters)')
    plt.xlabel('Area in Square Meters')
    plt.ylabel('Price in GBP')
    plt.grid(True)
    plt.show()
    
    # Pie chart for 'POSTED_BY'
    plt.figure(figsize=(8, 8))
    df['POSTED_BY'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['green', 'purple', 'orange'])
    plt.title('Distribution of Listings by Seller Type')
    plt.ylabel('') # Hide the y-label
    plt.show()


def build_and_evaluate_model(X, y, model_name):
    """
    Builds and evaluates a linear regression model.
    
    Args:
        X (pandas.DataFrame): Feature data.
        y (pandas.Series): Target data.
        model_name (str): A name for the model being evaluated.
    """
    print(f"\n--- Evaluating Model: {model_name} ---")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Coefficients: {regressor.coef_}")
    print(f"Intercept: {regressor.intercept_:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Coefficient of Determination (R^2): {r2:.2f}")
    
def main():
    """
    Main function to run the data analysis and modeling pipeline.
    """
    # Resolve train.csv next to this script so it runs from any directory
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.csv')
    prices_dataset = load_and_prepare_data(filepath)
    
    if prices_dataset is None:
        return

    # --- Exploratory Analysis ---
    exploratory_data_analysis(prices_dataset)

    # --- Predictive Modeling ---
    
    # First, remove outliers based on price as described in the paper
    Q1 = prices_dataset['TARGET(PRICE_IN_BGP)'].quantile(0.25)
    Q3 = prices_dataset['TARGET(PRICE_IN_BGP)'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_no_outliers = prices_dataset[(prices_dataset['TARGET(PRICE_IN_BGP)'] > lower_bound) & (prices_dataset['TARGET(PRICE_IN_BGP)'] < upper_bound)]
    print(f"\nOriginal dataset size: {len(prices_dataset)}")
    print(f"Dataset size after removing price outliers: {len(df_no_outliers)}")

    # Define features and target
    features = ['UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'SQUARE_M', 'READY_TO_MOVE', 'RESALE', 'LONGITUDE', 'LATITUDE']
    target = 'TARGET(PRICE_IN_BGP)'

    # Model 1: For 'Builder' listings
    df_builder = df_no_outliers[df_no_outliers['POSTED_BY'] == 'Builder']
    X_builder = df_builder[features]
    y_builder = df_builder[target]
    build_and_evaluate_model(X_builder, y_builder, "Builder Listings Only")

    # Model 2: For 'Dealer' listings
    df_dealer = df_no_outliers[df_no_outliers['POSTED_BY'] == 'Dealer']
    X_dealer = df_dealer[features]
    y_dealer = df_dealer[target]
    build_and_evaluate_model(X_dealer, y_dealer, "Dealer Listings Only")

    # Model 3: For 'Owner' listings
    df_owner = df_no_outliers[df_no_outliers['POSTED_BY'] == 'Owner']
    X_owner = df_owner[features]
    y_owner = df_owner[target]
    build_and_evaluate_model(X_owner, y_owner, "Owner Listings Only")

    # Model 4: Using all data with 'POSTED_BY' as a categorical feature
    df_full = df_no_outliers.copy()
    df_full['POSTED_BY'] = df_full['POSTED_BY'].astype('category').cat.codes # Convert to numerical codes
    
    X_full = df_full[['POSTED_BY'] + features]
    y_full = df_full[target]
    build_and_evaluate_model(X_full, y_full, "All Listings (with POSTED_BY as a feature)")

if __name__ == '__main__':
    main()
