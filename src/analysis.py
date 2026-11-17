import sys
from pathlib import Path
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

from extract import get_raw_data_NBA
from transform import clean_data
from features import feature_engineering

# NBA file paths 
NBA_PERF_FULL_FEATS = 'Datasets/features_data/nba_team_perf_full_feats.csv'


def setup():
    # This fetches the raw data for the NBA team performances and saves it to raw data folder
    get_raw_data_NBA()
    print("ANALYSIS SETUP: Successfully extracted team performances data.")

    # This cleans the data NBA team performances and saves it to cleaned data folder
    clean_data()
    print("ANALYSIS SETUP: Successfully cleaned team performances data.")

    # This handles the feature engineering and saves dataframes to features data folder
    feature_engineering()
    print("ANALYSIS SETUP: Successfully feature engineered the team performance data.")


# This analysis is meant for 
def analysis1():

    output_dir = Path('Analysis/analysis1')

    if not Path(NBA_PERF_FULL_FEATS):
        print("The file 'Datasets/features_data/nba_team_perf_full_feats.csv' does not exist. Unable to get distance feature.")
        return

    nba_team_perf_with_dist = pd.read_csv(NBA_PERF_FULL_FEATS)

    # plot what the distance vs OFF_Rating looks like
    data_plot = nba_team_perf_with_dist.plot.scatter(x='distance', y='OFF_RATING')
    plt.savefig(output_dir / 'Original_distance_vs_off_rating.png')

    ## OLS Regression: distance vs. OFF_RATING ##

    # Prepare dataframe for the ols regression
    data_to_fit = pd.DataFrame({
        'y': nba_team_perf_with_dist['OFF_RATING'],
        'x': nba_team_perf_with_dist['distance'],
        'one': 1
    })

    # Fit OLS
    a1_model = sm.OLS(data_to_fit['y'], data_to_fit[['x', 'one']]).fit()

    # get the prediction values from the model 
    y_pred = a1_model.predict(data_to_fit[['x', 'one']])

    # plot the data with the scatter plot
    data_plot.plot(data_to_fit['x'], y_pred, color='red', linewidth=2, label='OLS Fit Line')
    plt.title('Test 1: Distance vs OFF_RATING (OLS Fit)')
    plt.xlabel('Distance')
    plt.ylabel('Offensive Rating')
    plt.legend()
    plt.savefig(output_dir / 'regression_distance_vs_off_rating.png')


    # Save the OLS summary to a file
    with open(output_dir / 'a1_ols_summary.txt', 'w') as f:
        f.write(a1_model.summary().as_text())


################################################################################################
################################################################################################


def analysis2_without_dist():

    output_dir = Path('Analysis/analysis2')

    if not Path(NBA_PERF_FULL_FEATS):
        print("The file 'Datasets/features_data/nba_team_perf_full_feats.csv' does not exist. Unable to get distance feature.")
        return

    df_perf_full_feats = pd.read_csv(NBA_PERF_FULL_FEATS, parse_dates=['GAME_DATE'])

    # This helps select the feature columns
    feature_cols = [
        'is_home',
        'rest_days',
        'team_last10_off_rating',
        'opp_last10_def_rating',
    ]

    # get the features and the target
    X = df_perf_full_feats[feature_cols]
    y = df_perf_full_feats['OFF_RATING']

    # scale the continous features
    cols_to_scale = [
        'team_last10_off_rating',
        'opp_last10_def_rating',
    ]
    scaler = StandardScaler()
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

    # Add intercept column (creates the vector of 1s)
    X = sm.add_constant(X) 

    # Fit OLS model
    a2_model = sm.OLS(y, X).fit()

    # set up the parameters for the plot of coefficients
    coefs = a2_model.params.drop('const')
    pvalues = a2_model.pvalues.drop('const')

    # Plot horizontal bar chart
    plt.figure(figsize=(8, 4))
    colors = ['gray' if p > 0.05 else 'blue' for p in pvalues]
    coefs.plot(kind='bar', color=colors)

    plt.title('Test 2: OLS Feature Coefficients Without Dist (Blue = Significant p < 0.05)')
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'ols_multi_regress_without_dist_coefficients.png')
    plt.close()

    # Save the OLS summary to a file
    with open(output_dir / 'a2_ols_without_dist_summary.txt', 'w') as f:
        f.write(a2_model.summary().as_text())


def analysis2_with_dist():

    output_dir = Path('Analysis/analysis2')

    if not Path(NBA_PERF_FULL_FEATS):
        print("The file 'Datasets/features_data/nba_team_perf_full_feats.csv' does not exist. Unable to get distance feature.")
        return

    df_perf_full_feats = pd.read_csv(NBA_PERF_FULL_FEATS, parse_dates=['GAME_DATE'])

    # This helps select the feature columns
    feature_cols = [
        'distance',
        'is_home',
        'rest_days',
        'team_last10_off_rating',
        'opp_last10_def_rating',
    ]

    # get the features and the target
    X = df_perf_full_feats[feature_cols]
    y = df_perf_full_feats['OFF_RATING']

    # scale the continous features
    cols_to_scale = [
        'distance',
        'team_last10_off_rating',
        'opp_last10_def_rating',
    ]
    scaler = StandardScaler()
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

    # Add intercept column (creates the vector of 1s)
    X = sm.add_constant(X) 

    # Fit OLS model
    a2_model = sm.OLS(y, X).fit()

    # set up the parameters for the plot of coefficients
    coefs = a2_model.params.drop('const')
    pvalues = a2_model.pvalues.drop('const')

    # Plot horizontal bar chart
    plt.figure(figsize=(8, 4))
    colors = ['gray' if p > 0.05 else 'blue' for p in pvalues]
    coefs.plot(kind='bar', color=colors)

    plt.title('Test 3: OLS Feature Coefficients With Dist (Blue = Significant p < 0.05)')
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'ols_multi_regress_with_dist_coefficients.png')
    plt.close()

    # Save the OLS summary to a file
    with open(output_dir / 'a2_ols_with_dist_summary.txt', 'w') as f:
        f.write(a2_model.summary().as_text())


################################################################################################
################################################################################################


def analysis3_without_dist():

    output_dir = Path('Analysis/analysis3')

    if not Path(NBA_PERF_FULL_FEATS):
        print("The file 'Datasets/features_data/nba_team_perf_full_feats.csv' does not exist. Unable to get distance feature.")
        return

    df_perf_full_feats = pd.read_csv(NBA_PERF_FULL_FEATS, parse_dates=['GAME_DATE'])

    # split the season years into train and valid sets (current split is 80-20)
    train_years = ['2021-22', '2022-23', '2023-24', '2024-25']
    valid_years = ['2025-26']

    # Split the data into train and valid sets (using season years to ensure no data leakage)
    train_df = df_perf_full_feats[df_perf_full_feats['SEASON_YEAR'].isin(train_years)]
    valid_df = df_perf_full_feats[df_perf_full_feats['SEASON_YEAR'].isin(valid_years)]

    # This helps select the feature columns
    feature_cols = [
        'is_home',
        'rest_days',
        'team_last10_off_rating',
        'opp_last10_def_rating',
    ]

    # get the train features and target
    X_train = train_df[feature_cols]
    y_train = train_df['OFF_RATING']

    # get the valid features and target
    X_valid = valid_df[feature_cols]
    y_valid = valid_df['OFF_RATING']

    # Fit Random Forest Regressor to the train dataset
    # rf_model = RandomForestRegressor(n_estimators=100)
    rf_model = RandomForestRegressor(n_estimators=200, 
                                     max_depth=6,           # we want to limit the depth to make broader boundaries
                                     min_samples_leaf=40,   # we want at least 40 games in a leaf to make a decision
                                     min_samples_split=80,  # we want at least 80 games before making another decision
                                    )
    rf_model.fit(X_train, y_train)
    
    # write the training score and the model score to .txt file
    train_r2 = rf_model.score(X_train, y_train)
    valid_r2 = rf_model.score(X_valid, y_valid)

    score_summary = f"""
    ================================================
        RANDOM FOREST MODEL WITHOUT DIST RESULTS        
    ================================================
    Training R^2 Score:    {train_r2:.4f}
    Validation R^2 Score:  {valid_r2:.4f}
    ================================================
    """

    with open(output_dir / 'a3_rf_regress_without_dist_scores.txt', "w") as f:
        f.write(score_summary.strip())

    # Extract feature importances and plot it
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True)

    # Plot Graph (X = Features, Y = Importance Score)
    plt.figure(figsize=(8, 5))
    colors = ['blue']
    importances.plot(kind='bar', color=colors)

    plt.title(
        'Test 4: Random Forest Without Dist Feature Importances',
        fontsize=12,
        weight='bold',
    )
    plt.xlabel('Features')
    plt.ylabel('Relative Importance Score')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'randfor_regress_without_dist_importance_scores.png')


def analysis3_with_dist():

    output_dir = Path('Analysis/analysis3')

    if not Path(NBA_PERF_FULL_FEATS):
        print("The file 'Datasets/features_data/nba_team_perf_full_feats.csv' does not exist. Unable to get distance feature.")
        return

    df_perf_full_feats = pd.read_csv(NBA_PERF_FULL_FEATS, parse_dates=['GAME_DATE'])

    # split the season years into train and valid sets (current split is 80-20)
    train_years = ['2021-22', '2022-23', '2023-24', '2024-25']
    valid_years = ['2025-26']

    # Split the data into train and valid sets (using season years to ensure no data leakage)
    train_df = df_perf_full_feats[df_perf_full_feats['SEASON_YEAR'].isin(train_years)]
    valid_df = df_perf_full_feats[df_perf_full_feats['SEASON_YEAR'].isin(valid_years)]

    # This helps select the feature columns
    feature_cols = [
        'distance',
        'is_home',
        'rest_days',
        'team_last10_off_rating',
        'opp_last10_def_rating',
    ]

    # get the train features and target
    X_train = train_df[feature_cols]
    y_train = train_df['OFF_RATING']

    # get the valid features and target
    X_valid = valid_df[feature_cols]
    y_valid = valid_df['OFF_RATING']

    # Fit Random Forest Regressor to the train dataset
    # rf_model = RandomForestRegressor(n_estimators=100)
    rf_model = RandomForestRegressor(n_estimators=200, 
                                     max_depth=6,           # we want to limit the depth to make broader boundaries
                                     min_samples_leaf=40,   # we want at least 40 games in a leaf to make a decision
                                     min_samples_split=80,  # we want at least 80 games before making another decision
                                    )
    rf_model.fit(X_train, y_train)
    
    # write the training score and the model score to .txt file
    train_r2 = rf_model.score(X_train, y_train)
    valid_r2 = rf_model.score(X_valid, y_valid)

    score_summary = f"""
    =============================================
        RANDOM FOREST MODEL WITH DIST RESULTS        
    =============================================
    Training R^2 Score:    {train_r2:.4f}
    Validation R^2 Score:  {valid_r2:.4f}
    =============================================
    """

    with open(output_dir / 'a3_rf_regress_with_dist_scores.txt', "w") as f:
        f.write(score_summary.strip())

    # Extract feature importances and plot it
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True)

    # Plot Graph (X = Features, Y = Importance Score)
    plt.figure(figsize=(8, 5))
    colors = ['blue']
    importances.plot(kind='bar', color=colors)

    plt.title(
        'Test 5: Random Forest With Dist Feature Importances',
        fontsize=12,
        weight='bold',
    )
    plt.xlabel('Features')
    plt.ylabel('Relative Importance Score')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'randfor_regress_with_dist_importance_scores.png')



def main():

    # run this setup functions to handle the data fetching, cleaning and feature engineering
    # NOTE: this is only required if you need to get the data from the NBA endpoints. 
    # setup()

    # run the first analysis: naively test for distance vs. off_rating
    analysis1()
    print("ANALYSIS: Completed analyis 1.")

    # run the second analysis: regression test with more features
    analysis2_without_dist()
    analysis2_with_dist()
    print("ANALYSIS: Completed analyis 2.")

    # run the third analysis: non linear regression test (random forest)
    analysis3_without_dist()
    analysis3_with_dist()
    print("ANALYSIS: Completed analyis 3.")

    

if __name__=='__main__':
    main()