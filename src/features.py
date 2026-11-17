import pandas as pd
from pathlib import Path
from utils import distance

CLEANED_NBA_TEAM_PERF = 'Datasets/cleaned_data/nba_team_performance_cleaned.csv'
NBA_TEAM_PERF_WITH_DIST = 'Datasets/features_data/NBA_TEAM_PERF_WITH_DISTance.csv'


# This function handles the feature engineering for the distance between two points (lat1, lon1) and lat2, lon2)
def distance_feature(df):

    # Sort the dataframe by team and date 
    df = df.sort_values(by=['SEASON_YEAR', 'TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

    # get the location of each team's previous game
    df['prev_lon'] = df.groupby(['SEASON_YEAR', 'TEAM_ID'])['lon'].shift(1)
    df['prev_lat'] = df.groupby(['SEASON_YEAR', 'TEAM_ID'])['lat'].shift(1)

    # Calculate the distance between the current game and the previous game
    df['distance'] = distance(
        df['prev_lat'],
        df['prev_lon'],
        df['lat'],
        df['lon']
    )

    print("Successfully added the distance feature to the team performance data.")

    return df


######################## Functions to add more features ########################

def court_advantage_feature(df):
    # Calculate the absolute number of time zones crossed based on longitude
    df['is_home'] = (df['TEAM_ABBREVIATION'] == df['home_team']).astype(int)

    return df

def rest_days_feature(df):
    # Sort the the values chronologically
    df = df.sort_values(by=['TEAM_ABBREVIATION', 'GAME_DATE']).reset_index(drop=True)

    # Calculate the date difference between consecutive games for each team
    df['rest_days'] = df.groupby('TEAM_ABBREVIATION')['GAME_DATE'].diff().dt.days
    df['rest_days'] = df['rest_days'] - 1 # This is show the actual amount of rest days

    return df

def team_off_rating_feature(df):
    # Sort the data chronologically
    df = df.sort_values(by=['SEASON_YEAR', 'TEAM_ID', 'GAME_DATE']).reset_index(
        drop=True
    )

    # Calculate each team's 10-game rolling offensive rating
    df['team_last10_off_rating'] = (
        df.groupby(['SEASON_YEAR', 'TEAM_ID'])['OFF_RATING']
        .transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
    )

    return df

def opp_def_rating_feature(df):

    # Sort the data chronologically
    df = df.sort_values(by=['SEASON_YEAR', 'TEAM_ID', 'GAME_DATE']).reset_index(
        drop=True
    )

    # Calculate each team's 10-game rolling defensive rating
    df['team_last10_def_rating'] = (
        df.groupby(['SEASON_YEAR', 'TEAM_ID'])['DEF_RATING']
        .transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
    )

    # Create a lookup table containing pre-game defensive ratings
    def_lookup = df[['GAME_ID', 'TEAM_ID', 'team_last10_def_rating']].copy()

    # Rename columns to represent the opponent team id
    def_lookup = def_lookup.rename(
        columns={
            'TEAM_ID': 'opponent_team_id',
            'team_last10_def_rating': 'opp_last10_def_rating',
        }
    )

    # Join the data back to your main dataframe where opponent_team matches
    df = df.merge(def_lookup, on=['GAME_ID', 'opponent_team_id'], how='left')

    return df

########################################################################

# Function that structurizes the functions that adds the other features
def add_features():
    if not Path(CLEANED_NBA_TEAM_PERF).exists():
        print("The file 'Datasets/cleaned_data/nba_team_performance_cleaned.csv' does not exist. Unable to get distance feature.")
        return

    # Create a dataframe of the data with distance
    df = pd.read_csv(CLEANED_NBA_TEAM_PERF, parse_dates=['GAME_DATE'])

    # 1. add the distance feature
    df_dist = distance_feature(df)

    # 2. add the court advantage feature
    df_dist_court = court_advantage_feature(df_dist)

    # 3. add the rest days feature
    df_dist_court_rest = rest_days_feature(df_dist_court)

    # 4. add the teams rolling off_ratings
    df_dist_court_rest_offrt = team_off_rating_feature(df_dist_court_rest)

    # 5. add the opponent team's off_ratings
    df_dist_court_rest_offrt_defrt= opp_def_rating_feature(df_dist_court_rest_offrt)

    # Drop Game 1 rows where the these values are Null
    df_final = df_dist_court_rest_offrt_defrt.dropna(
            subset=['rest_days', 'team_last10_def_rating', 'opp_last10_def_rating']
            ).reset_index(drop=True)


    df_final.to_csv('Datasets/features_data/nba_team_perf_full_feats.csv', index=False)



def feature_engineering():

    # Adds the rest day feature
    add_features()



if __name__ == "__main__":
    feature_engineering()
