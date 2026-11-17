import pandas as pd
from pathlib import Path

# NBA file paths
NBA_TEAM_PERF = 'Datasets/raw_data/nba_team_regular_performances.csv'
SPORT_LOC = 'Datasets/raw_data/pro_venues_clean_final.csv'
NBA_LOC = 'Datasets/raw_data/nba_locations.csv'



# This function gets the nba locations (including long, lat) of their arenas from "pro_venues_cleean_final.csv"
def get_NBA_LOC():

    if not Path(SPORT_LOC).exists():
        print("The file 'Datasets/pro_venues_clean_final.csv' does not exist. Unable to get NBA locations.")
        return

    # Where the raw data came from: https://www.kaggle.com/datasets/peteyy/pro-sports-venues-nba-nfl-mlb-nhl/data
    output_filename_path = "Datasets/raw_data/nba_locations.csv"

    # gets the entire location data from the sports data "pro_venue_clean_final.csv"
    sports_locations_df = pd.read_csv(SPORT_LOC)

    # get only the NBA rows
    NBA_LOC = sports_locations_df[sports_locations_df['league'] == 'NBA']

    # drop the columns that aren't needed: 'league', 'state_province', 'arena', 'color', 
    # 'alternate_color', 'logo'
    NBA_LOC = NBA_LOC.drop(['league', 'state_province', 'arena', 'color','alternate_color', 'logo'], axis=1)

    # transform the 'team' abbreviation columns to approrpiate abbreviations for the NBA teams
    # Define your mapping for non-standard abbreviations
    abbrev_map = {
        'GS': 'GSW',
        'NO': 'NOP',
        'NY': 'NYK',
        'SA': 'SAS',
        'UTAH': 'UTA',
        'WSH': 'WAS'
    }   

    # Updates the 'abbreviation' column in-place across the full DataFrame
    NBA_LOC['abbreviation'] = NBA_LOC['abbreviation'].replace(abbrev_map)
    
    # Save the NBA Locations to a .csv file
    NBA_LOC.to_csv(output_filename_path, index=False)
    print('Succesfully extracted team locations.')


def data_setup_teamLocations():

    if not Path(NBA_TEAM_PERF).exists():
        print("The file 'Datasets/nba_team_regular_performances.csv' does not exist. Unable to get NBA locations.")
        return

    if not Path(NBA_LOC).exists():
        print("The file 'Datasets/nba_locations.csv' does not exist. Unable to get NBA locations.")
        return

    # Create dataframes from the csv files for performance and locations
    df_team_performance = pd.read_csv(NBA_TEAM_PERF, parse_dates=['GAME_DATE'])
    df_NBA_LOC = pd.read_csv(NBA_LOC)    

    # seperate the home and away games 
    home_games = df_team_performance[df_team_performance['MATCHUP'].str.contains('vs')]
    away_games = df_team_performance[df_team_performance['MATCHUP'].str.contains('@')]

    # add columns for home and opponent team
    home_games['home_team'] = home_games['TEAM_ABBREVIATION']
    home_games['opponent_team'] = home_games['MATCHUP'].str.split('vs.').str[1].str.strip()
    away_games['home_team'] = away_games['MATCHUP'].str.split('@').str[1].str.strip()
    away_games['opponent_team'] = away_games['MATCHUP'].str.split('@').str[1].str.strip()

    # Add the lon and lat to the home games
    home_games['lon'] = home_games['TEAM_NAME'].map(df_NBA_LOC.set_index('team')['long'])
    home_games['lat'] = home_games['TEAM_NAME'].map(df_NBA_LOC.set_index('team')['lat'])

    away_games['lon'] = away_games['home_team'].map(df_NBA_LOC.set_index('abbreviation')['long'])
    away_games['lat'] = away_games['home_team'].map(df_NBA_LOC.set_index('abbreviation')['lat'])

    # merge the dataframes back together
    df_team_performance = pd.concat([home_games, away_games], ignore_index=True)

    # sort the dataframe by date
    df_team_performance = df_team_performance.sort_values(by='GAME_DATE')

    # map the opponent_team id to each row
    team_id_map = (
        df_team_performance.set_index('TEAM_ABBREVIATION')['TEAM_ID'].to_dict()
    )
    df_team_performance['opponent_team_id'] = df_team_performance['opponent_team'].map(team_id_map)

    # check if there is any missing value (there shouldn't be)
    if df_team_performance.isna().any().any():
        print("There are missing values in the dataframe after adding locations data.")
        return

    print("Successfully added the location data to the team performance data.")

    return df_team_performance


def clean_data():

    # check if file exists, and gets the location data
    get_NBA_LOC()

    # adds the location data to the team performance data
    df_team_loc_perf = data_setup_teamLocations()

    # drop any unnessary columns from the dataframe
    df_team_loc_perf = df_team_loc_perf.drop([
                                        # Metadata
                                        "TEAM_NAME",
                                        "MATCHUP",
                                        "AVAILABLE_FLAG",
                                        # Performance metrics that are not the target
                                        "WL",
                                        "MIN",
                                        "POSS",
                                        "PACE",
                                        "PACE_PER40",
                                        "E_PACE",
                                        "AST_PCT",
                                        "AST_TO",
                                        "AST_RATIO",
                                        "OREB_PCT",
                                        "DREB_PCT",
                                        "REB_PCT",
                                        "TM_TOV_PCT",
                                        "EFG_PCT",
                                        "TS_PCT",
                                        "PIE",
                                        # Estimated metrics
                                        "E_OFF_RATING",
                                        "E_DEF_RATING",
                                        "E_NET_RATING",
                                        "NET_RATING",
                                        # Rank columns
                                        "GP_RANK",
                                        "W_RANK",
                                        "L_RANK",
                                        "W_PCT_RANK",
                                        "MIN_RANK",
                                        "OFF_RATING_RANK",
                                        "DEF_RATING_RANK",
                                        "NET_RATING_RANK",
                                        "AST_PCT_RANK",
                                        "AST_TO_RANK",
                                        "AST_RATIO_RANK",
                                        "OREB_PCT_RANK",
                                        "DREB_PCT_RANK",
                                        "REB_PCT_RANK",
                                        "TM_TOV_PCT_RANK",
                                        "EFG_PCT_RANK",
                                        "TS_PCT_RANK",
                                        "PACE_RANK",
                                        "PIE_RANK"],
                                        axis=1)

    # save the data to the cleaned data folder
    df_team_loc_perf.to_csv('Datasets/cleaned_data/nba_team_performance_cleaned.csv', index=False)

                                        
if __name__ == "__main__":
    clean_data()
    # get_NBA_LOC()
