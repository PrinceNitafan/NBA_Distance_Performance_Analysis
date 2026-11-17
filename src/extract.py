import os
from pathlib import Path
import time
import pandas as pd
from nba_api.stats.endpoints import teamgamelogs, leaguegamelog, leaguedashteamstats



# This function sends the log request to get data
def fetch_nba_data(log_request):
    try:
        # Call the endpoint and create a dataframe
        season_df = log_request.get_data_frames()[0]
        print(f"Successfully retrieved {len(season_df)}.")

        return season_df
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return


# gets the stats data for all nba games in seasons
def fetch_nba_team_games(output_filepath, seasons, season_type, player_or_team):
    ######### team performances ######### 
    all_seasons_team_perf = []

    print("Fetching NBA game data...")
    for season in seasons:
        log_request = teamgamelogs.TeamGameLogs(
            season_nullable=season,
            league_id_nullable='00', # get only nba league data
            season_type_nullable=season_type,
            measure_type_player_game_logs_nullable="Advanced"
        )

        curr_season = fetch_nba_data(log_request)

        if curr_season is not None:

            all_seasons_team_perf.append(curr_season)
            print(f"Successfully retrieved NBA games for season: {season}.")
        else:
            print(f"Failed to retrieve NBA games for season: {season}.")
            break

    # If we get all the data we wanted, concatenate and produce a csv file
    if len(all_seasons_team_perf) == len(seasons):
        NBA_seasons_game_data = pd.concat(all_seasons_team_perf, ignore_index=True)
        
        # Save as a CSV file 
        NBA_seasons_game_data.to_csv(output_filepath, index=False)
        print("Successfully extracted NBA games data.")
    else:
        print("Failed to extract NBA games data.")


# This function gets the data for the NBA team performances and stats, and saves to to csv files
def get_raw_data_NBA():

    ## Define the variables for what data to get ##
    # This defines what nba seasons to fetch
    seasons = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']

    # This defines what type of season to get: 'Regular Season', 'Playoffs'
    season_type = 'Regular Season'

    ######### team performances #########
    # define the file output name
    output_filename_path = "Datasets/raw_data/nba_team_regular_performances.csv"

    
    # This defines whether to get team or player stats: 'P', 'T'
    player_or_team = 'T'

    # get and write data to a csv file
    fetch_nba_team_games(output_filename_path, seasons, season_type, player_or_team)




if __name__ == "__main__":
    get_raw_data_NBA()
