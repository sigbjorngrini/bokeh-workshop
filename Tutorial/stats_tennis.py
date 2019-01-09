"""
Script to change the dataset of tennis matches from https://github.com/JeffSackmann/tennis_wta

One function currently made:
:player_stats: Makes a table of all the players that played (or rather lost) out of the matches.

Author: Sigbjørn Grini
"""

import pandas as pd
import numpy as np

def player_stats(matches):
    """
    Returns a pandas dataframe with player info for all players who lost a match
    in the pandas dataframe :matches:
    
    :matches: (pandas dataframe) matches which contains the same info as
        wta_matches_2018.csv on https://github.com/JeffSackmann/tennis_wta
        
    This function is made as a part of the bokeh workshop and is not properly unit tested.
    """
    
    players = list(np.unique(matches['loser_name']))
    player_stats = pd.DataFrame(players)
    player_stats.columns = ['Name']
    losses = []
    wins = []
    rank = []
    age = []
    minutes_played = []
    first_sv_pct = []
    bp_save_pct = []

    matches_won = matches.groupby('winner_name')
    matches_lost = matches.groupby('loser_name')

    for index, row in player_stats.iterrows():



        # Wins
        if row['Name'] in matches_won.groups.keys():
            wins.append(matches_won.count().loc[row['Name'], :].iloc[0])
        else:
            wins.append(0)

        # Losses
        losses.append(matches_lost.count().loc[row['Name'], :].iloc[0])
        
        # Minutes played
        
        minutes = []
        if row['Name'] in matches_won.groups.keys():
            minutes.append(matches_won.sum().loc[row['Name'], :].loc['minutes'])
        minutes.append(matches_lost.sum().loc[row['Name'], :].loc['minutes'])
        minutes_played.append(sum(minutes))

        # Age
        player_age_avg = []
        if row['Name'] in matches_won.groups.keys():
            player_age_avg.append(matches_won.mean().loc[row['Name'], :].loc['winner_age'])
        player_age_avg.append(matches_lost.mean().loc[row['Name'], :].loc['loser_age'])
        age.append(np.mean(player_age_avg))

        # Rank
        player_rank_avg = []
        if row['Name'] in matches_won.groups.keys():
            player_rank_avg.append(matches_won.mean().loc[row['Name'], :].loc['winner_rank'])
        player_rank_avg.append(matches_lost.mean().loc[row['Name'], :].loc['loser_rank'])
        rank.append(np.mean(player_rank_avg))
        

        # Serve stats
        serve_points = []
        first_serve = []
        if row['Name'] in matches_won.groups.keys():
            serve_points.append(matches_won.sum().loc[row['Name'], :].loc['w_svpt'])
            first_serve.append(matches_won.sum().loc[row['Name'], :].loc['w_1stIn'])
        serve_points.append(matches_lost.sum().loc[row['Name'], :].loc['l_svpt'])
        first_serve.append(matches_lost.sum().loc[row['Name'], :].loc['l_1stIn'])
        first_sv_pct.append(sum(first_serve) / sum(serve_points))

        # Break points saved
        bp_faced = []
        bp_saved = []
        if row['Name'] in matches_won.groups.keys():
            bp_faced.append(matches_won.sum().loc[row['Name'], :].loc['w_bpFaced'])
            bp_saved.append(matches_won.sum().loc[row['Name'], :].loc['w_bpSaved'])
        bp_faced.append(matches_lost.sum().loc[row['Name'], :].loc['l_bpFaced'])
        bp_saved.append(matches_lost.sum().loc[row['Name'], :].loc['l_bpSaved'])
        bp_save_pct.append(sum(bp_saved) / sum(bp_faced))

    player_stats['Wins'] = wins
    player_stats['Losses'] = losses
    player_stats['Win/loss ratio'] = player_stats['Wins'] / player_stats['Losses']
    player_stats['Matches played'] = player_stats['Wins'] + player_stats['Losses']
    player_stats['Minutes played'] = minutes_played
    player_stats['Average match time'] = player_stats['Minutes played'] / player_stats['Matches played']
    player_stats['Age'] = age
    player_stats['Rank'] = rank
    player_stats['First serve %'] = first_sv_pct
    player_stats['Break points save %'] = bp_save_pct
    
    return player_stats