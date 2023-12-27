#Individual variables in pybaseball.statcast may change year by year

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pickle
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pybaseball
from pybaseball import schedule_and_record
from pybaseball import statcast
from pybaseball import playerid_lookup
from pybaseball import statcast_pitcher
from pybaseball import team_batting
from pybaseball import team_pitching
from pybaseball import batting_stats_range
from pybaseball import pitching_stats_range
from pybaseball import statcast_single_game
from pybaseball import playerid_reverse_lookup

all_play_data1=statcast('2022-05-20','2022-09-29')#all pitches from 2022
all_play_data2=statcast('2023-3-30','2023-10-01')
all_play_data=pd.concat([all_play_data1,all_play_data2])

fi_2022 = all_play_data[all_play_data['inning'] == 1] #subset only first inning

fi_ab_2022 = fi_2022[fi_2022['events'].notnull()] #subset only last pitch of AB
fi_ab_2022['barrel'] = fi_ab_2022.apply(lambda row: np.nan if pd.isna(row['launch_speed']) or pd.isna(row['launch_angle']) else (1 if row['launch_speed'] > 98.0 and 25.9 <= row['launch_angle'] <= 30.1 else 0), axis=1)


fi_ab_events_2022 = pd.concat([fi_ab_2022, pd.get_dummies(fi_ab_2022['events'])], axis=1) #create dummy variable for all possible events

fi_ab_events_2022["fielding_team"] = np.where(fi_ab_events_2022["inning_topbot"] == "Bot", 
                                              fi_ab_events_2022["away_team"], fi_ab_events_2022["home_team"]) #columns for home and away teams

grouped = fi_ab_events_2022.groupby(['game_pk','inning_topbot']) #one row for each half inning
fi_grouped_2022 = grouped.agg({'game_date': 'first',
                         'player_name': 'first',
                         'pitcher': 'first',
                         'home_team': 'first',
                         'away_team': 'first',
                         'p_throws': 'first',
                         'fielding_team': 'first',
                         'launch_speed': 'mean',
                         'estimated_woba_using_speedangle': 'mean',
                         'estimated_ba_using_speedangle': 'mean',
                         'home_score':'max',
                         'away_score':'max',
                         'delta_run_exp':'sum',
                         'strikeout':'sum',
                         'grounded_into_double_play':'sum',
                         'single':'sum',
                         'double':'sum',
                         'triple':'sum',
                         'home_run':'sum',
                         'walk':'sum',
                         'field_out':'sum',
                         'fielders_choice_out':'sum',
                         'hit_by_pitch':'sum',
                         'sac_fly':'sum',
                         'field_error':'sum',
                         'caught_stealing_2b':'sum',
                         'catcher_interf':'sum',
                         'fielders_choice':'sum',
                         'double_play':'sum',
                         'sac_bunt':'sum',
                         'sac_fly_double_play':'sum',
                         'caught_stealing_home':'sum',
                         'caught_stealing_3b':'sum',
                         'pickoff_1b':'sum',
                         'other_out':'sum',
                         'barrel':'mean'
                        }).reset_index()

fi_grouped_2022["hitting_team"] = np.where(fi_grouped_2022["inning_topbot"] == "Bot", #create variables for hitting team, and how many runs they scored
                                              fi_grouped_2022["home_team"], fi_grouped_2022["away_team"])
fi_grouped_2022["hitting_runs"] = np.where(fi_grouped_2022["inning_topbot"] == "Bot", 
                                              fi_grouped_2022["home_score"], fi_grouped_2022["away_score"]) 


fi_grouped_2022['hits'] = fi_grouped_2022['single']+fi_grouped_2022['double']+fi_grouped_2022['triple']+fi_grouped_2022['home_run']
fi_grouped_2022['outs'] = fi_grouped_2022['strikeout']+fi_grouped_2022['grounded_into_double_play']+fi_grouped_2022['field_out']+fi_grouped_2022['fielders_choice_out']+fi_grouped_2022['field_error']+fi_grouped_2022['fielders_choice']+fi_grouped_2022['double_play']+fi_grouped_2022['sac_fly_double_play']
fi_grouped_2022['walks'] = fi_grouped_2022['walk']+fi_grouped_2022['hit_by_pitch']
fi_grouped_2022['xbh'] = fi_grouped_2022['double']+fi_grouped_2022['triple']+fi_grouped_2022['home_run'] #add variables

fi_grouped_2022['avg'] = fi_grouped_2022['hits'] / (fi_grouped_2022['hits'] + fi_grouped_2022['outs']) #create basic statistics
fi_grouped_2022['era'] = fi_grouped_2022['hitting_runs'] * 9
fi_grouped_2022['slg'] = (fi_grouped_2022['single'] + 2*fi_grouped_2022['double']+ 3*fi_grouped_2022['triple']+ 4*fi_grouped_2022['home_run']) / (fi_grouped_2022['hits'] + fi_grouped_2022['outs'])
fi_grouped_2022['obp'] = (fi_grouped_2022['hits'] + fi_grouped_2022['walks']) / (fi_grouped_2022['hits'] + fi_grouped_2022['walks'] +  fi_grouped_2022['outs'])
fi_grouped_2022['ops'] = fi_grouped_2022['slg']+fi_grouped_2022['obp']
fi_grouped_2022['whip'] = fi_grouped_2022['hits'] + fi_grouped_2022['walks']
fi_grouped_2022['games_played'] = 1 #gives us the ability to count # of games started for pitchers
#fi_grouped_2022['fip'] = (((13*fi_grouped_2022['home_run']+3*(fi_grouped_2022['hit_by_pitch']+fi_grouped_2022['walk'])-2*fi_grouped_2022['strikeout'])/fi_grouped_2022['games_played'])*3.2),
fi_grouped_2022['hr9'] = (fi_grouped_2022['home_run'] / fi_grouped_2022['games_played']) * 9

fi_grouped_2022['era'] = fi_grouped_2022['era'].astype('float') #fix data types for modeling
fi_grouped_2022['strikeout'] = fi_grouped_2022['strikeout'].astype('float')
fi_grouped_2022['home_run'] = fi_grouped_2022['home_run'].astype('float')
fi_grouped_2022['whip'] = fi_grouped_2022['whip'].astype('float')
fi_grouped_2022['launch_speed'] = fi_grouped_2022['launch_speed'].astype('float')
fi_grouped_2022['estimated_woba_using_speedangle'] = fi_grouped_2022['estimated_woba_using_speedangle'].astype('float')
fi_grouped_2022['estimated_ba_using_speedangle'] = fi_grouped_2022['estimated_ba_using_speedangle'].astype('float')
fi_grouped_2022['hitting_runs'] = fi_grouped_2022['hitting_runs'].astype('float')


groupsp = fi_grouped_2022.groupby(['player_name']) #creates dataframe of all pitchers' stats
pitchers1 = groupsp.agg({'fielding_team': 'first',
                         'launch_speed': 'mean',
                         'estimated_woba_using_speedangle': 'mean',
                         'estimated_ba_using_speedangle': 'mean',
                         'delta_run_exp': 'sum',
                         'strikeout': 'mean',
                         'avg': 'mean',
                         'era': 'mean',
                         'slg': 'mean',
                         'ops': 'mean',
                         'whip': 'mean',
                         'hitting_runs': 'sum',
                         'games_played':'sum',
                         'hr9': 'mean',
                         'barrel':'mean'
                        }).reset_index()
pitchers1.rename(columns={'fielding_team': 'p_fielding_team',
                         'launch_speed': 'p_launch_speed',
                         'estimated_woba_using_speedangle': 'p_estimated_woba_using_speedangle',
                         'estimated_ba_using_speedangle': 'p_estimated_ba_using_speedangle',
                         'delta_run_exp': 'p_delta_run_exp',
                         'strikeout': 'p_strikeout',
                         'avg': 'p_avg',
                         'era': 'p_era',
                         'slg': 'p_slg',
                         'ops': 'p_ops',
                         'whip': 'p_whip',
                         'hitting_runs': 'p_hitting_runs',
                         'games_played':'p_games_played',
                         'hr9': 'p_hr9',
                         'barrel':'p_barrel'
                          }, inplace=True)

groupsh = fi_grouped_2022.groupby(['hitting_team']) #creates df for all team's hitting stats 
hitters1 = groupsh.agg({'launch_speed': 'mean',
                         'estimated_woba_using_speedangle': 'mean',
                         'estimated_ba_using_speedangle': 'mean',
                         'delta_run_exp': 'sum',
                         'strikeout': 'mean',
                         'avg': 'mean',
                         'slg': 'mean',
                         'ops': 'mean',
                         'hitting_runs': 'sum',
                         'games_played':'sum',
                         'hitting_runs': 'mean',
                         'barrel': 'mean'
                        }).reset_index()
hitters1.rename(columns={'launch_speed': 'h_launch_speed',
                         'estimated_woba_using_speedangle': 'h_estimated_woba_using_speedangle',
                         'estimated_ba_using_speedangle': 'h_estimated_ba_using_speedangle',
                         'delta_run_exp': 'h_delta_run_exp',
                         'strikeout': 'h_strikeout',
                         'avg': 'h_avg',
                         'slg': 'h_slg',
                         'ops': 'h_ops',
                         'hitting_runs': 'h_hitting_runs',
                         'games_played':'h_games_played',
                         'hitting_runs': 'h_avg_runs',
                         'barrel': 'h_barrel'
                          }, inplace=True)


xls = pd.ExcelFile('NRFI.xlsx') #load park factor data from excel -- create your own file
pf = pd.read_excel(xls, 'PF')
pf = pf.iloc[:, 0:2]

fi_grouped_2022 = pd.merge(fi_grouped_2022, pf, on = 'home_team')
df1 = pd.merge(fi_grouped_2022, pitchers1, on='player_name')
df = pd.merge(df1, hitters1, on='hitting_team') #merge stats to create final df
df['run_first_inning_yes'] = np.where(df['hitting_runs'] > 0, 1, 0)

model_df = df[['park_factor','p_strikeout','p_barrel','p_era','p_launch_speed','p_estimated_woba_using_speedangle','p_estimated_ba_using_speedangle','p_hr9','p_whip','p_ops','p_games_played','h_avg_runs','h_strikeout','h_launch_speed','h_barrel','h_estimated_woba_using_speedangle','h_estimated_ba_using_speedangle','h_ops','hitting_runs','run_first_inning_yes']]

model_df.fillna(model_df.mean(), inplace=True)
model_df['p_games_played'] = model_df['p_games_played'].astype('float')

model_df_log1 = model_df.drop('hitting_runs', axis=1)
model_df_log = model_df_log1.sort_index(axis = 1)
