df['player_name'].unique()

# user input data
player_name_home = input("Home Starting Pitcher (Last, First):")
hitting_team_home = input("Home Team (ABR):")
player_name_away = input("Away Starting Pitcher (Last, First):")
hitting_team_away = input("Away Team (ABR):")
home_team = hitting_team_home
#odds = input("NRFI odds:")

######## 
def american_to_decimal(odds):  #calculate vegas implied probability
    if odds[0] == '+':
        return (float(odds[1:])/100 ) + 1
    elif odds[0] == '-':
        return (100 / float(odds[1:])) + 1
    else:
        return 0
#decimal_odds = american_to_decimal(odds)
#vegas_prob = 1/decimal_odds

#Everything below sets up the prediction dataframe for the game in question. 
#See workbook for more info on what each section does

hitters_home_td1 = {'hitting_team': [hitting_team_home]}
hitters_home_td = pd.DataFrame(hitters_home_td1).reset_index()

pitcher_home_td1 = {'player_name': [player_name_home]}
pitcher_home_td = pd.DataFrame(pitcher_home_td1).reset_index()

hitters_away_td1 = {'hitting_team': [hitting_team_away]}
hitters_away_td = pd.DataFrame(hitters_away_td1).reset_index()

pitcher_away_td1 = {'player_name': [player_name_away]}
pitcher_away_td = pd.DataFrame(pitcher_away_td1).reset_index()

home_td1 = {'home_team': [home_team]}
home_td = pd.DataFrame(home_td1).reset_index()


pred_df1 = df[['player_name','home_team','hitting_team',
               'park_factor',
               'p_launch_speed','p_strikeout','p_barrel','p_estimated_woba_using_speedangle','p_estimated_ba_using_speedangle','p_hr9','p_era','p_whip','p_ops','p_games_played',
               'h_launch_speed','h_strikeout','h_barrel','h_estimated_woba_using_speedangle','h_estimated_ba_using_speedangle','h_ops','h_avg_runs']]

hitters_today_home = pd.merge(hitters_home_td, pred_df1[['hitting_team','h_strikeout','h_barrel','h_avg_runs','h_launch_speed','h_estimated_woba_using_speedangle','h_estimated_ba_using_speedangle','h_ops']], on = 'hitting_team')
pitcher_today_home = pd.merge(pitcher_home_td, pred_df1[['player_name','p_strikeout','p_barrel','p_launch_speed','p_era','p_estimated_woba_using_speedangle','p_estimated_ba_using_speedangle','p_hr9','p_whip','p_ops','p_games_played']])
hitters_today_away = pd.merge(hitters_away_td, pred_df1[['hitting_team','h_strikeout','h_barrel','h_avg_runs','h_launch_speed','h_estimated_woba_using_speedangle','h_estimated_ba_using_speedangle','h_ops']], on = 'hitting_team')
pitcher_today_away = pd.merge(pitcher_away_td, pred_df1[['player_name','p_strikeout','p_barrel','p_launch_speed','p_era','p_estimated_woba_using_speedangle','p_estimated_ba_using_speedangle','p_hr9','p_whip','p_ops','p_games_played']])


home_today = pd.merge(home_td, pred_df1[['home_team','park_factor']], on = 'home_team')

today_stats_home = pd.concat([hitters_today_home, pitcher_today_away, home_today], axis=1) #combine horizontally
today_stats_home = today_stats_home[:1]


today_stats_away = pd.concat([hitters_today_away, pitcher_today_home, home_today], axis=1) #combine horizontally
today_stats_away = today_stats_away[:1]

today_stats = pd.concat([today_stats_away, today_stats_home],    # Combine vertically
                          ignore_index = True,
                          sort = False)

today_stats_predict1 = today_stats.drop(columns = ['index','hitting_team','player_name','home_team'])
today_stats_predict = today_stats_predict1.sort_index(axis = 1)

#rfc prediction
pred_proba = forest.predict_proba(today_stats_predict)[:,1:]
today_stats_predict['pred_proba'] = pred_proba

runs_top = round(today_stats_predict['pred_proba'][0],3)
runs_bot = round(today_stats_predict['pred_proba'][1],3)
total_pred_proba = (runs_top+runs_bot) - (runs_top*runs_bot)

nrfi_prob = round(1-total_pred_proba,3)


print("We predict the probability of a NRFI in the", hitting_team_away,'vs',hitting_team_home,'game is:', f"\033[1m{nrfi_prob}\033[0m\n")
print("The Probability of a run in the top of the first is", f"\033[1m{runs_top}\033[0m", "with a sample size of", f"\033[1m{today_stats_predict['p_games_played'][0]}\033[0m", "games for the", hitting_team_home, "starter")
print("The Probability of a run in the bottom of the first is", f"\033[1m{runs_bot}\033[0m", "with a sample size of", f"\033[1m{today_stats_predict['p_games_played'][1]}\033[0m","games for the", hitting_team_away, "starter")


###############################

means = model_df_log.mean()
means['pred_proba'] = means['run_first_inning_yes']
means.drop('run_first_inning_yes')
df_scaled1 = today_stats_predict / means * 100
df_scaled = df_scaled1.drop(columns = ['run_first_inning_yes'])
df_scaled_top = df_scaled.iloc[[0]]
df_scaled_bot = df_scaled.iloc[[1]]


# Transpose the dataframe to make plotting easier
df_t1 = df_scaled_top.transpose()
df_t1 = df_t1.reset_index()

# Rename the columns for easier understanding
df_t1.columns = ['Column', 'Value']

df_t2 = df_scaled_bot.transpose()
df_t2 = df_t2.reset_index()
df_t2.columns = ['Column', 'Value']

colors1 = ['red' if (val > 100 and col not in ['h_strikeout', 'p_games_played', 'p_strikeout']) 
          or (val < 100 and col in ['h_strikeout', 'p_games_played', 'p_strikeout']) 
          else 'green' 
          for col, val in zip(df_t1['Column'], df_t1['Value'])]

colors2 = ['red' if (val > 100 and col not in ['h_strikeout', 'p_games_played', 'p_strikeout']) 
          or (val < 100 and col in ['h_strikeout', 'p_games_played', 'p_strikeout']) 
          else 'green' 
          for col, val in zip(df_t2['Column'], df_t2['Value'])]


#colors = ['green' if (col.startswith('p_games') and val > 100) or (not col.startswith('p_games') and val < 100) else 'red' for col, val in zip(df_t1['Column'], df_t1['Value'])]
#colors = ['green' if (col.startswith('p_games') and val > 100) or (not col.startswith('p_games') and val < 100) else 'red' for col, val in zip(df_t2['Column'], df_t2['Value'])]


# Plot
fig, axs = plt.subplots(2, figsize=(15,15))

plot1 = sns.barplot(x='Column', y='Value', data=df_t1, palette=colors1, ax=axs[0])
plot1.set(title = 'Top 1 Values')
plot1.set_xlabel('')
axs[0].axhline(100, color='black', linestyle='--')
for i, v in enumerate(df_t1['Value']):
    axs[0].text(i, v + 3, str(round(v, 2)), ha='center', va='bottom', fontsize=12, rotation=0)

plot1.set_xticklabels(plot1.get_xticklabels(), rotation=60, fontsize = 10)

plot2 = sns.barplot(x='Column', y='Value', data=df_t2, palette=colors2, ax=axs[1])
plot2.set(title = 'Bot 1 Values')
plot2.set_xlabel('')
axs[1].axhline(100, color='black', linestyle='--')
for i, v in enumerate(df_t2['Value']):
    axs[1].text(i, v + 3, str(round(v, 2)), ha='center', va='bottom', fontsize=12, rotation=0)
plot2.set_xticklabels(plot2.get_xticklabels(), rotation=60, fontsize = 10)

plt.subplots_adjust(hspace=0.57)

plt.show()

explainer = shap.TreeExplainer(forest)
instance = today_stats_predict
shap_values = explainer.shap_values(instance)
print("Shapley Values:")
shap.summary_plot(shap_values, features=instance, plot_type='bar', plot_size=(10, 6))

plt.show()
