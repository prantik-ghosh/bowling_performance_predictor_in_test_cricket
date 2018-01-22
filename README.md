# Bowling Performance Prediction in Test Cricket


## __Abstract__:
On the game of cricket, Sir Neville Cardus once famously said, "We remember not the scores and the results in after years; it is the men who remain in our minds, in our imagination." That may be true, but ultimately those scores and results definitely help make those men memorable. Here's another one of his famous quotes, "In cricket, as in no other game, a great master may well go back to the pavilion scoreless.... In no other game does the law of averages get to work so potently, so mysteriously." Anybody who watches cricket know how true these words are. The unpredictable nature of the game does make it notoriously difficult to predict the result of a game and gauge an individual player's performance.

In this project, I have tried to achieve exactly that for a bowler. Bowlers often get less accolades than their batting counterparts, but they are the ones win matches for their team especially in the longer version of the game, because a team cannot win a test match unless its bowlers are capable of capturing twenty opposition wickets. In this study, I have tried to predict the number of test wickets a bowler will take in the upcoming season.


## __Technology__:
* Python
  - Pandas and Numpy for data processing/cleaning
  - SciKit Learn and Statsmodels for model fitting (Linear Regression, Random Forest Regressor, and Gradient Boost Regressor)
  - matplotlib for plotting/visualization

* Database
  - PostgreSQL for table creation, data loading, and querying
  - PL/pgSQL for data manipulation/feature engineering


## __Data Retrieval__:
The data (bowler's performance by match) was pulled from popular ESPN Cricinfo website's STATSGURU property (URL: http://stats.espncricinfo.com/ci/engine/stats/index.html?class=1;filter=advanced;type=bowling). Unfortunately, it didn't have all the features like bowler type (pace or spin), bowling arm (right or left) and home/away information available in the output. However, the query form allowed to query on those parameters, which helped to retrieve those features in an indirect fashion.
Following filters were applied before retrieving the data:
1. Data was retrieved for years 2000 to 2017 only. As the game changes pretty rapidly, I believed older data may not be quite relevant.
2. A bowler's performance was counted only for those matches where he bowled at least 60 deliveries, which is equivalent to 10 overs.


## __Data Cleaning__:
At the next step, data files were loaded into pandas dataframes, pulled into a single dataframe and then the following cleaning were performed (1_initial_data_load_and_manipulation.ipynb):
1. Player names were extracted.
2. Players' countries were extracted and standardized.
3. Opposition country name cleaned.
4. Year extracted from the start date.
5. Redundant columns were dropped.
6. Filtered to keep only major test playing nations with respect to number of matches played.
7. Data from the modified dataframe were dumped into a csv file to be loaded into SQL database for further manipulation and feature extraction.


## __Feature Engineering__:
Next, in order to perform feature engineering, a table was created in postgresql database and data were loaded from the csv file (psql_create_and_load_sql_table.sql). Several processes were run to update the table. Following features were extracted:
1. For the last 5 years, how many test matches this player has played and how many wickets per match this player has taken each year (psql_func1_update_bowler_stat.sql)? These features are supposed to measure the bowler's past performance.
2. What is the ratio of average wickets taken per match by this player against this opposition vs the same against all oppositions (psql_func2_update_bowler_oppo_stat.sql)? This feature is supposed to measure if this bowler does better or worse on average against this particular opposition.
3. A measure between 0 and 1 to indicate how well this player performs at home vs how well he does at away venues on average (psql_func3_update_bowler_home_adv_stat.sql).
4. What is the ratio of average wickets taken per match by bowler of this type (pace/spin and right/left arm) against this opponent vs bowler of all types against this opponent in the "last" 5 years(psql_func4_update_oppo_bowltyp_stat.sql)? This feature is supposed to provide a measure how this particular opposition fared against this particular type of bowling.
5. What is the ratio of average wickets taken per match by bowler of this type (pace/spin) in this ground vs average wickets taken per match by bowler of all types in this ground (psql_func5_update_ground_bowltyp_stat.sql)? This feature is supposed to measure how much this particular ground supports pace or spin bowling.
Finally, created a dump of the engineered data in form of a csv file.

*__Note__: Since I needed data from the last 5 years to calculate the engineered features, out of the 2000-2017 data I started with, I could only get these features for 2005-2017 data.*


## __Cross-validation Strategy__:
Out of the data I had from 2005 to 2017, the idea was to train and validate the models on 2000 to 2016 data, and then perform the final testing on 2017 data. However, because of the time series type nature of the data, a standard cross validation wouldn't make sense here. Hence, I decided to work on a rolling window of 6 years data to train a model and use it to predict the following year's performance. So, I started with the years 2005 to 2010 to train a model and validated with 2011 data. Next, I would use 2006-2011 to train the model and validate on 2012. Going forward like this, the last training set would be 2010-2015 data and the corresponding validation data would be 2016. After running the model for all these different training and validation sets, I would calculate average score to compare a model with another. The scores I used for this purpose were MSE (Mean Squared Error) and Explained Variance.


## __Target (Grouping by Player)__:
The idea here was that the model would forecast expected number of wickets for each player for each match in the upcoming season and then we would group that data by players to forecast the expected total number of wickets to be captured by a player in the entire season. Thus, for each bowler, the target would be the number of wickets taken in the entire season, not in each match.


## __Setting the Baseline__:
Last year’s performance is generally a very good indicator of a player’s current year’s performance. Average number of wickets taken per match is a straightforward measure of performance. Hence, before jumping into model fitting, I set the baseline for each player to be the average number of wickets taken per match in the previous year multiplied with the number of matches in the current year.


## __Model fitting and feature selection__:
Initially, I tried Linear Regression and a Grid-Searched Random Forest (run_lr_and_rf_for_various_feature_sets.py). Unfortunately, not one model was consistently superior. Next, I tried a Grid-Searched Gradient Boosting model (run_gb_for_various_feature_sets). After comparing all three models, still none of the models was consistently superior; however, Gradient Boosting came out on top more often than others.

As far as features are concerned, I tried different sets of features and also compared feature importance information returned by the various models. After comparing the features for various models, it was evident that none of the features involving number of matches played in the last 5 years was significant. Also, the feature involving the bowler's relative performance against this particular opposition as well as the feature related to this opposition's relative performance against this particular type of bowling were not significant either. That leaves us with the smaller feature set involving the wickets per match captured by this bowler in each of the last 5 years, the home/away factor, and how the particular ground (venue) supports the bowing type (pace or spin).

Till now, to choose the optimum model, I was solely using test scores. Next, I took the winning GB model and calculated training score (run_models_for_years.py). I found that across all the traning/validation sets, the model's traning score was always much higher than the test score - an average explained variance of 95% on the training data vs an average explained variance of 78% on the validation data. Normally, this would mean the model was overfitting. So, I tried to underfit the GB model by tuning its various hyperparameters. But none of those variants could better the performance of the chosen model. In each case both the training and test score went down. The conclusion to be drawn from here is that, because of the time series type nature of the problem, the training and the validation data are not identically distributed and so there will always be a significant gap between the two scores.


## __Final model run with 2017 test data__:
Ran the optimized GB model for the final test data to predict bowlers' performance in the year 2017 (final_model_run_on_test_data.py). Got an explained variance score of 81% against the baseline score of 66% and a MSE (Mean Squared Error) of 30.2 against a baseline MSE of 59.9.


## __Future Considerations__:
- Consider bowler subtype in the fray. For instance, what kind of spinner - a leg-break, an off-break, a left-arm-orthodox or a chinaman bowler?
- Consider weather data and how it would interact with bowler type. For instance, cloudy heavey conditions support swing bowlers.
- Perhaps consider domestic performance for those bowlers who are new to test cricket.

