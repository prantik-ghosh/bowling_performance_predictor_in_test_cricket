import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error




def fit_gb_model(df, X_train, y_train, X_test, y_test, mask_test):
    """
    Runs a grid-search to find the best gradient boost model for the passed X_train and y_train,
    predicts target for the passed X_test using the best grid-searched model,
    displays the winning model's parameters,
    displays the feature importances,
    groups data by player to find expected number of wickets for the entire season,
    and finally calculates explained variance and MSE.
    """
    print ("**** GRADIENT BOOSTING Grid Search ****")
    gradient_boosting_grid = {'max_depth': [3, None],
                              'max_features': ['sqrt', 'log2', round(X_train.shape[1]/3), None],
                              'n_estimators': [100,300,500],
                              'learning_rate': [0.1,0.05,0.01],
                              'subsample': [0.5,1.0],
                              'random_state': [10]}

    gb_gridsearch = GridSearchCV(GradientBoostingRegressor(),
                                 gradient_boosting_grid,
                                 verbose=1,
                                 scoring='neg_mean_squared_error')
    gb_gridsearch.fit(X_train, y_train)
    print("Best Parameters:", gb_gridsearch.best_params_)
    print(' ')

    best_gb_model = gb_gridsearch.best_estimator_

    feature_importance = {}
    for label, importance in zip(X_train.columns, best_gb_model.feature_importances_):
        feature_importance[label] = importance
    print("Sorted Feature Importance:")
    sorted_feature_imp = sorted(feature_importance.items(), key=lambda x: (-x[1]))
    for e in sorted_feature_imp:
        print(e)

    y_pred_test = best_gb_model.predict(X_test)
    df_test = pd.concat([df[mask_test][['player','wkts','year1_wkts_pm']].reset_index(),
                         pd.DataFrame(y_pred_test).reset_index()],axis=1,)
    df_test = df_test.drop('index',axis=1)
    df_test.columns = ['player','wkts','wkts_baseline','wkts_exp']

    df_by_player = df_test.groupby('player').sum()

    print(' ')
    print('Explained Variance (GB model): ' + str(explained_variance_score(df_by_player.wkts,df_by_player.wkts_exp)))
    print('Explained Variance (Baseline): ' + str(explained_variance_score(df_by_player.wkts,df_by_player.wkts_baseline)))
    print('----')
    print('Mean Squared Error (GB model): ' + str(mean_squared_error(df_by_player.wkts,df_by_player.wkts_exp)))
    print('Mean Squared Error (Baseline): ' + str(mean_squared_error(df_by_player.wkts,df_by_player.wkts_baseline)))
    print('----')
    print(' ')




def fitting_gb_models(file, test_yr):
    """
    Loads the passed file into pandas dataframe,
    defines masks for training and test data for the passed test_yr,
    creates training and test data based on those masks,
    defines four different sets of features,
    and finally for each set of features, calls fit_gb_model.
    """
    df = pd.read_csv(file)

    mask_test = (df.year == test_yr)
    mask_train = (df.year >= test_yr-6) & (df.year <= test_yr-1)

    target = 'wkts'

    features_full = ['year1_mtchs_pld', 'year2_mtchs_pld', 'year3_mtchs_pld', 'year4_mtchs_pld', 'year5_mtchs_pld',
                     'year1_wkts_pm', 'year2_wkts_pm', 'year3_wkts_pm','year4_wkts_pm', 'year5_wkts_pm',
                     'bowler_agnst_oppo', 'oppo_agnst_bowl_typ', 'bowl_home_adv', 'ground_bowl_typ']
    features_small = ['year1_wkts_pm', 'year2_wkts_pm', 'year3_wkts_pm', 'year4_wkts_pm', 'year5_wkts_pm',
                      'bowler_agnst_oppo', 'oppo_agnst_bowl_typ', 'bowl_home_adv', 'ground_bowl_typ']
    features_smaller = ['year1_wkts_pm', 'year2_wkts_pm', 'year3_wkts_pm', 'year4_wkts_pm', 'year5_wkts_pm',
                        'bowl_home_adv', 'ground_bowl_typ']
    features_smallest = ['year1_wkts_pm', 'year2_wkts_pm', 'year3_wkts_pm', 'year4_wkts_pm', 'year5_wkts_pm']

    print("*********************************************")
    print("**** RUNNING MODELS FOR FULL FEATURE SET ****")
    print("*********************************************")

    features = features_full.copy()

    X_train = df[mask_train][features]
    y_train = df[mask_train][target]
    X_test = df[mask_test][features]
    y_test = df[mask_test][target]

    fit_gb_model(df, X_train, y_train, X_test, y_test, mask_test)


    print("**********************************************")
    print("**** RUNNING MODELS FOR SMALL FEATURE SET ****")
    print("**********************************************")

    features = features_small.copy()

    X_train = df[mask_train][features]
    y_train = df[mask_train][target]
    X_test = df[mask_test][features]
    y_test = df[mask_test][target]

    fit_gb_model(df, X_train, y_train, X_test, y_test, mask_test)


    print("************************************************")
    print("**** RUNNING MODELS FOR SMALLER FEATURE SET ****")
    print("************************************************")

    features = features_smaller.copy()

    X_train = df[mask_train][features]
    y_train = df[mask_train][target]
    X_test = df[mask_test][features]
    y_test = df[mask_test][target]

    fit_gb_model(df, X_train, y_train, X_test, y_test, mask_test)


    print("*************************************************")
    print("**** RUNNING MODELS FOR SMALLEST FEATURE SET ****")
    print("*************************************************")

    features = features_smallest.copy()

    X_train = df[mask_train][features]
    y_train = df[mask_train][target]
    X_test = df[mask_test][features]
    y_test = df[mask_test][target]

    fit_gb_model(df, X_train, y_train, X_test, y_test, mask_test)




if __name__ == "__main__":
    input_file = '../../data/bowling_data_enhanced.csv'
    for yr in range(2011,2017):
        fitting_gb_models(input_file, test_yr=yr)
