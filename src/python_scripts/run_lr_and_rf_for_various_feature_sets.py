import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm




def fit_lr_model(df, X_train, y_train, X_test, y_test, mask_test):
    """
    Trains a linear regression model for the passed X_train and y_train,
    predicts target for the passed X_test using the trained model,
    groups data by player to find expected number of wickets for the entire season,
    and finally calculates explained variance and MSE.
    """
    print("**** LINEAR REGRESSION ****")
    lin_mod = sm.OLS(y_train, sm.add_constant(X_train))
    fit_lin = lin_mod.fit()
    print(fit_lin.summary())

    y_pred_test = fit_lin.predict(sm.add_constant(X_test))
    df_test = pd.concat([df[mask_test][['player','wkts','year1_wkts_pm']].reset_index(),
                         pd.DataFrame(y_pred_test).reset_index()],axis=1,)
    df_test = df_test.drop('index',axis=1)
    df_test.columns = ['player','wkts','wkts_baseline','wkts_exp']

    df_by_player = df_test.groupby('player').sum()

    print('Explained Variance (LR model): ' + str(explained_variance_score(df_by_player.wkts,df_by_player.wkts_exp)))
    print('Explained Variance (Baseline): ' + str(explained_variance_score(df_by_player.wkts,df_by_player.wkts_baseline)))
    print('----')
    print('Mean Squared Error (LR model): ' + str(mean_squared_error(df_by_player.wkts,df_by_player.wkts_exp)))
    print('Mean Squared Error (Baseline): ' + str(mean_squared_error(df_by_player.wkts,df_by_player.wkts_baseline)))
    print('----')
    print(' ')




def fit_rf_model(df, X_train, y_train, X_test, y_test, mask_test):
    """
    Runs a grid-search to find the best random forest model for the passed X_train and y_train,
    predicts target for the passed X_test using the best grid-searched model,
    displays the winning model's parameters,
    displays the feature importances,
    groups data by player to find expected number of wickets for the entire season,
    and finally calculates explained variance and MSE.
    """
    print ("**** RANDOM FOREST Grid Search ****")
    random_forest_grid = {'max_depth': [3, None],
                          'max_features': ['sqrt', 'log2', round(X_train.shape[1]/3), None],
                          'min_samples_split': [2, 4],
                          'min_samples_leaf': [1, 2, 4],
                          'bootstrap': [True, False],
                          'n_estimators': [100,300,500],
                          'random_state': [10]}

    rf_gridsearch = GridSearchCV(RandomForestRegressor(),
                                 random_forest_grid,
                                 n_jobs=-1,
                                 verbose=True,
                                 scoring='neg_mean_squared_error')
    rf_gridsearch.fit(X_train, y_train)
    print("Best Parameters:", rf_gridsearch.best_params_)
    print(' ')

    best_rf_model = rf_gridsearch.best_estimator_

    feature_importance = {}
    for label, importance in zip(X_train.columns, best_rf_model.feature_importances_):
        feature_importance[label] = importance
    print("Sorted Feature Importance:")
    sorted_feature_imp = sorted(feature_importance.items(), key=lambda x: (-x[1]))
    for e in sorted_feature_imp:
        print(e)

    y_pred_test = best_rf_model.predict(X_test)
    df_test = pd.concat([df[mask_test][['player','wkts','year1_wkts_pm']].reset_index(),
                         pd.DataFrame(y_pred_test).reset_index()],axis=1,)
    df_test = df_test.drop('index',axis=1)
    df_test.columns = ['player','wkts','wkts_baseline','wkts_exp']

    df_by_player = df_test.groupby('player').sum()

    print(' ')
    print('Explained Variance (RF model): ' + str(explained_variance_score(df_by_player.wkts,df_by_player.wkts_exp)))
    print('Explained Variance (Baseline): ' + str(explained_variance_score(df_by_player.wkts,df_by_player.wkts_baseline)))
    print('----')
    print('Mean Squared Error (RF model): ' + str(mean_squared_error(df_by_player.wkts,df_by_player.wkts_exp)))
    print('Mean Squared Error (Baseline): ' + str(mean_squared_error(df_by_player.wkts,df_by_player.wkts_baseline)))
    print('----')
    print(' ')




def fitting_lr_and_rf(file, test_yr, fit_lr, fit_rf):
    """
    Loads the passed file into pandas dataframe,
    defines masks for training and test data for the passed test_yr,
    creates training and test data based on those masks,
    defines four different sets of features,
    and finally for each set of features, calls fit_lr_model and fit_rf_model.

    Parameters
    ----------
    file: String
        Input file name.
    test_yr: Numeric
        Year for which test will be run. Training will be done using last 6 years' data.
    fit_lr: Boolean
        Whether to fit the linear regression model.
    fit_rf: Boolean
        Whether to fit the grid-searched random forest model.
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

    if fit_lr:
        fit_lr_model(df, X_train, y_train, X_test, y_test, mask_test)

    if fit_rf:
        fit_rf_model(df, X_train, y_train, X_test, y_test, mask_test)


    print("**********************************************")
    print("**** RUNNING MODELS FOR SMALL FEATURE SET ****")
    print("**********************************************")

    features = features_small.copy()

    X_train = df[mask_train][features]
    y_train = df[mask_train][target]
    X_test = df[mask_test][features]
    y_test = df[mask_test][target]

    if fit_lr:
        fit_lr_model(df, X_train, y_train, X_test, y_test, mask_test)

    if fit_rf:
        fit_rf_model(df, X_train, y_train, X_test, y_test, mask_test)


    print("************************************************")
    print("**** RUNNING MODELS FOR SMALLER FEATURE SET ****")
    print("************************************************")

    features = features_smaller.copy()

    X_train = df[mask_train][features]
    y_train = df[mask_train][target]
    X_test = df[mask_test][features]
    y_test = df[mask_test][target]

    if fit_lr:
        fit_lr_model(df, X_train, y_train, X_test, y_test, mask_test)

    if fit_rf:
        fit_rf_model(df, X_train, y_train, X_test, y_test, mask_test)


    print("*************************************************")
    print("**** RUNNING MODELS FOR SMALLEST FEATURE SET ****")
    print("*************************************************")

    features = features_smallest.copy()

    X_train = df[mask_train][features]
    y_train = df[mask_train][target]
    X_test = df[mask_test][features]
    y_test = df[mask_test][target]

    if fit_lr:
        fit_lr_model(df, X_train, y_train, X_test, y_test, mask_test)

    if fit_rf:
        fit_rf_model(df, X_train, y_train, X_test, y_test, mask_test)




if __name__ == "__main__":
    input_file = '../../data/bowling_data_enhanced.csv'
    for yr in range(2011,2017):
        fitting_lr_and_rf(input_file, test_yr=yr, fit_lr=True, fit_rf=True)
