import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error




def fit_model_for_yrs(file, model, features, test_yr_from, test_yr_to,
                      show_feat_imp, show_mse, show_graph1, show_graph2, show_graph3):
    """
    Loads data from the passed file into pandas dataframe,
    for each year in the from test_yr_from upto test_yr_to,
    defines masks for training and test data for the passed test_yr,
    creates training and test data based on those masks,
    trains the passed model on the traning data for the passed features,
    predicts on the test data features,
    computes explained variance on training and test data,
    and finally provides an average of those measures over all the years.

    Parameters
    ----------
    file: String
        Input file name.
    model: String
        The model to be trained and tested.
    features: List of Strings
        The list of features to be used by the model.
    test_yr_from: Numeric
        Lower limit of the year range for which the test will be run.
    test_yr_to: Numeric
        Upper limit of the year range for which the test will be run.
    show_feat_imp: Boolean
        Whether to display feature importances.
    show_mse: Boolean
        Whether to calculate and display train and test MSEs.
    show_graph1: Boolean
        Whether to display the scatter plot of actual and expected #wickets by player.
    show_graph2: Boolean
        Whether to display the line plot of actual and expected #wickets by player.
    show_graph3: Boolean
        Whether to display the scatter plot of actual #wickets vs the residuals.
    """
    print('++++++++++ MODEL ++++++++++')
    print(model)
    print('++++++++++ MODEL ++++++++++')
    print('')
    print('++++++++ FEATURES +++++++++')
    print(features)
    print('++++++++ FEATURES +++++++++')
    df = pd.read_csv(file)

    #features = ['year1_wkts_pm', 'year2_wkts_pm', 'year3_wkts_pm', 'year4_wkts_pm', 'year5_wkts_pm',
    #            'bowl_home_adv', 'ground_bowl_typ']
    target = 'wkts'

    exp_var_tot_train = 0
    exp_var_tot_test = 0

    for test_yr in range(test_yr_from, test_yr_to+1):
        print('')
        #print("*********************************************")
        print("************** TEST YEAR: " + str(test_yr) + " **************")
        #print("*********************************************")

        mask_test = (df.year == test_yr)
        mask_train = (df.year >= test_yr-6) & (df.year <= test_yr-1)

        X_train = df[mask_train][features]
        y_train = df[mask_train][target]
        X_test = df[mask_test][features]
        y_test = df[mask_test][target]

        model.fit(X_train, y_train)

        if (show_feat_imp):
            print('')
            print('Feature Importance:')
            feature_importance = {}
            for label, importance in zip(X_train.columns, model.feature_importances_):
                feature_importance[label] = importance
            print("Sorted Feature Importance:")
            sorted_feature_imp = sorted(feature_importance.items(), key=lambda x: (-x[1]))
            for e in sorted_feature_imp:
                print(e)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        df_train = pd.concat([df[mask_train][['player','wkts','year1_wkts_pm']].reset_index(),
                              pd.DataFrame(y_pred_train).reset_index()],axis=1,)
        df_train = df_train.drop('index',axis=1)
        df_train.columns = ['player','wkts','wkts_baseline','wkts_exp']
        df_train_by_player = df_train.groupby('player').sum()

        df_test = pd.concat([df[mask_test][['player','wkts','year1_wkts_pm']].reset_index(),
                             pd.DataFrame(y_pred_test).reset_index()],axis=1,)
        df_test = df_test.drop('index',axis=1)
        df_test.columns = ['player','wkts','wkts_baseline','wkts_exp']
        df_test_by_player = df_test.groupby('player').sum()

        exp_var_tot_train += explained_variance_score(df_train_by_player.wkts, df_train_by_player.wkts_exp)
        exp_var_tot_test += explained_variance_score(df_test_by_player.wkts, df_test_by_player.wkts_exp)

        print(' ')
        print('Explained Variance (Train): ' + str(explained_variance_score(df_train_by_player.wkts,
                                                                            df_train_by_player.wkts_exp)))
        print('Explained Variance (Test): ' + str(explained_variance_score(df_test_by_player.wkts,
                                                                           df_test_by_player.wkts_exp)))
        print('----')
        if (show_mse):
            print('Mean Squared Error (Train): ' + str(mean_squared_error(df_train_by_player.wkts,
                                                                          df_train_by_player.wkts_exp)))
            print('Mean Squared Error (Test): ' + str(mean_squared_error(df_test_by_player.wkts,
                                                                         df_test_by_player.wkts_exp)))
            print('----')

        if (show_graph1):
            df_test_by_player.plot(kind='scatter',x='wkts',y='wkts_exp',figsize=(7,5))
            plt.title("Actual vs Expected #Wkts by Player")
            plt.show()

        if (show_graph2):
            df_test_by_player[['wkts','wkts_exp']].plot(figsize=(12,5))
            plt.title("Actual and Expected #Wkts by Player")
            plt.show()

        if (show_graph3):
            df_test_by_player['residual'] = df_test_by_player.wkts - df_test_by_player.wkts_exp
            df_sorted = df_test_by_player.sort_values(['wkts','residual'])
            df_sorted.plot(kind='scatter',x='wkts',y='residual',figsize=(8,5),c='m')
            plt.title("Actual #Wkts by Player vs Residual")
            plt.show()

        print(' ')

    print('========== AVERAGE EXPLAINED VARIANCE ==========')
    print('Train: ' + str(exp_var_tot_train/(test_yr_to - test_yr_from+1)))
    print('Test: ' + str(exp_var_tot_test/(test_yr_to - test_yr_from+1)))
    print('========== AVERAGE EXPLAINED VARIANCE ==========')
    print('')




if __name__ == "__main__":
    input_file = '../../data/bowling_data_enhanced.csv'

    model1 = GradientBoostingRegressor(learning_rate=0.01, n_estimators=300, max_depth=3, max_features=2,
                                       subsample=0.5, verbose=0, random_state=10)
    model2 = GradientBoostingRegressor(learning_rate=0.005, n_estimators=300, max_depth=3, max_features=2,
                                       subsample=0.5, verbose=0, random_state=10)
    model3 = GradientBoostingRegressor(learning_rate=0.01, n_estimators=100, max_depth=3, max_features=2,
                                       subsample=0.5, verbose=0, random_state=10)
    model4 = GradientBoostingRegressor(learning_rate=0.01, n_estimators=300, max_depth=2, max_features=2,
                                       subsample=0.5, verbose=0, random_state=10)
    models = [model1, model2, model3, model4]

    features = ['year1_wkts_pm', 'year2_wkts_pm', 'year3_wkts_pm', 'year4_wkts_pm', 'year5_wkts_pm',
                'bowl_home_adv', 'ground_bowl_typ']

    for model in models:
        fit_model_for_yrs(input_file, model, features, 2011, 2017, False, False, False, False, False)
