import unittest
from data_provider import simula_data
import pandas as pd
import os

class MyTestCase(unittest.TestCase):
    def test_train_test_val_sets(self):

        root_path = os.getcwd()
        data_path = "complete_dataset.csv"
        path = os.path.join(root_path, data_path)
        df = pd.read_csv(path)
        df = df[df["player_name_x"].str.startswith("TeamA")]

        #check if all of the columns make sense (remember Cise said sleep was useless)
        column_names = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "stress"]

        player_index = 0
        data = df
        n_in = 1
        n_out = 7
        simulaData = simula_data.SimulaTimeSeries(column_names=column_names, player_index=player_index, data=data, n_in=n_in, n_out=n_out,root_path=root_path )
        training_players, test_player, eval_player = simulaData.create_player_sets()

        self.assertEqual(len(training_players),23)
        self.assertEqual(len([test_player]), 1)
        self.assertEqual(len([eval_player]), 1)
        self.assertTrue(test_player not in [training_players])
        self.assertTrue(eval_player not in [training_players])
        self.assertTrue(eval_player not in [test_player])

    def test_train_test_val_splits(self):

        root_path = os.getcwd()
        data_path = "complete_dataset.csv"
        path = os.path.join(root_path, data_path)
        df = pd.read_csv(path)
        df = df[df["player_name_x"].str.startswith("TeamA")]

        #check if all of the columns make sense (remember Cise said sleep was useless)
        column_names = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "stress"]

        player_index = 0
        data = df
        n_in = 1
        n_out = 7
        simulaData = simula_data.SimulaTimeSeries(column_names=column_names, player_index=player_index, data=data, n_in=n_in, n_out=n_out,root_path=root_path )
        training_players, test_player, eval_player = simulaData.create_player_sets()

        self.assertEqual(len(training_players),23)
        self.assertEqual(len([test_player]), 1)
        self.assertEqual(len([eval_player]), 1)
        self.assertTrue(test_player not in [training_players])
        self.assertTrue(eval_player not in [training_players])
        self.assertTrue(eval_player not in [test_player])

        X_train, y_train, X_test, y_test, X_val, y_val, target_columns, feature_columns = simulaData.create_train_test_val_splits()

        self.assertEqual(len(X_train.columns.tolist()), len(feature_columns))
        self.assertEqual(len(y_train.columns.tolist()), len(target_columns))

        self.assertEqual(len(X_test.columns.tolist()), len(feature_columns))
        self.assertEqual(len(y_test.columns.tolist()), len(target_columns))


        self.assertEqual(len(X_val.columns.tolist()), len(feature_columns))
        self.assertEqual(len(y_val.columns.tolist()), len(target_columns))

if __name__ == '__main__':
    unittest.main()
