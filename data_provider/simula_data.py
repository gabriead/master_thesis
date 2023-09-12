import random
from datetime import timedelta

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import torch


class SimulaTimeSeries(Dataset):

    #check sequence length with Cise (less then 7 makes no sense?)
    def __init__(self, column_names, player_index, data, n_in, n_out, root_path, flag='train', size=None,
                 features='S', data_path='team_a_complete', sequence_length=7,
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):

        self.column_names = column_names
        self.player_index = player_index
        self.train_players = []
        self.test_player = []
        self.val_player = []
        self.flag = flag
        self.seasonal_patterns = seasonal_patterns

        #if data.ndim == 1:
        #    data = data.reshape(-1, 1)

        self.data = data.copy()
        self.n_in = n_in
        self.n_out = n_out
        self.sequence_length = sequence_length
        self.train_x_date_column = []
        self.test_x_date_column = []
        self.val_x_date_column = []

        self.train_y_date_column = []
        self.test_y_date_column = []
        self.val_y_date_column = []

        #############################################################################################################

        #if size == None:
        #    self.seq_len = 24 * 4 * 4
        #    self.label_len = 24 * 4
        #    self.pred_len = 24 * 4
        #else:
        #    self.seq_len = size[0]
        #    self.label_len = size[1]
        #    self.pred_len = size[2]
        # init

        # Sets whether we are in train/test or validation mode
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.train_date_column = []
        self.test_date_column = []
        self.val_date_column = []

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def create_train_test_data(self):


        # self.scaler = StandardScaler()
        # df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                  self.data_path))

        # read complete data per teamA/teamB
        # df_raw = pd.read_pickle(os.path.join(self.root_path,self.data_path))

        # [start_train_index, start_test_index, start_val_index]
        # border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]

        # [end_train_index, end_test_index, end_val_index]
        # border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]

        # simula variation

        # First column is date!!!!
        # if self.features == 'M' or self.features == 'MS':
        #    cols_data = df_raw.columns[1:]
        #    df_data = df_raw[cols_data]
        # elif self.features == 'S':
        #    df_data = df_raw[[self.target]]

        #        if self.scale:
        #            train_data = df_data[border1s[0]:border2s[0]]
        #            self.scaler.fit(train_data.values)
        #            data = self.scaler.transform(df_data.values)
        #        else:
        #            data = df_data.values

        df_stamp = pd.DataFrame()
        # train
        if self.flag == "train":
            self.data_x = self.X_train
            self.data_y = self.y_train
            df_stamp["date"] = self.train_date_column
        # test
        elif self.flag == "test":
            self.data_x = self.X_test
            self.data_y = self.y_test
            df_stamp["date"] = self.test_date_column
        # val
        else:
            self.data_x = self.X_val
            self.data_y = self.y_val
            df_stamp["date"] = self.val_date_column

        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # only date column
        self.data_stamp = data_stamp

        return self.data_x, self.data_y, self.data_stamp

    # Prepares train/test/val data according to the type
    def __read_data__(self):
        self.create_player_sets()
        self.create_train_test_val_splits()
        data_x_, data_y_, data_stamp_ = self.create_train_test_data()


    def __getitem__(self, index):
        #s_begin = index
        #s_end = s_begin + self.seq_len
        #r_begin = s_end - self.label_len
        #r_end = r_begin + self.label_len + self.pred_len

        #seq_x = self.data_x[s_begin:s_end]
        #seq_y = self.data_y[r_begin:r_end]
        #date_x_mark = self.data_stamp[s_begin:s_end]
        #date_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = self.data.iloc[index, :]
        seq_y = self.y.iloc[index, :]

        date_x_mark =[]
        date_y_mark = []

        #double check here
        if self.flag == "train":
            date_x_mark = self.train_x_date_column
            date_y_mark = self.train_y_date_column
        # test
        elif self.flag == "test":
            date_x_mark = self.test_x_date_column
            date_y_mark = self.test_y_date_column
        # val
        else:
            date_x_mark = self.val_x_date_column
            date_y_mark = self.val_y_date_column

        date_y_mark = self.data_stamp.iloc[index, :]
        date_x_mark = self.data_stamp.iloc[index, :]

        seq_x = torch.tensor(seq_x.values, device = ('gpu')).float()
        seq_y = torch.tensor(seq_y.values, device = ('gpu')).float()

        #what are the types of date_x_mark, date_y_mark

        return seq_x, seq_y, date_x_mark, date_y_mark

    def __len__(self):

        return len(self.data)
        #return len(self.data_x) - self.seq_len - self.pred_len + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):

        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('ft%d_t-%d' % (j + 1, i)) for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            # if i == 0:
            #    names += [('ft%d_t' % (j + 1)) for j in range(n_vars)]
            # else:
            names += [('ft%d_t+%d' % (j + 1, i)) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)

        return agg

    import random
    def create_player_sets(self):
        players = list(self.data['player_name_x'].unique())

        print("amount of players to train each config: ", len(players))

        #        for i in range(nr_players):
        #            print(str(i) + "/" + str(len(players)))
        #            all_but_one = players[:i] + players[i + 1:]
        #            train = df[df['player_name_x'].isin(all_but_one)]
        #            test = df.loc[df['player_name_x'] == players[i]]

        test_player = players[self.player_index]
        training_players = [player for player in players if player != test_player]
        random.shuffle(training_players)
        eval_player = random.choice(training_players)
        training_players.remove(eval_player)

        # train on all but the current player
        # test on current player
        # val on on of the players from the training

        self.training_players = training_players
        self.test_player=test_player
        self.eval_player = eval_player

        return training_players, test_player, eval_player

    def subtract_days_from_date(self,date, days):
        """Subtract days from a date and return the date.

        Args:
            date (string): Date string in YYYY-MM-DD format.
            days (int): Number of days to subtract from date

        Returns:
            date (date): Date in YYYY-MM-DD with X days subtracted.
        """

        subtracted_date = pd.to_datetime(date) - timedelta(days=days)
        subtracted_date = subtracted_date.strftime("%Y-%m-%d")

        return subtracted_date

    def add_days_to_date(self,date, days):
        """Add days to a date and return the date.

        Args:
            date (string): Date string in YYYY-MM-DD format.
            days (int): Number of days to add to date

        Returns:
            date (date): Date in YYYY-MM-DD with X days added.
        """

        added_date = pd.to_datetime(date) + timedelta(days=days)
        added_date = added_date.strftime("%Y-%m-%d")

        return added_date


    def create_train_test_val_splits(self):

        df_train = self.data[self.data['player_name_x'].isin(self.training_players)]
        df_test = self.data[self.data['player_name_x'].isin([self.test_player])]
        df_val = self.data[self.data['player_name_x'].isin([self.eval_player])]

        train = df_train[self.column_names]
        test = df_test[self.column_names]
        val = df_val[self.column_names]

        self.train_date_column = train["date"]
        self.test_date_column = test["date"]
        self.val_date_column = val["date"]

        self.train_x_date_column = [self.subtract_days_from_date(current_date, self.n_in) for current_date in self.train_date_column.to_list()]
        self.test_x_date_column = [self.subtract_days_from_date(current_date, self.n_in) for current_date in self.test_date_column.to_list()]
        self.val_x_date_column = [self.subtract_days_from_date(current_date, self.n_in) for current_date in self.val_date_column.to_list()]

        self.train_y_date_column = [self.add_days_to_date(current_date, self.n_out) for current_date in self.train_date_column.to_list()]
        self.test_y_date_column = [self.add_days_to_date(current_date, self.n_out) for current_date in self.test_date_column.to_list()]
        self.val_y_date_column =[self.add_days_to_date(current_date, self.n_out) for current_date in self.val_date_column.to_list()]

        train.drop("date", axis=1, inplace=True)
        test.drop("date", axis=1, inplace=True)
        val.drop("date",axis=1, inplace=True)

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        val = val.reset_index(drop=True)
        num_features = len(train.columns.tolist())

        train_scalar = StandardScaler()
        train_transformed = train_scalar.fit_transform(train)
        test_transformed = train_scalar.transform(test)
        val_transformed = train_scalar.transform(val)

        train_direct = self.series_to_supervised(train_transformed.copy(), self.n_in, self.n_out)
        test_direct = self.series_to_supervised(test_transformed.copy(), self.n_in, self.n_out)
        val_direct = self.series_to_supervised(val_transformed.copy(), self.n_in, self.n_out)

        feature_columns = train_direct.columns.tolist()[:(self.n_in * num_features)]
        target_columns = test_direct.columns.tolist()[(self.n_in * num_features):]

        self.X_train = train_direct[feature_columns]
        self.y_train = train_direct[target_columns]

        self.X_test = test_direct[feature_columns]
        self.y_test = test_direct[target_columns]

        self.X_val = val_direct[feature_columns]
        self.y_val = val_direct[target_columns]

        return self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val, target_columns, feature_columns



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data.iloc[idx, :]
        y = self.y.iloc[idx, :]

        x_transformed = x.reset_index().drop(["index"], axis=1).to_numpy()
        y_transformed = y.reset_index().drop(["index"], axis=1).to_numpy()

        x_transformed = x_transformed.astype(np.float32)
        y_transformed = y_transformed.astype(np.float32)

        return x_transformed, y_transformed
