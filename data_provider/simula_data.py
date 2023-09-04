import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
from utils.timefeatures import time_features


class SimulaTimeSeries(Dataset):

    def __init__(self, column_names, player_index, data, n_in, n_out, root_path, flag='train', size=None,
                 features='S', data_path='team_a_complete',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):

        self.column_names = column_names
        self.player_index = player_index
        self.train_players = []
        self.test_player = []
        self.val_player = []

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self.data = data.copy()
        self.n_in = n_in
        self.n_out = n_out

        self.start_indices_train_test_val =[]
        self.end_indices_train_test_val = []


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

        self.root_path = root_path
        self.data_path = data_path
        #self.__read_data__()

    def create_train_test_data(self):

        if self.flag == "train":
            self.data = self.X_train
            self.y = self.y_train
        elif self.flag == "val":
            self.data = self.X_val
            self.y = self.y_val
        elif self.flag == "test":
            self.data = self.X_test
            self.y = self.y_test
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

        # df_simula_data = self.create_normalized_raw_data_features_and_target(df_raw, self.column_names, self.n_in,self.n_out)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # train
        if self.set_type == 0:
            self.data_x = self.X_train
            self.data_y = self.y_train
        # test
        elif self.set_type == 1:
            self.data_x = self.X_test
            self.data_y = self.y_test
        # val
        else:
            self.data_x = self.X_val
            self.data_y = self.y_val

        # num_features = len(df_simula_data.columns.tolist())
        # raw_data_features = df_simula_data.columns.tolist()[:(self.n_in * num_features)]
        # raw_data_targets = df_simula_data.columns.tolist()[(self.n_in * num_features):]

        self.data_x = self.data_x  # df_simula_data[border1:border2]
        self.data_y = self.data_y  # df_simula_data[border1:border2]

        # only date column
        self.data_stamp = data_stamp

    # Prepares train/test/val data according to the type
    def __read_data__(self):

        self.create_player_sets(self.data, self.player_index)
        self.create_train_test_val_splits(self.data, self.column_names, self.n_in, self.n_out)
        #self.create_train_test_data()



    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


    def __len_simula__(self):
        return len(self.data)


    def __getitem_simula__(self, idx):
        x = self.data.iloc[idx, :]
        y = self.y.iloc[idx, :]

        x_transformed = x.reset_index().drop(["index"], axis=1).to_numpy()
        y_transformed = y.reset_index().drop(["index"], axis=1).to_numpy()

        x_transformed = x_transformed.astype(np.float32)
        y_transformed = y_transformed.astype(np.float32)

        return x_transformed, y_transformed

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


    def create_train_test_val_splits(self):

        df_train = self.data[self.data['player_name_x'].isin(self.training_players)]
        df_test = self.data[self.data['player_name_x'].isin([self.test_player])]
        df_val = self.data[self.data['player_name_x'].isin([self.eval_player])]

        train = df_train[self.column_names]
        test = df_test[self.column_names]
        val = df_val[self.column_names]

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
