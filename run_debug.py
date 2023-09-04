import argparse

import pandas as pd
import torch

from exp.exp_long_term_forecasting_debug import Exp_Long_Term_Forecast_Debug

import random
import numpy as np


#def run_debug():
if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    parser = argparse.ArgumentParser(description='TimesNet')
    model_name = "TimesNet"

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')

    parser.task_name = "long_term_forecast"
    parser.is_training = 1
    parser.model_id = "model_id"
    parser.model = "Autoformer"
    parser.data = "team_a_complete"
    parser.root_path = './dataset/simula/'
    parser.itr = 1
    parser.data = "SimulaTimeSeries"
    parser.features = "M"
    parser.seq_len = 96
    parser.label_len = 48
    parser.pred_len = 96
    parser.d_model = 512
    parser.n_heads = 8
    parser.d_ff = 2048
    parser.e_layers = 3
    parser.d_layers = 7
    parser.factor = 7
    parser.enc_in = 7
    parser.dec_in = 7
    parser.c_out = 7
    parser.embed = "timeF"
    parser.distil = True
    parser.des = "Exp"
    parser.itr = 1
    parser.use_gpu = True
    parser.use_multi_gpu = False
    parser.gpu = 0
    parser.output_attention = True
    parser.moving_avg = 25
    parser.freq = 'h'
    parser.dropout = 0.1
    parser.activation = 'gelu'
    parser.batch_size = 32
    parser.data_path = 'complete_dataset.csv'
    parser.target = 'OT'
    parser.seasonal_patterns = 'Monthly'
    parser.num_workers = 10
    parser.checkpoints = './checkpoints/'
    parser.patience = 3
    parser.learning_rate = 0.0001
    parser.use_amp = False
    parser.train_epochs = 10
    parser.player_index = 0
    use_gpu = True if torch.cuda.is_available() else False
    print(True if torch.cuda.is_available() else False)

    if parser.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast_Debug

    import os
    df = pd.read_csv(os.path.join(parser.root_path, parser.data_path))
    df = df[df["player_name_x"].str.startswith("TeamA")]
    players = list(df['player_name_x'].unique())
    parser.data_dataframe = df

    for player_index in players:
        if parser.is_training:
            for ii in range(parser.itr):
                # setting record of experiments
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    parser.task_name,
                    parser.model_id,
                    parser.model,
                    parser.data,
                    parser.data_dataframe,
                    parser.features,
                    parser.seq_len,
                    parser.label_len,
                    parser.pred_len,
                    parser.d_model,
                    parser.n_heads,
                    parser.e_layers,
                    parser.d_layers,
                    parser.d_ff,
                    parser.factor,
                    parser.embed,
                    parser.distil,
                    parser.des,
                    parser.player_index,
                    ii)

                exp = Exp(parser)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
                torch.cuda.empty_cache()
        else:
            ii = 0
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                parser.task_name,
                parser.model_id,
                parser.model,
                parser.data,
                parser.features,
                parser.seq_len,
                parser.label_len,
                parser.pred_len,
                parser.d_model,
                parser.n_heads,
                parser.e_layers,
                parser.d_layers,
                parser.d_ff,
                parser.factor,
                parser.embed,
                parser.distil,
                parser.des, ii
            )

            exp = Exp(parser)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
            torch.cuda.empty_cache()

#run_debug()
