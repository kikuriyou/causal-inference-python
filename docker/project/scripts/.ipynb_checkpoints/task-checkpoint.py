#!/usr/bin/env python3

####
# TODO:
# - 普通の回帰も入れる
# - LTV読み込む方も直す？→集計側で処理する？
####

import os
import math
import argparse
from decimal import Decimal, ROUND_HALF_UP
import copy
from time import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import pystan


def parse_arguments():
    """Parse job arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exec_date',
        help='Execution date',
        required=True
    )
    parser.add_argument(
        '--uu_type',
        help='UU type',
        required=True
    )
    parser.add_argument(
        '--debug',
        help='Debug mode',
        required=True
    )
    parser.add_argument(
        '--input_dir',
        help='Input directory',
        required=True
    )
    parser.add_argument(
        '--output_dir',
        help='Output directory',
        required=True
    )
    parser.add_argument(
        '--model_dir',
        help='Model directory',
        required=True
    )
    parser.add_argument(
        '--model_ver',
        help='Model version',
        required=True
    )
    """
    coef_lim: トレンド成分の変化量の上限
    - 小さいほどトレンド変化が小さくなる
    - 目安: 0.01~0.02
    """
    parser.add_argument(
        '--coef_lim',
        help='Regularization coefficient for trend component',
        default=0.015
    )
    """
    extract_method: 結果の抽出方法
    'mean': 平均値を抽出
    'mode': KDEの最頻値を抽出
    ※ isVarying=Trueの場合は'mean'でやる → TODO: そのうち直す
    """
    parser.add_argument(
        '--extract_method',
        help='Method to extract results from MCMC samples',
        default='mean'
    )

    args = parser.parse_args()
    arguments = args.__dict__
    params = {k: arg for k, arg in arguments.items() if arg is not None}

    if (params['uu_type'] == 'NUU') | (params['uu_type'] == 'RUU') | (params['uu_type'] == 'DAU'):
        params['uu_type'] == params['uu_type'].lower()

    params['vary'] = '_varying' if params['uu_type']=='nuu' else ''

    if params['debug'] == 'True':
        params['n_iter'] = 1000
        params['n_warmup'] = 500
    else:
        params['n_iter'] = 8000
        params['n_warmup'] = 1500
    
    if not os.path.exists(params['output_dir']):
        os.makedirs(params['output_dir'])

    return params


def set_columns(UU_TYPE):
    """計算対象カラムの指定"""
    if UU_TYPE == 'nuu':
        feat_y = 'NUU_organic'
        feat_x = ['TVCM_time', 'TVCM_spot', 'App_Store', 'Google_Play', 'Fes']
    elif UU_TYPE == 'ruu':
        feat_y = 'RUU_organic'
        feat_x = ['TVCM_time', 'TVCM_spot', 'App_Store', 'Google_Play', 'Fes']
    elif UU_TYPE == 'dau':
        feat_y = 'DAU'
        feat_x = ['TVCM_time', 'TVCM_spot', 'Fes']
    return feat_x, feat_y


def modify_push_ruu(df, col, date_push):
    df_copy = df.copy()
    for i in range(1, df[col].shape[0]):
        if (df_copy[col][i-1] == 0) & (df_copy[col][i] == 1):
            df[col][i] = 1
        else:
            df[col][i] = 0
    df[col][df['date']==date_push] = 1

    return df


def generate_stan_data(df, feat_y, feat_x, coef_lim):
    """
    Stanの入力データを生成する → 以下と等価
    stan_data = {
        't_max': df.shape[0]
        , 'y': df[feat_y]
        , 'trend_limit': df[feat_y].mean()*coef_lim
        , 'x1': df[feat_x[0]]
        , 'x2': df[feat_x[1]]
        , .....(以下、回帰成分数に応じて増える)
    }
    """
    keys = ['t_max', 'y', 'trend_limit']
    keys_reg = ['x{}'.format(i+1) for i in range(len(feat_x))]
    keys.extend(keys_reg)
    
    values = [df.shape[0], df[feat_y], df[feat_y].mean()*coef_lim]
    values_reg = [df[feat] for feat in feat_x]
    values.extend(values_reg)
    
    stan_data = dict(zip(keys, values))
    
    return stan_data


def save_mcmc_res(fit, OUTPUT_DIR, UU_TYPE):
    """サンプリング結果を出力"""
    # fit.summary を出力
    summary_file = os.path.join(OUTPUT_DIR, 'mcmc_{}.txt'.format(UU_TYPE))
    with open(summary_file, mode='w') as f:
        f.write(str(fit))
    
    # rhat を降順に並べ替えて出力
    feats = fit.sim['fnames_oi']
    irhat = fit.summary()['summary_colnames']
    irhat = irhat.index('Rhat')
    rhat = fit.summary()['summary'][:, irhat]

    rhat_file = os.path.join(OUTPUT_DIR, 'mcmc_ordered_{}.txt'.format(UU_TYPE))
    rhat_df = pd.DataFrame({'feats': feats, 'Rhat': rhat})
    rhat_df.sort_values('Rhat', ascending=False).to_csv(rhat_file)


def kde_mode(y):
    """
    Parameters
    y: 任意の確率分布(scipyオブジェクト)
    
    Returns
    kde: カーネル密度推定結果(n_grid個に離散化) → 今は使わないので出力値から外す
    x_mode: カーネル密度が最大になるx
    """
    n_grid = 100
    x = np.linspace(min(y), max(y), num=n_grid)
    kde = [gaussian_kde(y)(x[i])[0] for i in range(n_grid)]

    kde_df = pd.DataFrame({'x': x, 'kde': kde})
    x_mode = kde_df[kde_df['kde']==kde_df['kde'].max()]['x'].values[0]
    
    return x_mode


def get_mean(UU_TYPE, y_pred_arr, y_trend_arr, y_dow_arr, y_a_arr, y_b_arr):
    """分布の平均を代表値として抽出"""
    # 予測値、トレンド成分
    y_pred = y_pred_arr.mean(axis=0)
    y_trend = y_trend_arr.mean(axis=0)

    # 周期成分
    if (UU_TYPE == 'nuu') | (UU_TYPE == 'dau'):
        y_dow = y_dow_arr.mean(axis=0)
    else:
        y_dow = np.zeros(len(y_trend))

    # 回帰成分
    y_a = [a.mean(axis=0) for a in y_a_arr]
    y_b = y_b_arr.mean(axis=0)

    # トレンド推移(トレンド成分+回帰成分の定数項)
    cum_mean = np.cumsum(y_trend)
    trend = cum_mean + y_b
    
    return y_pred, y_trend, y_dow, y_a, y_b, trend
    
    
def get_mode(y_pred_arr, y_trend_arr, y_dow_arr, y_a_arr, y_b_arr):
    """分布のKDE最頻値を代表値として抽出 → 少し時間がかかる"""
    print('\nThis method takes a few seconds. Please wait.....')
    t0 = time()

    # 予測値、トレンド成分
    y_pred = [kde_mode(prd) for prd in y_pred_arr.T]
    y_trend = [kde_mode(trd) for trd in y_trend_arr.T]

    # 周期成分
    if (UU_TYPE == 'nuu') | (UU_TYPE == 'dau'):
        y_dow = [kde_mode(dow) for dow in y_dow_arr.T]
    else:
        y_dow = np.zeros(len(y_trend))

    # 回帰成分
    y_a = [kde_mode(a) for a in y_a_arr]
    y_b = kde_mode(y_b_arr)

    # トレンド推移(トレンド成分+回帰成分の定数項)
    cum_weight = np.cumsum(y_trend)
    trend = cum_weight + y_b

    print('Process done. Elapsed time: {}s \n'.format(time() -  t0))
    
    return y_pred, y_trend, y_dow, y_a, y_b, trend


def extract_results(fit, UU_TYPE, num_reg):
    feat_y_pred = 'y_pred'
    feat_trend = 'trend_pred'
    if (UU_TYPE == 'nuu') | (UU_TYPE == 'dau'):
        feat_dow = 'dow_pred'
    feat_a = ['a{}'.format(str(i+1)) for i in range(num_reg)]
    feat_b = 'b'

    y_pred_arr = fit.extract(feat_y_pred)[feat_y_pred]
    y_trend_arr = fit.extract(feat_trend)[feat_trend]
    if (UU_TYPE == 'nuu') | (UU_TYPE == 'dau'):
        y_dow_arr = fit.extract(feat_dow)[feat_dow]
    else:
        y_dow_arr = None
    y_a_arr = [fit.extract(feat)[feat] for feat in feat_a]
    y_b_arr = fit.extract(feat_b)[feat_b]

    return y_pred_arr, y_trend_arr, y_dow_arr, y_a_arr, y_b_arr


def MAPE(true, pred):
    """MAPE(Mean Absolute Percentage Error)を算出"""
    return (abs(pred - true) / true).mean()


def save_csv(x, trend, y_dow, feat_x, y_a, mape, EXEC_DATE, OUTPUT_DIR, UU_TYPE, MODEL_VER):
    # トレンド成分
    df_table = pd.DataFrame({'date': x, 
                            'exec_date': EXEC_DATE,
                            'type': 'output',
                            'component': 'trend',
                            'channel': '',
                            'uu_effect': trend,
                            'mape': mape,
                            'model_version': MODEL_VER})

    # 周期成分
    df_table = pd.concat([
        df_table, pd.DataFrame({'date': x, 
                                'exec_date': EXEC_DATE,
                                'type': 'output',
                                'component': 'day_of_week',
                                'channel': '',
                                'uu_effect': y_dow,
                                'mape': mape,
                                'model_version': MODEL_VER})
    ])

    # 回帰成分
    for i in range(len(feat_x)):
        df_table = pd.concat([
            df_table, pd.DataFrame({'date': x, 
                                    'exec_date': EXEC_DATE,
                                    'type': 'output',
                                    'component': 'regression',
                                    'channel': feat_x[i],
                                    'uu_effect': np.mean(y_a[i]),
                                    'mape': mape,
                                    'model_version': MODEL_VER})
        ])

    df_table.reset_index(drop=True, inplace=True)
    table_file = os.path.join(OUTPUT_DIR, '{}_table.csv'.format(UU_TYPE))
    df_table.to_csv(table_file, header=False, index=False)


def main(args):
    # データ読み込み
    print('Loading input data....')
    input_file = os.path.join(args['input_dir'], args['uu_type']+'.csv')
    df = pd.read_csv(input_file)

    # RUUの場合のみfes影響を変更
    """
    NUU, DAUはfesの盛り上がりを反映する → 期間中は1
    RUUは主にpushに影響を受けるものと考える → 期間初日のみ1 (2016-03-02は別途効果があるpushがあったため)
    """
    if args['uu_type'] == 'ruu':
        modify_push_ruu(df, col='fes', date_push='2016-03-02')


    # Stanモデルのコンパイル
    print('Compiling Stan model....')
    t0 = time()
    model_file = os.path.join(args['model_dir'], 
        'state_space_model_{}{}.stan'.format(args['uu_type'], args['vary']))
    sm = pystan.StanModel(file = model_file)
    print('Elapsed time for compiling Stan model: {}s'.format(time() - t0))


    # 計算対象カラムの指定
    feat_x, feat_y = set_columns(args['uu_type'])

    # 入力データの生成
    stan_data = generate_stan_data(df, feat_y, feat_x, args['coef_lim'])

    # MCMCサンプリング
    print('Executing MCMC sampling....')
    t0 = time()
    fit = sm.sampling(data = stan_data, iter=args['n_iter'], warmup=args['n_warmup'], seed=1, chains=3)
    print('Elapsed time for MCMC sampling in Stan: {:8.2f}s'.format(time() - t0))

    # MCMCサンプリングの収束状況を出力
    print('Saving sampling conversion conditions....')
    save_mcmc_res(fit, args['output_dir'], args['uu_type'])

    # 結果を抽出
    print('Extracting results')
    x = df['date']
    y_pred_arr, y_trend_arr, y_dow_arr, y_a_arr, y_b_arr = extract_results(fit, args['uu_type'], len(feat_x))    # 分布を抽出
    y_pred, y_trend, y_dow, y_a, y_b, trend = get_mean(args['uu_type'], y_pred_arr, y_trend_arr, y_dow_arr, y_a_arr, y_b_arr)    # 代表値に絞る

    # 回帰成分の係数を表示
    for i in range(len(feat_x)):
        print('Coefficient of x{}: {:8.4f}   ({})'.format(str(i), np.mean(y_a[i]), feat_x[i]))

    # MAPE(Mean Absolute Percentage Error) を算出
    mape = MAPE(df[feat_y], y_pred)
    print('MAPE of estimated result: {:.4f}'.format(mape))

    # BQアップロード用のcsvを出力
    print('Saving csv file....')
    save_csv(x, trend, y_dow, feat_x, y_a, mape, args['exec_date'], args['output_dir'], args['uu_type'], args['model_ver'])


if __name__ == '__main__':
    job_args = parse_arguments()
    main(job_args)
