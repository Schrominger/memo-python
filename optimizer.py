# -*- coding: utf-8 -*-
# @201812

import time
import os
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing

# READ the newest tutorial v1.1.0 rather than v0....++..
from scipy import optimize as spo
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html

import sys
sys.path.append("..")
# my modules
from strategy.rsv5m import Strg_rsv5min

class optimizer(object):

    def __init__(self, holdbar, *args, **kwargs):
        # set holdbar and parameter range.
        self.holdbar = holdbar
        self.wkbar = kwargs.get('wkbar')
        self.xmin = kwargs.get('xmin', 0.1)
        self.xmax = kwargs.get('xmax', 1.0)
        # ad hoc
        self.offset = kwargs.get('offset', 5000)

        # realize after read_data()
        self.DataBase = None
        self.tradeDate = None
        self.timeStampList = None

    # @classmethod
    def read_data(self, h5dataPath, key):
        """READ DATA FROM HDF5 FILE"""
        if self.DataBase is None:
            print('READ DATA FROM H5 FILE.')
            data_store = pd.HDFStore(h5dataPath)
            # Retrieve data using key
            DF = data_store[key]
            data_store.close()

            self.DataBase = DF[DF['tradeDate']>20110101]
            del DF # for saving memory
        else:
            print('DataBase already in hands !')

        if self.timeStampList is None:
            # take all timeStamp
            all_timestamp_list = sorted(set(self.DataBase.timeStamp))
            self.timeStampList = all_timestamp_list[self.offset::self.holdbar]

        print(self.timeStampList)
        print(self.DataBase.head())
        print('Data in hands now!')
        print('holdbar=%s, wkbar=%s,(xmin,xmax)=(%.4f,%.4f)' \
                %(self.holdbar, self.wkbar, self.xmin, self.xmax))

    def optimizer(self, timestamp, x0, WKBAR,**kwargs):
        """ OPTIMIZATION FOR CURRENT BAR
        tiemstamp: current bar label
        x0: initial guess, if not parallel, it should be last x_op.
        WKBAR: the window size observed for some perfomrance scores.
        """

        bounds = [(self.xmin, self.xmax)]
        take_step = RandomDisplacementBounds(self.xmin, self.xmax)

        (niter, niter_success) = (250,25)
        minimizer_kwargs = dict(args=(timestamp,WKBAR), method='L-BFGS-B', bounds=bounds)

        prnt = kwargs.get('prnt',True)
        if prnt:
            ret = spo.basinhopping(self.fcn, x0, T=1.0, minimizer_kwargs=minimizer_kwargs, niter=niter, \
                            niter_success=niter_success,take_step=take_step,callback=self.print_fun)
            # ret = spo.basinhopping(self.fcn, x0, T=1.0, minimizer_kwargs=minimizer_kwargs, niter=niter, \
            #     niter_success=niter_success,take_step=take_step,callback=self.print_fun)
        else:
            ret = spo.basinhopping(slef.fcn, x0, T=1.0, minimizer_kwargs=minimizer_kwargs, niter=niter, \
                        niter_success=niter_success,take_step=take_step,callback=None)
        print('finish once:\n', ret)
        res = (ret.x, ret.fun)
        return res

    # @classmethod
    def fcn(self, x, *args):
        """
        args=(timestamp,WKBAR,), see minimizer_kwargs in method 'optimizer()'
        x: optimizing parameter, might be a vector.
        MEMO:
            data looks like: [more history bars][hs bars][wk bars][current bar][...]
        """
        timestamp, wkbar = args[0], args[1]
        # NOTE THE DEFAULT VALUE, need reconsider?
        hsbar = int(1000*np.abs(x)/np.ceil(np.abs(x))) if np.abs(x)>0.001 else 100

        slice_df = self.select_slice(timestamp, wkbar=wkbar, hsbar=hsbar)

        if len(slice_df)>1 and len(slice_df.symbol.unique())>4:
            # add signals for the dataframe slice.
            slice_has_signals = Strg_rsv5min.generateSignal(slice_df, interval=hsbar)
            # 只对working bars 做计算pnl操作
            wkbars_has_signals = slice_has_signals.groupby('symbol', as_index=False).apply(lambda dd: dd.iloc[-wkbar:,:])
            wkbars_has_signals = wkbars_has_signals.reset_index(drop=True)
            # print(wkbars_has_signals) # SUCCESSS

            # calculate return of every bar for the portfolio.
            wkbars_pnl = add_pos_cal_pnl(wkbars_has_signals)
            # print('wkbars_pnl=%s' %wkbars_pnl)

            # calculate pnl for this portfolio, and then sharpe.
            wkbars_sharpe = np.nanmean(wkbars_pnl) / np.nanstd(wkbars_pnl) if np.nanstd(wkbars_pnl) != 0. else -100.

            # 添加惩罚项

        else:
            wkbars_sharpe = -100.

        return -wkbars_sharpe

    @classmethod
    def print_fun(cls, x, f, accepted):
        print("x=%.3f, at minmm %.4f accpt %d" % (x,f, int(accepted)))

    # @classmethod
    def select_slice(self, timestamp, **kwargs):
        '''
        bar_loc: the most accurate description for locating this bar,
                 here for 5min-bar, it could be 201101040905.
        '''
        wkbar = kwargs.get('wkbar', 1)
        hsbar = kwargs.get('hsbar', 2)

        # reset_index()
        slice_df = self.DataBase.groupby('symbol').apply(self.extract_bars_by_timestamp, timestamp, wkbar=wkbar, hsbar=hsbar)
        slice_df = slice_df.reset_index(drop=True)
        return slice_df

    @staticmethod
    def extract_bars_by_timestamp(df, timestamp, **kwargs):
        '''for a given symbol'''
        wkbar = kwargs.get('wkbar')
        hsbar = kwargs.get('hsbar')

        df = df.reset_index(drop=True)

        idx = df.index[df['timeStamp']==timestamp].tolist()
        if len(idx) != 0:
            idx = idx[0]
        else:
            # cannot find given timestamp
            return None

        # NOTE: current bar is the last bar.
        start, end = idx - (hsbar+wkbar)+1, idx+1

        # number of total bars = hsbar + wkbar-1
        return df.iloc[start+1:end, :] if (start >= 0  and end <=len(df)) else None

    # unfinished...
    def extract_bars_by_timestamp_tradeDate():
        pass

    def run_optmiz(self, *args, **kwargs):

        t0 = time.time()

        # 0. get DataBase and timeStampList for optimization.
        defaultFile = './processed_5mindata.h5'
        defaultKey = 'preprocessed_5min'
        h5dataPath = kwargs.get('dataPath', defaultFile)

        self.read_data(h5dataPath, defaultKey)

        # 1. --- SET INITIALIZING PARAMETERS ---
        WKBAR = self.wkbar # 1*12*8*5
        # 注意： WKBAR可以设置为序列和timeStampList相同长度，
        #       对应每个时间戳观察窗口bar数不一样多，应该以自然时间相同为准

        # holding time for optimized x .
        HOLDBAR = self.holdbar if self.holdbar>1 else 1

        # recording of final results
        xop_df = pd.DataFrame(np.nan, index=self.timeStampList, columns=['xop','sh_op'])

        # 2. optimize for every timestamp in timeStampList
        n_cpus = kwargs.get('n_cpus', 2)
        print('multiprocess with %s cpus.'% n_cpus)
        p = multiprocessing.Pool(n_cpus);
        res = [];    cnt = 0
        for timestamp in self.timeStampList:
            # initial guess, slight random setting
            x0 = np.random.uniform(self.xmin, self.xmax)
            print('cnt=%s,' % cnt, ' timestamp = %s .' % timestamp, 'x0=%s .'%x0)
            res.append(p.apply_async(self.optimizer, args=(timestamp, x0, WKBAR)))
            print(str(cnt) + ' processor started !')
            cnt += 1
        print ('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print ('All subprocesses done.')

        # merge all signals
        for j,timestamp in enumerate(self.timeStampList):
            x_opt, sh_opt = res[j].get()
            print(x_opt, sh_opt)
            xop_df.loc[timestamp,'xop'] = x_opt if sh_opt!=100. else np.nan
            xop_df.loc[timestamp, 'sh_op'] = -sh_opt if sh_opt!=100. else np.nan

        # save results in local folder
        if kwargs.get('csv',True):
            csvSuffix = kwargs.get('suffix','')
            folderName = './xopts/'
            # save results in local folder
            if not os.path.exists(folderName):
                os.makedirs(folderName)

            fileName = 'x_sh_op'+ '_xmin='+str(self.xmin) + 'max='+str(self.xmax)+\
                        '_wk='+str(self.wkbar)+'hld='+str(self.holdbar)

            xop_df.to_csv(folderName + fileName + csvSuffix +'.csv')

        print('Cost time:',time.time()-t0)

# set bounds for x.
class MyBounds(object):
    def __init__(self, xmin=0.1, xmax=1.):
         self.xmax = xmax
         self.xmin = xmin
    def __call__(self, **kwargs):
     x = kwargs["x_new"]
     tmax = bool(np.abs(x) <= self.xmax)
     tmin = bool(np.abs(x) >= self.xmin)
     return tmax and tmin

class RandomDisplacementBounds(object):
    """random displacement with bounds"""
    def __init__(self, xmin, xmax, stepsize=0.5):
        # test 0.5 is good enough for the random number uniform.
        # do not change stepsize if no any good idea.
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds"""
        while True:
            xnew = x + np.random.uniform(-self.stepsize, self.stepsize, 1)
            # xnew = np.random.random(1)
            if (xnew <= self.xmax) and (xnew >=self.xmin):
                break
        return xnew

# 计算指标分数
def add_pos_cal_pnl(df):
    # 0. get positions for every symbol
    position_all_symbol = df.groupby('symbol', as_index=False).apply(signal2Position)
    # reset the index
    position_all_symbol = position_all_symbol.reset_index(drop=True)
    df['position'] = position_all_symbol

    # 1.calculate PnL for every symbol with known position.
    df = df.groupby('symbol', as_index=False).apply(calpnl).reset_index(drop=True)

    # 2. sum on the cross-section
    pnl_ts = df[['Rt']].groupby(df['timeStamp'], as_index=False).sum()

    return pnl_ts

def calpnl(df):
    '''
    for a given symbol, add one column
    '''
    df['Rt'] = (df['prcClose']-df['prcClose'].shift(1))/df['prcClose'].shift(1)
    df['Rt'] = df['Rt']*df['position']
    return df

def signal2Position(signals, init_pos=0, mechanism='LS',mask=('Indicator_open','Indicator_cutlong', 'Indicator_cutshort')):
    '''
    signals: a DataFrame contains at least Index,
         signals[['Indicator_open','Indicator_cutlong', 'Indicator_cutshort']]
    '''
    # inputs are : Indocator_open, cutlong and cutshort
    # ------------------------------
    pos = init_pos
    # signals[['Indicator_open','Indicator_cutlong', 'Indicator_cutshort']]
    Ind_open, cut_long, cut_short = mask

    position = pd.Series(np.nan, index=signals.index)
    # Indicator_open = pd.Series(np.nan, index=signals.index)
    # if mechanism=='LS':
    for id, sig in signals.iterrows():
        open = sig[Ind_open]
        if pos == 0:
            pos = pos + open
        else:
            # long position with short signal
            if pos>0 and sig[cut_long]:
                pos = pos - sig[cut_long]
            # short position with long signal
            elif pos<0 and sig[cut_short]:
                pos = pos + sig[cut_short]
        position[id] = pos

    position.fillna(method='pad', inplace=True)
    return position


# 假定多次并行的实验的结果都输出在文件夹 './xopts/'
def get_XOPT(xopPath):
    fileList = os.listdir(xopPath)
    fileList = [file for file in fileList if file.endswith('.csv')]
    print(fileList)
    for i,file in enumerate(fileList):
        print(i,file)
        dd=pd.read_csv(os.path.join(xopPath, file), index_col=0)
        dd = dd.fillna(method='pad')
        ds = dd[np.isnan(dd['xop'])==False]
        ds['xop'] = (ds['xop']*1000).apply(lambda x: np.nan if np.isnan(x) else np.int(x))
        ds.rename(columns= {'xop':'xop'+str(i), 'sh_op':'sh'+str(i)}, inplace=True)

        if i==0:
            df = ds
        else:
            df = pd.concat([df, ds], axis=1)
    df['xop'] = df.apply(vote, axis=1)
    # save to file
    df.to_csv(os.path.join('./', 'XOPPO.csv'))

def vote(df):
    import operator
    size = int(len(df)/2)
    col0 = ['xop'+str(i) for i in range(size)]
    col1 = ['sh'+str(i) for i in range(size)]
    x = df[col0].values
    sh = df[col1].values
    L = sorted(zip(x,sh), key=operator.itemgetter(1),reverse=True)

    ret = np.array([i for (i,j) in L])[0:2]
    return int(ret.mean())




if __name__ == '__main__':

    holdbar = 300
    wkbar = 1*12*8*5
    myopt = optimizer(holdbar, wkbar=wkbar, xmin=0.3, xmax=0.9)
    dataPath='../dataBase/processed_5mindata.h5'

    # 每个优化运行多次，以避免随机数的误差
    myopt.run_optmiz(dataPath=dataPath, n_cpus=10, suffix='0')
