# see my CSDN blog: https://blog.csdn.net/Papageno_Xue/article/details/86669855
# ---------------------- Numpy array rolling window操作  ---------------
# 简单操作如果是pandas有的 就转为pandas再处理比较快
def rolling_window(a, window, axis=0):
    """
    返回2D array的滑窗array的array
    """
    if axis == 0:
        shape = (a.shape[0] - window +1, window, a.shape[-1])
        strides = (a.strides[0],) + a.strides
        a_rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    elif axis==1:
        shape = (a.shape[-1] - window +1,) + (a.shape[0], window) 
        strides = (a.strides[-1],) + a.strides
        a_rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return a_rolling





# 对于pandas DataFrame自带的函数，对于大维度，还是pandas rolling method快
# 小数组的话，此方法快
# 上述的快，都是能十倍的速度比。
def rolling_mean(A, window=None):
    ret = np.full(A.shape, np.nan)
    A_rolling = rolling_window(A, window=window, axis=0)
    Atmp = np.stack(map(lambda x:np.mean(x, axis=0), A_rolling))
    ret[window-1:,:] = Atmp
    return ret

def rolling_nanmean(A, window=None):
    ret = np.full(A.shape, np.nan)
    A_rolling = rolling_window(A, window=window, axis=0)
    Atmp = np.stack(map(lambda x:np.nanmean(x, axis=0), A_rolling))
    ret[window-1:,:] = Atmp
    return ret
