

# %%
import pandas as pd
import numpy as np



# NOTE: check IdentifyBestClusterApproachesUsingTSPredictionMetrics.py
# to test results

# %%

# TrainOrVal = 'train'

dir_work = 'D:/Projects/GAGESii_ANNStuff/Data_Out/Results'

df_ma = pd.read_csv(f'{dir_work}/PerfMetrics_MeanAnnual.csv')

# df_ma = df_ma[df_ma['train_val'] == TrainOrVal]

df_ma['|residuals|'] = df_ma['residuals'].abs()

# df_ma = df_ma[df_ma['clust_method'].str.contains('Anth')]


df_ma['region'] = df_ma['clust_method'].str.cat(\
            df_ma['region'].astype(str), sep = '_')

qnts_in = np.round(np.arange(0.05, 1.0, 0.05), 2)


if 'df_qntls' in globals():
        del(df_qntls)
for q in qnts_in:
        if not 'df_qntls' in globals():
                df_qntls = df_ma.groupby(
                        ['clust_method', 'model', 'train_val', 'time_scale']
                        )['|residuals|'].quantile(q).reset_index()
                df_qntls = df_qntls.rename(columns = {'|residuals|': f'Q{q}'})
        else:
                df_qntls[f'Q{q}'] = df_ma.groupby(
                        ['clust_method', 'model', 'train_val', 'time_scale']
                        ).quantile(q).reset_index()['|residuals|']


subset_cols = df_qntls.columns[df_qntls.columns.str.contains('Q')].values.tolist()

df_qntls['Qmean'] = df_qntls[subset_cols].apply('mean', axis = 1)

df_meanqntl = df_qntls[['clust_method', 'model', 'train_val', 'time_scale', 'Qmean']]













# %%
# drop noise


# df_madn = df_ma[df_ma['region'] != -1]
df_madn = df_ma[~df_ma['region'].str.contains('-1')]

if 'df_qntlsdn' in globals():
        del(df_qntlsdn)
for q in qnts_in:
        if not 'df_qntlsdn' in globals():
                df_qntlsdn = df_madn.groupby(
                        ['clust_method', 'model', 'train_val', 'time_scale']
                        )['|residuals|'].quantile(q).reset_index()
                df_qntlsdn = df_qntlsdn.rename(columns = {'|residuals|': f'Q{q}'})
        else:
                df_qntlsdn[f'Q{q}'] = df_madn.groupby(
                        ['clust_method', 'model', 'train_val', 'time_scale']
                        ).quantile(q).reset_index()['|residuals|']


subset_cols = df_qntlsdn.columns[df_qntlsdn.columns.str.contains('Q')].values.tolist()

df_qntlsdn['Qmean'] = df_qntlsdn[subset_cols].apply('mean', axis = 1)

df_meanqntldn = df_qntlsdn[['clust_method', 'model', 'train_val', 'time_scale', 'Qmean']]

# %%

# read in previous perfrank csv

df_perfRank = pd.read_csv(f'{dir_work}/PerfRankMeanAnnual_ByClustMeth.csv')