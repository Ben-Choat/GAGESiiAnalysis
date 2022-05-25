# %% Intro

# Defining a class to hold and process methods associated with 
# 1. clustering methods
# 2. Feature selection and regression
# 3. Other methods such as standardization


# %% Import libraries

import numpy as np # matrix operations/data manipulation
from sklearn.preprocessing import StandardScaler # standardizing data
from sklearn.preprocessing import MinMaxScaler # normalizing data (0 to 1)
from sklearn.cluster import KMeans # kmeans
from sklearn.metrics import silhouette_samples
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt # plotting
from matplotlib import cm
import plotnine as p9
# from mpl_toolkits.mplot3d import Axes3D # plot 3d plot
# import plotly as ply # interactive plots
# import plotly.io as pio5 # easier interactive plots
# import plotly.express as px # easier interactive plots
# import seaborn as sns # easier interactive plots
import pandas as pd # data wrangling
# import umap # umap
# import hdbscan # hdbscan
# from mlxtend.plotting import heatmap # plotting heatmap


# %% load data

# water yield directory
dir_WY = 'D:/DataWorking/USGS_discharge/train_val_test'
# explantory var (and other data) directory
dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out'

# Annual Water yield
df_train_anWY = pd.read_csv(
    f'{dir_WY}/yrs_98_12/annual_WY/Ann_WY_train.csv',
    dtype={"site_no":"string"}
    )

# mean annual water yield
df_train_mnanWY= df_train_anWY.groupby(
    'site_no', as_index=False
).mean().drop(columns = ["yr"])

# GAGESii explanatory vars
df_train_expl = pd.read_csv(
    f'{dir_expl}/ExplVars_Model_In/All_ExplVars_Train_Interp_98_12.csv',
    dtype = {'STAID': 'string'}
)

# mean GAGESii explanatory vars
df_train_mnexpl = df_train_expl.groupby(
    'STAID', as_index = False
).mean().drop(columns = ['year'])

# %% define clustering class

# Define a clusterer class that allows 

class Clusterer:
    """Clustering class.


    Parameters
    -------------
    Place parameters and descriptions here:
    
    clust_vars: numeric pandas dataframe
        variables used for clustering


    Attributes
    -------------
    Place attirbues here (These are values or characteristics/attributes
        that are generated within the class)
    e.g., w_ : 1d-array
        Weights after fitting
    
    out_: numeric
        a dummy output variable

    distortions_: numeric
        SSE measurement from k-means clustering

    dist_plot_: plot
        distortion plot from k-means distortion vs number of clusters

    """

    def __init__(self, clust_vars):
        self.clust_vars = clust_vars

    def stand_norm(self, method = 'standardize', not_tr = []):
        """
        Standardize or normalize clustering variables

        Parameters
        --------------
        method: string equal to 'standardize' or 'normalize'
        not_tr: list of column names not to transform
        
        Attributes
        --------------
        clust_vars_tr: transformed clustering variables
        """

        if method == 'standardize':
            stdsc = StandardScaler()
            clust_vars_tr = pd.DataFrame(
                stdsc.fit_transform(self.clust_vars)
            )
            # give columns names to transformed data
            clust_vars_tr.columns = self.clust_vars.columns
            # replace transformed vars with untransformed for specified columns
            clust_vars_tr[not_tr] = self.clust_vars[not_tr]

            self.clust_vars_tr = clust_vars_tr

        if method == 'normalize':
            normsc = MinMaxScaler()
            self.clust_vars_tr = normsc.fit_transform(self.clust_vars)

    
    def clust(self, 
                method = 'kmeans', 
                ki = 2, kf = 20, 
                plot_mean_sil = True, 
                plot_distortion = True,
                kmed_method = 'alternate'):
        """
        Parameters
        -------------
        method: string
                either 'kmeans' or 'kmedoids' (or maybe add hdbscan here too?)
        ki: integer
                initial (or smallest) number of clusters to use
        kf: integer
                final (or largest) number of clusters to use
        plot_mean_sil: Boolean
                Either 'True' or 'False'
        kmed_method: string
                either 'alternate' or 'pam'. 'alternate' is fast but 'pam' 
                is more accurate
        """
        # initialize vectors to hold outputs
        distortions = []
        ks_out = []
        silhouette_mean_scores = []
        silhouette_k_scores = []

        for i in range(ki, kf + 1):
            if method == 'kmeans':
                km = KMeans(n_clusters = i,
                            init = 'k-means++',
                            n_init = 10,
                            max_iter = 300,
                            tol = 1e-04,
                            random_state = 100)

            if method == 'kmedoids':
                km = KMedoids(n_clusters = i,
                            metric = 'manhattan',
                            # method = 'pam',
                            method = kmed_method,
                            init = 'k-medoids++',
                            max_iter = 300,
                            random_state = 100)
            # if clust_vars_tr exists, use them, else use clust_vars directly
            try:
                km_fit = km.fit(self.clust_vars_tr)

                # calc individual silhouette scores
                silhouette_vals = silhouette_samples(
                    self.clust_vars_tr, km_fit.labels_, metric = 'euclidean'
                )                
            except AttributeError:
                km_fit = km.fit(self.clust_vars)

                # calc individual silhouette scores
                silhouette_vals = silhouette_samples(
                    self.clust_vars, km.fit.labels_, metric = 'euclidean'
                )

            # calc silhouette scores
            mean_sil = np.array(silhouette_vals).mean()
            mean_sil_k = []
            for j in np.unique(km_fit.labels_):
                mn_sil = np.array(silhouette_vals[km_fit.labels_ == j]).mean()

                mean_sil_k.append(mn_sil)

            # update vectors
            distortions.append(km.inertia_)
            ks_out.append(km_fit.labels_)
            silhouette_mean_scores.append(mean_sil)
            silhouette_k_scores.append(mean_sil_k)
        
        self.distortions_ = distortions
        self.ks_predicted_ = ks_out
        self.silhouette_mean = silhouette_mean_scores
        self.silhouette_k_mean = silhouette_k_scores

        # if plot_mean_sil = true, plot mean silhouette coef. vs number of clusters
        if plot_mean_sil is True:
            data_in = pd.DataFrame({
                'NumberOfk': range(ki, kf+1),
                'Mean_Sil_Coef': silhouette_mean_scores
            })
            plot = (
                p9.ggplot(data = data_in) +
                p9.geom_point(p9.aes(x = 'NumberOfk', y = 'Mean_Sil_Coef')) +
                p9.xlab('Number of clusters') + p9.ylab('Mean Silhouette Coeficient') +
                p9.ggtitle(f'Mean Silhouette values - ', {method}) +
                p9.scale_x_continuous(breaks = range(ki, kf+1)) +
                p9.theme_minimal()
            )

            print(plot)


        # if plot_distortion = True, plot distortion vs number of clusters
        if plot_distortion is True:
            data_in = pd.DataFrame({
                'NumberOfk': range(ki, kf+1),
                'Distortion': self.distortions_
            })
            plot2 = (
                p9.ggplot(data = data_in) +
                p9.geom_line(p9.aes(x = 'NumberOfk', y = 'Distortion')) +
                p9.geom_point(p9.aes(x = 'NumberOfk', y = 'Distortion')) +
                p9.xlab("Number of clusters") + p9.ylab('Distortion') + 
                p9.ggtitle(f'Elbow Plot - ', {method}) +
                p9.theme_minimal()
            )

            print(plot2)




    def plot_silhouette_vals(self, k = 3, method = 'kmeans', kmed_method = 'alternate'):
        """
        Parameters
        -------------
        k: integer
            number of clusters to use in calculating silhouette coeficients
        method: string
                either 'kmeans' or 'kmedoids' (or maybe add hdbscan here too?)
        kmed_method: string
                either 'alternate' or 'pam'. 'alternate' is fast but 'pam' 
                is more accurate
        """

        if method == 'kmeans':
            km = KMeans(n_clusters = k,
                        init = 'k-means++',
                        n_init = 10,
                        max_iter = 300,
                        tol = 1e-04,
                        random_state = 100)

        if method == 'kmedoids':
            km = KMedoids(n_clusters = k,
                        metric = 'manhattan',
                        # method = 'pam',
                        method = kmed_method,
                        init = 'k-medoids++',
                        max_iter = 300,
                        random_state = 100)
        try:
            cluster_labels = np.unique(self.clust_vars_tr)
            km_fit = km.fit(self.clust_vars_tr)
            # calc individual silhouette scores
            silhouette_vals = silhouette_samples(
                self.clust_vars_tr, km_fit.labels, metric = 'euclidean'
            )
            
        except AttributeError:
            cluster_labels = np.unique(self.clust_vars)
            km_fit = km.fit_predict(self.clust_vars)
            # calc individual silhouette scores
            silhouette_vals = silhouette_samples(
                self.clust_vars, km_fit.labels_, metric = 'euclidean'
            )

        


        # create dataframe for use in plotting
        self.sil_datain = pd.DataFrame({
            'k': pd.Series(km_fit.labels_, dtype = 'category'),
            'Sil_score': silhouette_vals
        }).sort_values(by = ['k', 'Sil_score'])
        self.sil_datain['number'] = range(1, len(silhouette_vals)+1, 1)

        # define tic marks
        tics_in = []
        for i in np.unique(self.sil_datain.k):
            tics_in.append(
                np.array(self.sil_datain.number[self.sil_datain.k == i]).mean()
            )

        # Plot

        plot_out = (
            p9.ggplot(data = self.sil_datain) +
            p9.geom_bar(p9.aes(x ='number', y =  'Sil_score', color = 'k'),
                stat = 'identity') +
            p9.geom_hline(p9.aes(yintercept = np.array(silhouette_vals).mean()),
                linetype = 'dashed', color = 'red') +
            p9.scale_x_continuous(breaks = tics_in, 
                labels = np.unique(self.sil_datain.k)) +
            p9.xlab('Cluster') + p9.ylab('Silhouette Coeficient') +
            p9.coord_flip() +
            p9.theme_minimal()
        )

        print(plot_out)

        
        
        
        #return self.dist_plot_

# %%
# define list of columns not to transform
# these columns are OHE so already either 0 or 1. 
# for distance metrics, use Manhattan which lends itself to capturing 
not_tr_in = ['GEOL_REEDBUSH_DOM_anorthositic', 'GEOL_REEDBUSH_DOM_gneiss',
       'GEOL_REEDBUSH_DOM_granitic', 'GEOL_REEDBUSH_DOM_quarternary',
       'GEOL_REEDBUSH_DOM_sedimentary', 'GEOL_REEDBUSH_DOM_ultramafic',
       'GEOL_REEDBUSH_DOM_volcanic']
test = Clusterer(clust_vars = df_train_mnexpl.drop(columns = ['STAID']))
test.stand_norm(method = 'standardize', not_tr = not_tr_in) # 'normalize'
test.clust(ki = 2, kf = 20, plot_mean_sil = True, plot_distortion = True)
test.clust(
    ki = 2, kf = 20, 
    method = 'kmedoids', 
    plot_mean_sil = True, 
    plot_distortion = True,
    kmed_method = 'alternate')

for i in range(2, 11):
    test.plot_silhouette_vals(k = i)
# %%
