# %% Intro

# Defining a class to hold and process methods associated with 
# 1. clustering methods
# 2. Feature selection and regression
# 3. Other methods such as standardization and normalization


# %% Import libraries

from NSE_KGE_timeseries import NSE_KGE_Apply # NSE and KGE applied to many catchments
from sklearn.preprocessing import StandardScaler # standardizing data
from sklearn.preprocessing import MinMaxScaler # normalizing data (0 to 1)
from sklearn.cluster import KMeans # kmeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics import mean_squared_error # returns negative (larger is better)
                                                # squared = True: MSE; squared = False: RMSE
from sklearn.metrics import mean_absolute_error # returns negative (larger is better)
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV # for hyperparameter tuning
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV # for hyperparameter tuning
from sklearn.model_selection import RepeatedKFold # for repeated k-fold validation
# from sklearn.model_selection import cross_val_score # only returns a single score (e.g., MAE)
from sklearn.model_selection import cross_validate # can be used to return multiple scores 
# from sklearn.feature_selection import RFE
# from sklearn.feature_selection import RFECV
# from sklearn.pipeline import Pipeline
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA # dimension reduction
# import statsmodels.formula.api as smf # for ols, aic, bic, etc.
# import statsmodels.api as sm # for ols, aic, bic, etc.
# from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs # plot sfs results

import xgboost as xgb
import skfuzzy as fuzz # for fuzzy c-means

import umap # for umap training and projection (dimension reduction)
import hdbscan # for hdbscan clustering
# from mpl_toolkits.mplot3d import Axes3D # plot 3d plot
# import plotly as ply # interactive plots
# import plotly.io as pio5 # easier interactive plots
import plotly.express as px # easier interactive plots
import matplotlib.pyplot as plt # plotting
# from matplotlib import cm
import plotnine as p9
import seaborn as sns # easier interactive plots

import numpy as np # matrix operations/data manipulation
import pandas as pd # data wrangling
import time
from Regression_PerformanceMetrics_Functs import *



# %% define clustering class

# Define a clusterer class that allows:
# standardization and normalization of data
# k-means and k-medoids clustering
# plot_silhouette_vals
# HDBSCAN
# UMAP

class Clusterer:
    """Clustering class.


    Parameters
    -------------
    Place parameters and descriptions here:
    
    clust_vars: numeric pandas dataframe
        variables used for clustering

    id_vars: numeric or character array type object
        e.g., numpy array or pandas.series
        unique identifiers for each data point (e.g., gage number)

    color_vars: string
        grouping variable (e.g., aggecoregion) to use
        when coloring plots

    Attributes
    -------------
    Place attirbues here (These are values or characteristics/attributes
        that are generated within the class)
    e.g., w_ : 1d-array
        Weights after fitting

    distortions_: numeric
        SSE measurement from k-means clustering

    dist_plot_: plot
        distortion plot from k-means distortion vs number of clusters

    """

    def __init__(self, clust_vars, id_vars):
        self.clust_vars = clust_vars
        self.id_vars = id_vars
        # self.color_vars = color_vars

    def stand_norm(self, method = 'standardize', not_tr = []):
        """
        Standardize or normalize clustering variables

        Parameters
        --------------
        method: string equal to 'standardize' or 'normalize'
        not_tr: string or string vector
            list of column names not to transform
        
        Attributes
        --------------
        clust_vars_tr_: transformed clustering variables
        scaler_: the transform variables stored for application to other data
            (e.g., mean and standard deviation of each column for standardization)
        """

        if method == 'standardize':
            stdsc = StandardScaler()
            self.scaler_ = stdsc.fit(self.clust_vars)
            clust_vars_tr_ = pd.DataFrame(
                self.scaler_.transform(self.clust_vars)
            )
            # give columns names to transformed data
            clust_vars_tr_.columns = self.clust_vars.columns
            # replace transformed vars with untransformed for specified columns
            clust_vars_tr_[not_tr] = self.clust_vars[not_tr]

            self.clust_vars_tr_ = clust_vars_tr_

        if method == 'normalize':
            normsc = MinMaxScaler()
            self.scaler_ = normsc.fit(self.clust_vars)
            clust_vars_tr_ = pd.DataFrame(
                self.scaler_.transform(self.clust_vars)
            )
            # give columns names to transformed data
            clust_vars_tr_.columns = self.clust_vars.columns
            # replace transformed vars with untransformed for specified columns
            clust_vars_tr_[not_tr] = self.clust_vars[not_tr]

            self.clust_vars_tr_ = clust_vars_tr_

    
    def k_clust(self,
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

        Attributes
        -------------
        distortions_: distortion scores used in elbow plot
        df_ks_predicted_: pandas DataFrame
            which group k, in which each catchment was placed
        silhouette_mean_: mean silhouette score for each number of
            k groups between ki and kf
        silhouette_k_mean_: individual silhoueete scores for each
            group, k, for each number of k groups between ki and kf
        self.km_clusterer_: k-means cluster object that can be used to fit new data to

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
            # if clust_vars_tr_ exists, use them, else use clust_vars directly
            try:
                km_fit = km.fit(self.clust_vars_tr_)

                # calc individual silhouette scores
                silhouette_vals = silhouette_samples(
                    self.clust_vars_tr_, km_fit.labels_, metric = 'euclidean'
                )                
            except AttributeError:
                km_fit = km.fit(self.clust_vars)

                # calc individual silhouette scores
                silhouette_vals = silhouette_samples(
                    self.clust_vars, km_fit.labels_, metric = 'euclidean'
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
        
        # assign outputs to object
        self.distortions_ = distortions
        self.df_ks_predicted_ = pd.DataFrame(ks_out).T
        self.silhouette_mean_ = silhouette_mean_scores
        self.silhouette_k_mean_ = silhouette_k_scores

        # assign km object to object
        self.km_clusterer_ = km_fit

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
                p9.ggtitle(f'Mean Silhouette values - {method}') +
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
                p9.ggtitle(f'Elbow Plot - {method}') +
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

        Attributes
        --------------
        sil_datain_: pandas dataframe holding k labels, 
            silhouette scores, and number of stations
            Used to create silhouette plot for a single number
            of k groups

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
            km_fit = km.fit(self.clust_vars_tr_)
            # calc individual silhouette scores
            silhouette_vals = silhouette_samples(
                self.clust_vars_tr_, km_fit.labels_, metric = 'euclidean'
            )
            
        except:
            km_fit = km.fit(self.clust_vars)
            # calc individual silhouette scores
            silhouette_vals = silhouette_samples(
                self.clust_vars, km_fit.labels_, metric = 'euclidean'
            )


        # create dataframe for use in plotting
        self.sil_datain_ = pd.DataFrame({
            'k': pd.Series(km_fit.labels_, dtype = 'category'),
            'Sil_score': silhouette_vals
        }).sort_values(by = ['k', 'Sil_score'])
        self.sil_datain_['number'] = range(1, len(silhouette_vals)+1, 1)

        # define tic marks
        tics_in = []
        for i in np.unique(self.sil_datain_.k):
            tics_in.append(
                np.array(self.sil_datain_.number[self.sil_datain_.k == i]).mean()
            )

        # Plot

        plot_out = (
            p9.ggplot(data = self.sil_datain_) +
            p9.geom_bar(p9.aes(x = 'number', y = 'Sil_score', color = 'k'),
                stat = 'identity') +
            p9.geom_hline(p9.aes(yintercept = np.array(silhouette_vals).mean()),
                linetype = 'dashed', color = 'red') +
            p9.scale_x_continuous(breaks = tics_in, 
                labels = np.unique(self.sil_datain_.k)) +
            p9.xlab('Cluster') + p9.ylab('Silhouette Coeficient') +
            p9.coord_flip() +
            p9.theme_minimal()
        )

        print(plot_out)


    def hdbscanner(self,
        min_clustsize = 20,
        min_sample = None,
        gm_spantree = True,
        metric_in = 'euclidean',
        clust_seleps = 0,
        clustsel_meth = 'leaf'):

        """
        Parameters
        -------------
        min_clustsize: integer
            Primary parameter to effect the resulting clustering.
            The minimum number of samples in a cluster
        min_sample: integer
            Provides measure of how conservative you want your clustering to be.
            Larger the value, the more conservative the clustering - more points
            will be declared as noise and clusters will be restricted to progressively
            more dense areas.
            If not specified (commented out), then defaults to same value as min_clustize
        metric_in: string 'euclidean' or 'manhattan'
        gm_spantree: boolean
            Whether to generate the minimum spanning tree with regard to mutual reachability
            distance for later analysis
        clust_seleps: float between 0 and 1
            default value is 0. E.g., if set to 0.5 then DBSCAN clusters will be extracted
            for epsilon = 0.5, and clusters that emerged at distances greater than 0.5 will
            be untouched (helps allow small clusters w/o splitting large clusters)
        clustsel_meth: string
            'leaf' or 'eom' - Excess of Mass
            which cluster method to use
            eom has greater tendency to pick small number of clusters 
            leaf tends towards smaller homogeneous clusters

        Attributes
        --------------
        hd_clusterer_: the clusterer object
            this object holds outputs produced and can be used for further clustering
        
        hd_out_: dataframe with:
            cluster labels
            probabilities (confidence that each point is in the most appropriate cluster)
            outlier scores (the higher the score, the more likely the point
                is to be an outlier)

        """    
        min_sample = min_clustsize if min_sample is None else min_sample

        hd_clusterer = hdbscan.HDBSCAN(min_cluster_size = min_clustsize,
                                min_samples = min_sample, 
                                gen_min_span_tree = gm_spantree,
                                metric = metric_in,
                                cluster_selection_epsilon = clust_seleps,
                                cluster_selection_method = clustsel_meth,
                                prediction_data = True)

        # cluster data
        try:
            # hd_clusterer.fit(self.clust_vars_tr_)
            data_in = self.clust_vars_tr_
        except:
            # hd_clusterer.fit(self.clust_vars)
            data_in = self.clust_vars
        
        hd_clusterer.fit(data_in)

        # plot condensed tree
        plt.figure()
        plot2 = hd_clusterer.condensed_tree_.plot(select_clusters = True,
                                       selection_palette = sns.color_palette())
        print(plot2)


        # Print number of catchments in each cluster
        clst_lbls = np.unique(hd_clusterer.labels_)

        for i in clst_lbls:
            temp_sum = sum(hd_clusterer.labels_ == i)
            print(f'cluster {i}: sum(clst_lbls == {temp_sum})')

        # assign clusterer object to self
        self.hd_clusterer_ = hd_clusterer

        # create dataframe with:
            # cluster labels
            # probabilities (confidence that each point is in the most appropriate cluster)
            # outlier scores (the higher the score, the more likely the point
            # is to be an outlier)

        self.hd_out_ = pd.DataFrame({
            'ID': self.id_vars,
            'labels': hd_clusterer.labels_,
            'probabilities': hd_clusterer.probabilities_,
            'outlier_scores': hd_clusterer.outlier_scores_
        })
    
        # Other potential plots of interest

        # plot the spanning tree if gm_spantree = True
        if gm_spantree == True:
            plt.figure()
            plot1 = hd_clusterer.minimum_spanning_tree_.plot(edge_cmap = 'viridis',
                                                edge_alpha = 0.6,
                                                node_size = 1,
                                                edge_linewidth = 0.1)
            print(plot1)
        # plot1.show()

        # plot cluster heirarchy
        #clusterer.single_linkage_tree_.plot(cmap = 'viridis',
        #                                    colorbar = True)
        # condensed tree
        # clusterer.condensed_tree_.plot()
        # look at outliers
        # sns.distplot(clusterer.outlier_scores_,
        #             rug = True)

        # get 90th quantile of outlier scores
        # pd.Series(clusterer.outlier_scores_).quantile(0.9)


        # histogram of cluster probabilities (how strongly staid fits into cluster)
        # plot3 = sns.distplot(hd_clusterer.probabilities_,
        #              rug = True)

        # print(plot3)

    def hd_predictor(self,
        points_to_predict,
        id_vars):
        """
        Parameters
        -------------
        points_to_predict: pd.DataFrame type object of explantory variables
            to be predicted on
        id_vars: variables to identify each catchment (e.g., STAID)

        Attributes
        ------------
        df_hd_pred_: pd.DataFrame with id_vars, predicted cluster, and 
            measure of confidence in that prediction
        """

        # assign predictor function to self
        clusters, pred_strength = hdbscan.approximate_predict(
            clusterer = self.hd_clusterer_,
            points_to_predict = points_to_predict
        )

        # define output dataframe with station IDs
        self.df_hd_pred_ = pd.DataFrame(
            {'ID': id_vars,
            'pred_cluster': clusters,
            'conf_pred': pred_strength}
        )




    def fuzzy_cm_clusterer(self,
        c_i = 3, c_f = 20,
        m_in = 2,
        error_in = 0.005,
        maxiter_in = 1000,
        init_in = None,
        seed_in = 100):

        """
        This function returns a plot of the fuzzy partition coefficient for clusters
        ranging from c_i to c_f as a measure of how cleanly the data was described
        by a model including that number of clusters. Values of fpc range from 0 to 1
        where 1 is the best and 0 the worst.

        Parameters
        ------------
        c_i: integer
            initial (or smallest) number of clusters to consider
        c_f: integer
            final (or largest) number of clusters to consider
        m_in: float
            array exponentiation applied to the membership function u_old at each iteration
            U_new = u_old ** m
        error_in: float
            stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error
        maxiter: int
            Maximum number of iterations allowed
        init_in:  2d array, size (S, N)
            initial fuzzy c-partitioned matrix. If none provided, algorithm is 
            randomly initilized
        seed_in: integer
            sets random seed of init. Only applies when init_in is set to None

        Attributes
        -------------
        fcm_centers: array of cluster centers from training
            these centers are used in the cmeans_predict function
        fcm_df_membership: pandas DataFrame
            contains columns for each cluster and how well each datapoint fits into that cluster
            and an additional column with the cluster label of the best fitting cluster
        # fcm_u: 2d array, size (S, N)
        #     Final fuzzy c-partitioned matrix
        # fcm_fpc: float
        #      final fuzzy parition coefficient
        commented lines just below here are the attributes of the skfuzzy cluster.cmeans
        command, but not necessarily attributes of theself.fuzzy_cm_clusterer object
        # cntr: 2d array, size (S, c)
        #     cluster centers. Data for each center along each feature provided
        #     for every cluster (of the c requested clusters)
        # u: 2d array, size (S, N)
        #     Final fuzzy c-partitioned matrix
        # u0: 2d array, (S, N)
        #     Initial guess at fuzzy c-partitioned matrix (either provided init or
        #     random guess used if init was provided).
        # d: 2d array, (S, N)
        #     Final Euclidian distance matrix
        # jm: 1d array, length P
        #     Objective funciton history
        # p: init
        #     Number of iterations run
        # fpc: float
        #     final fuzzy parition coefficient
        """

        # define data to work with
        # cluster data
        try:
            # hd_clusterer.fit(self.clust_vars_tr_)
            data_in = self.clust_vars_tr_
        except:
            # hd_clusterer.fit(self.clust_vars)
            data_in = self.clust_vars

        # initialize empty arrays to hold fpc values, the number of clusters
        fpcs = []
        cs_out = []

        # loop through the number of clusters specified (from c_i to c_f)
        for c_in in range(c_i, c_f+1, 1):
            # cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmean(
            cntrs, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
                data = data_in.T, 
                c = c_in, 
                m = m_in, 
                error = error_in, 
                maxiter = maxiter_in, 
                init = init_in, 
                seed = seed_in
            )

            fpcs.append(fpc)
            cs_out.append(c_in)

        self.fcm_centers = cntrs
        # write dataframe with probability of membership and the cluster
        # with highest probability
        df_temp = pd.DataFrame(u.T)
        df_temp['Cluster'] = np.argmax(u, axis = 0)
        self.fcm_df_membership = df_temp

        # define pd.DataFrame for plotting
        df_plot = pd.DataFrame({
            'Clusters': cs_out,
            'FPC': fpcs
        })

        # plot
        plot = (
                p9.ggplot(data = df_plot) +
                p9.geom_point(p9.aes(x = 'Clusters', y = 'FPC')) +
                p9.ggtitle('FPC vs # of Clusters')
        )

        print(plot)

    
    def fuzzy_cm_predictor(self,
        data_pred,
        # cntrs_trained = self.fcm_centers,
        m_in = 2,
        error_in = 0.005,
        maxiter_in = 1000,
        init_in = None,
        seed_in = 100):

        """
        This function returns a plot of the fuzzy partition coefficient for clusters
        ranging from c_i to c_f as a measure of how cleanly the data was described
        by a model including that number of clusters. Values of fpc range from 0 to 1
        where 1 is the best and 0 the worst.

        Parameters
        ------------
        data_pred: pd.DataFrame 
            data to predict clusters for
            Must have same columns as data which was used for training clusters
        m_in: float
            array exponentiation applied to the membership function u_old at each iteration
            U_new = u_old ** m
        error_in: float
            stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error
        maxiter: int
            Maximum number of iterations allowed
        init_in:  2d array, size (S, N)
            initial fuzzy c-partitioned matrix. If none provided, algorithm is 
            randomly initilized
        seed_in: integer
            sets random seed of init. Only applies when init_in is set to None

        Attributes
        -------------
        fcm_pred_fpc: float
            fuzzy patition coefficient
        fcm_pred_u: 2d array, size (S, N)
             Final fuzzy c-partitioned matrix
        """

        # u, u0, d, jm, p, fpc
        u, _, _, _, _, fpc = fuzz.cmeans_predict(
                test_data = data_pred.T, 
                cntr_trained = self.fcm_centers, 
                m = m_in, 
                error = error_in, 
                maxiter = maxiter_in, 
                init = init_in, 
                seed = seed_in
            )

        self.fcm_pred_u = u
        self.fcm_pred_fpc = fpc

        # write dataframe with probability of membership and the cluster
        # with highest probability
        df_temp = pd.DataFrame(u.T)
        df_temp['Cluster'] = np.argmax(u, axis = 0)
        self.fcm_pred_df_membership = df_temp



        
    def umap_reducer(self, 
        nn = 3, 
        mind = 0.1,
        sprd = 1,
        nc = 3,
        color_in = 'blue',
        n_jobs_in = -1):
        
        """
        Parameters
        -------------
        nn: integer: 2 to 100 is reasonable
            n_neighbors: How many points to include in nearest neighbor
            controls how UMAP balances local vs global structure.
            As nn increases focus shifts from local to global structure
            
        mind: float: 
            min_dist: minimum distance allowed to be in the low dimensional representation
                values between 0 and 0.99
            controls how tightly UMAP is allowed to pack points together
            low values of min_dist result in clumpier embeddings which can be useful
            if interested in clustering. Larger values will prevent UMAP from packing
            points together and will focus on the preservation of the broad 
            topological structure.
            
        sprd: integer;
            effective scale of embedding points. In combination with min_dist (mind) this
            determines how clustered/clumped the embedded points are.
            Generally leave set to 1 and use min_dist for tuning.

        nc: integer; 2 to 100 is reasonable
            n_components: number of dimensions (components) to reduce to

        n_jobs_in: -1 or a positive integer
            specifies number of cores to distribute job to
            value of -1 uses all available resources
        
        Attributes
        ------------
        df_embedding_:
            pd.DataFrame holding color variable (e.g., AggEcoregion), STAID, the point size
            and the UMAP embeddings of the training data

        umap_embedding_:
            UMAP object that can be used to fit new data to the trained UMAP space
        """    
              
        # define umap reducer
        reducer = umap.UMAP(n_neighbors = nn, # 2 to 100
                        min_dist = mind, # 
                        spread = sprd,
                        n_components = nc, # 2 to 100
                        random_state = 100,
                        n_epochs = 200, # 200 for large datasets; 500 small
                        n_jobs = n_jobs_in) 

        # working input data
        try:
            data_in = self.clust_vars_tr_

        except: 
            data_in = self.clust_vars

        # Scale data to z-scores (number of stdevs from mean)
        # data_in_scld = StandardScaler().fit_transform(data_in)

        # train reducer
        self.umap_embedding_ = reducer.fit(data_in)
        umap_embedding_transform = self.umap_embedding_.transform(data_in)

        # save embeddings as dataframe
        self.df_embedding_ = pd.DataFrame.from_dict({
            'Color': color_in,
            'ColSize' : np.repeat(0.1, len(umap_embedding_transform[:, 0]))
            }).reset_index().drop(columns = ["index"])

        self.df_embedding_['STAID'] = self.id_vars
        for i in range(0, nc):
            self.df_embedding_[f'Emb{i}'] = umap_embedding_transform[:, i]

        # Plot embedding

        fig = px.scatter_3d(self.df_embedding_,
                            x = 'Emb0',
                            y = 'Emb1',
                            z = 'Emb2',
                            color = 'Color',
                            size = 'ColSize',
                            size_max = 10,
                            title = f"nn: {nn}",
                            custom_data = ["STAID"]
                            )
        # Edit so hover shows station id
        fig.update_traces(
        hovertemplate = "<br>".join([
            "STAID: %{customdata[0]}"
        ])
        )

        print(fig.show())
        return f'Embedding shape: {umap_embedding_transform.shape}'


    def pca_reducer(self, 
        nc = None,
        color_in = 'blue',
        plot_out = True):
        
        """
        Parameters
        -------------
        nc: integer or 'None'
            if set to integer, then that is the number of components to keep (< number of features)
            if set to None, then keeps all components

        color_in: pandas series or string specifying color
            if string (e.g., 'blue') then that is the color all plotted points will be
            if pandas series (e.g., df_train_ID['AggEcoregion']), 
                then that variable will be used for coloring the plotted points
            
        plot_out: boolean,
            if True, then plot 3d components (NOTE: not set up to work with less than 3 dimensions/components)
            
        
        Attributes
        ------------
        pca_fit_: sklearn PCA object with associated attributes and methods
            The below three attributes are actually attributes of pca_fit_

        df_pca_components_: pandas DataFrame of shape (n_components, n_features)
            principal axes in features space. represents directions of maximum variance in the data.
            components are sorted by explained variance

        df_pca_embedding_: pandas DataFrame
            transformed training data

        df_pca_var_: pandas DataFrame of shape (n_components, 3 - [component, var_expl, ratio_var_expl])
            The amount of variance and the ratio of variance explained by each of the selected components.
            Columns are:
                'Component' which specifies which PCA component the row refers to
                'var_expl' which is the total variance explained by that component
                'ratio_var_expl' which is the ratio of total variance explained by that component
        pca95_: integer
            numer of componenets required to capture 95% of variance in explanatory vars
        
        """    
              
        # define PCA reducer
        reducer = PCA(n_components = nc)

        # working input data
        try:
            data_in = self.clust_vars_tr_

        except: 
            data_in = self.clust_vars

        # train and transform reducer
        self.pca_fit_ = reducer.fit(data_in)
        df_pca_embedding = pd.DataFrame(self.pca_fit_.transform(data_in))
        # rename columns of embedding df
        df_pca_embedding.columns = [f'Comp{i}' for i in np.arange(0, df_pca_embedding.shape[1], 1)]
        # add stations to embedding df
        df_pca_embedding['ID'] = self.id_vars
        self.df_pca_embedding_ = df_pca_embedding
        
        # output attributes
        # pca components saved as pandas DataFrame
        df_pca_components_ = pd.DataFrame(self.pca_fit_.components_)
        df_pca_components_.columns = [f'Comp{i}' for i in np.arange(0, df_pca_components_.shape[1], 1)]
        # add station IDs to df
        self.df_pca_components_ = df_pca_components_

        # variance explained
        df_pca_var = pd.DataFrame({'component': np.arange(0, df_pca_components_.shape[1], 1)})
        df_pca_var['var_expl'] = pd.Series(self.pca_fit_.explained_variance_)
        df_pca_var['ratio_var_expl'] = pd.Series(self.pca_fit_.explained_variance_ratio_)
        df_pca_var['Cum_Var'] = pd.Series(np.cumsum(df_pca_var['ratio_var_expl']))
        # define variable holding number of components when 50%, 75%, 90%, and 95% 
        # variance is explained
        var_50 = np.min(np.where(df_pca_var['Cum_Var'] > 0.5))
        var_75 = np.min(np.where(df_pca_var['Cum_Var'] > 0.75))
        var_90 = np.min(np.where(df_pca_var['Cum_Var'] > 0.90))
        var_95 = np.min(np.where(df_pca_var['Cum_Var'] > 0.95))
        self.df_pca_var_ = df_pca_var
        
        # return number of components where 50%, 75%, 90%, and 95% of 
        # variance is explained
        print(f'50% variance explained at {var_50} components')
        print(f'75% variance explained at {var_75} components')
        print(f'90% variance explained at {var_90} components')
        print(f'95% variance explained at {var_95} components')

        # assign number of components where 95% of variation is explained to a self.variable
        self.pca95_ = var_95

        # return f'Embedding shape: {df_pca_embedding.shape - 3}'


        if plot_out:
            # save embeddings as dataframe and add color and colsize
            df_pca_embedding['Color'] = color_in
            df_pca_embedding['ColSize'] = 0.1

            # Plot embedding

            fig = px.scatter_3d(df_pca_embedding,
                                x = 'Comp0',
                                y = 'Comp1',
                                z = 'Comp2',
                                color = 'Color',
                                size = 'ColSize',
                                size_max = 10,
                                title = 'PCA',
                                custom_data = ['ID']
                                )
            # Edit so hover shows station id
            fig.update_traces(
            hovertemplate = "<br>".join([
                "STAID: %{customdata[0]}"
            ])
            )

            print(fig.show())

            # plot ratio of variance explained by each component
            fig2 = (
                    p9.ggplot(df_pca_var, p9.aes(x = 'component', y = 'ratio_var_expl')) +
                    p9.geom_bar(stat = 'identity', width = 0.5) +
                    p9.theme_light() +
                    p9.ylab('Ratio of Variance Explained')
            )

            print(fig2)

            # define input data frame
            # df_in = df_pca_var
            

            # plot cumulative variance explained by components
            fig3 = (
                    p9.ggplot(df_pca_var, p9.aes(x = 'component', y = 'Cum_Var')) +
                    p9.geom_line() +
                    p9.geom_point() +
                    p9.theme_light() +
                    p9.ylab('Cumulative Variance Explained')
            )

            print(fig3)
            
       



# %%
# %% define regression class

# Define a regressor class that allows:
# Multiple linear regression with 
# feature selection methods:
# Lasso regression, recursive feature elimination, 
# rfe w/k-fold cross-validation, and manual selection

class Regressor:
    """
    Regressing class


    Parameters
    -------------
    expl_vars: a pandas dataframe of explanatory variables
    resp_var: a pandas core.series.Series (a column from a pandas DataFrame)
        of the response variable

    Attributes
    -------------


    """

    def __init__(self, expl_vars, resp_var):
        
        # explanatory variables
        self.expl_vars = expl_vars
        # response variable
        self.resp_var = resp_var
       

    def stand_norm(self, method = 'standardize', not_tr = []):
        """
        Standardize or normalize explanatory variables

        Parameters
        --------------
        method: string equal to 'standardize' or 'normalize'
        not_tr: string or string vector
            list of column names not to transform
        
        Attributes
        --------------
        expl_vars_tr_: transformed explanatory variables
        """

        if method == 'standardize':
            stdsc = StandardScaler()
            self.scaler_ = stdsc.fit(self.expl_vars)
            expl_vars_tr_ = pd.DataFrame(
                self.scaler_.transform(self.expl_vars)
            )
            # give columns names to transformed data
            expl_vars_tr_.columns = self.expl_vars.columns
            # replace transformed vars with untransformed for specified columns
            expl_vars_tr_[not_tr] = self.expl_vars[not_tr]

            self.expl_vars_tr_ = expl_vars_tr_

        if method == 'normalize':
            normsc = MinMaxScaler()
            self.scaler_ = normsc.fit(self.expl_vars)
            expl_vars_tr_ = pd.DataFrame(
                self.scaler_.transform(self.expl_vars)
            )
            # give columns names to transformed data
            expl_vars_tr_.columns = self.expl_vars.columns
            # replace transformed vars with untransformed for specified columns
            expl_vars_tr_[not_tr] = self.expl_vars[not_tr]

            self.expl_vars_tr_ = expl_vars_tr_

    def lin_regression_select(self, 
        sel_meth = 'forward', 
        float_opt = 'False',
        min_k = 1,
        klim_in = 30,
        timeseries = False, 
        n_jobs_in = 1,
        plot_out = False): #, fl, fh):

        """
        Using explanatory and response data provided, explore linear regression 
        using specified method (sequential forward selection , sequential backward 
        selection, or exhuastive selection) for a number of variables up to kmax_in.
        ------------

        Parameters
        -------------
        sel_meth: string
            sepecifies what selection method to use
            options are 'forward', 'backward', or 'exhaustive'
        float_opt: Boolean
            if True, then set SFS float option to True.
            When set to True, an extra step in feature selection is performed allowing
            features to be removed, once they have been added.
            NOTE: only applies when sel_meth = 'forward' or 'backward'            
        min_k: integer
            specifies the minimum number of features to consider
            NOTE: only applies when sel_meth = 'exhaustive'
        klim_in: integer
            maximum number of features considered if using forward selection or minimum
            number of features considered if using backward selection
        timeseries: Boolean
            If set to True, then also calculate NSE, KGE, and % bias metrics
            NOTE: NSE and KGE are not calculated for individual basins here, but all
            catchments are used in a single calculation of NSE and KGE
            This application of NSE is equal to RMSE
        n_jobs_in: integer
            Specify number of processors to distribute job to

        Attributes
        -------------
        sfscv_out_: results of the cross validation used for features selection
            including feature index, performance for each fold, average performance,
            feature names, information about variation in performance from cv
        sfs_out_: features and coefficients
        df_lin_regr_performance_: pandas DataFrame

        """
        try:
            X_train, y_train = self.expl_vars_tr_, self.resp_var
        except:
            X_train, y_train = self.expl_vars, self.resp_var

        # instantiate sklearn linear regression object
        reg = LinearRegression()

        # apply linear regression using all explanatory variables
        # this object is used in feature selection just below and
        # results from this will be used to calculate Mallows' Cp
        reg_all = reg.fit(X_train, y_train)
        # calc sum of squared residuals for all data for use in 
        # Mallows' Cp calc
        reg_all_ssr = ssr(y_train, reg_all.predict(X_train))

        # define selection method (forward, backward, or exhaustive)
        if sel_meth == 'forward':
            forward_in = True
        elif sel_meth =='backward':
            forward_in = False

        if sel_meth == 'forward' or sel_meth == 'backward':
            # create sequential selection object
            sfs_obj = SFS(reg_all, 
                n_jobs = n_jobs_in, # how to distribute
                k_features = klim_in, 
                forward = forward_in, # set to False for backward
                floating = float_opt,
                verbose = 1, # 0 for no output, 1 for number features in set, 2 for more detail
                scoring = 'r2', # 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_median_absolute_error'
                cv = 10
            )
        elif sel_meth == 'exhaustive':
            sfs_obj = EFS(reg_all,
                n_jobs = n_jobs_in,
                min_features = min_k,
                max_features = klim_in,
                scoring = 'r2',
                print_progress = True,
                cv = 2
                )
        
        else:
            print('Please enter a valid string for sel_meth')
        # for scoring options you can import sklearn and print options using code below
        # import sklearn
        # sklearn.metrics.SCORERS.keys()
        
        # save sequential selection results in dataframe
        sfs_fit = sfs_obj.fit(X_train, y_train)
        self.sfscv_out_ = pd.DataFrame.from_dict(sfs_fit.get_metric_dict()).T
        # reset index
        self.sfscv_out_ = self.sfscv_out_.reset_index()
        
        # intialize empty vectors to hold results below
        # number of features, ssr, r2, adj, r2, mae, rmse,
        # AIC, BIC, Mallows' Cp, VIF, % Bias, NSE, KGE
        n_f_all = []
        ssr_all = []
        r2_all = []
        r2adj_all = []
        mae_all = []
        rmse_all = []
        AIC_all = []
        BIC_all = []
        M_Cp_all = []
        VIF_all = []
        percBias_all = []
        if timeseries:
            # Nash-Sutcliffe Efficeincy
            NSE_all = []
            # Kling Gupta Efficiency
            KGE_all = []

        # Apply a regression model using each set of features and report out:
        # performance metrics, coefficients, and intercept.
        for i in range(0, (self.sfscv_out_.shape[0]), 1):
            # features from each model
            feat_in = list(self.sfscv_out_.loc[i, 'feature_names'])
            # input explanatory vars
            expl_in = X_train[feat_in]
            
            # apply linear regression object reg
            lr_out = reg.fit(expl_in, y_train)
            # predicted water yield from training data
            ypred_out = lr_out.predict(expl_in)
            # sample size
            n_k_out = expl_in.shape[0]
            # number of features
            n_f_out = expl_in.shape[1]
            # sum of squared residuals
            ssr_out = ssr(ypred_out, y_train)
            # unadjusted R-squared
            r2_out = r2_score(y_train, ypred_out)
            # Adjusted R-squared
            r2adj_out = R2adj(n_k_out, n_f_out, r2_out)
            # Mean absolute error
            mae_out = mean_absolute_error(y_train, ypred_out)
            # Root mean squared error
            rmse_out = mean_squared_error(y_train, ypred_out, squared = False)
            # Akaike Information Criterion
            AIC_out = AIC(n_k_out, ssr_out, n_f_out)
            # Bayesian Information Criterion
            BIC_out = BIC(n_k_out, ssr_out, n_f_out)
            # Mallows' Cp
            M_Cp_out = M_Cp(reg_all_ssr, ssr_out, n_k_out, n_f_out)
            # VIF
            VIF_out = VIF(expl_in)
            # Percent Bias
            percBias_out = PercentBias(ypred_out, y_train)

            # append results to vectors holding results for all models
            n_f_all.append(n_f_out)
            ssr_all.append(ssr_out)
            r2_all.append(r2_out)
            r2adj_all.append(r2adj_out)
            mae_all.append(mae_out)
            rmse_all.append(rmse_out)
            AIC_all.append(AIC_out)
            BIC_all.append(BIC_out)
            M_Cp_all.append(M_Cp_out)
            VIF_all.append(VIF_out)
            percBias_all.append(percBias_out)
            # If timeseries = True then also calculate NSE and KGE
            if timeseries:
                # Nash-Sutcliffe Efficeincy
                NSE_out = NSE(ypred_out, y_train)
                # Kling Gupta Efficiency
                KGE_out = KGE(ypred_out, y_train)

                # append results to vectors holding results
                NSE_all.append(NSE_out)
                KGE_all.append(KGE_out)


        # Create pandas DataFrame and add all results to it
        if timeseries:
            df_perfmetr = pd.DataFrame({
                'n_features': n_f_all,
                'ssr': ssr_all,
                'r2': r2_all,
                'r2adj': r2adj_all,
                'mae': mae_all,
                'rmse': rmse_all,
                'AIC': AIC_all,
                'BIC': BIC_all,
                'M_Cp': M_Cp_all,
                'VIF': VIF_all,
                'percBias': percBias_all,
                'NSE': NSE_all,
                'KGE': KGE_all
            })
        else:
            df_perfmetr = pd.DataFrame({
                'n_features': n_f_all,
                'ssr': ssr_all,
                'r2': r2_all,
                'r2adj': r2adj_all,
                'mae': mae_all,
                'rmse': rmse_all,
                'AIC': AIC_all,
                'BIC': BIC_all,
                'M_Cp': M_Cp_all,
                'VIF': VIF_all,
                'percBias': percBias_all
            })
        
        # assign performance metric dataframe to self, plot, and print results
        self.df_lin_regr_performance_ = df_perfmetr
        # if plot_out = True then output lots
        if plot_out:
            PlotPM(self.df_lin_regr_performance_, timeseries = False)
        print(df_perfmetr.sort_values(by = 'BIC')[0:5])
        
    

               

    def lasso_regression(self,
                        alpha_in = 1, 
                        max_iter_in = 1000,
                        n_splits_in = 10,
                        n_repeats_in = 1,
                        random_state_in = 100,
                        timeseries = False,
                        n_jobs_in = 1):
        # using info from here as guide: 
        # https://machinelearningmastery.com/lasso-regression-with-python/
        """
        Parameters
        -------------
        alpha_in: int, float, or list [0:1]
            parameter controlling strength of L1 penalty.
            alpha = 0 same as LinearRegression, but advised not to use alpha = 0
            if alpha_in is list, then cross validation will be used for Lasso regression
        max_iter_in: integer [0:Inf)
            Maximum number of iterations
            Default is 1000
        n_splits_in: integer
            Nunmber of folds for k-fold CV
        n_repeats_in: integer
            Number of times to repreat k-fold CV
        random_state_in: integer
            random seed for stochastic processes (helps ensure reproducability)
        timeseries: Boolean
            If set to True, then also calculate NSE, KGE, and % bias metrics
        n_jobs_in: -1 or positive integer
            -1 means run on all available cores
        id_var: pandas Series or array
            unique identifiers for catchments (e.g., STAID)

        Attributes
        -------------
        lasso_reg_: sklearn regression object
            output model from lasso regression
        lasso_scores_: numpy.ndarray
            scores from cross validation
        features_coef_: pandas DataFrame
            features with non-zero coefficients and the coefficients
        df_lasso_features_coef_: pandas DataFrame
            features and coefficients from lasso regression
        df_lasso_reg_performance_: pandas DataFrame
            performance metrics for training data
        lasso_alpha_: float
            the alpha value used in the final lasso model
            if a single alpha value is pvoided to the regressor object then lasso_alpha_ is
            equal to that value.
            if a list of alpha values are provided then lasso_alpha_ is equal to the best
            alpha value from cross validation.
        df_NSE_KGE_: pandas DataFrame
            holds output of NSE and KGE for each catchment
            columns = ['STAID', 'NSE', 'KGE']
        """
        try:
            X_train, y_train = self.expl_vars_tr_, self.resp_var
        except:
            X_train, y_train = self.expl_vars, self.resp_var

        self.lasso_reg_ = Lasso(alpha = alpha_in,
                            max_iter = max_iter_in)
        
        


        ##### k-fold CV if alpha_in is a list
        # Following code found here:
        # https://machinelearningmastery.com/lasso-regression-with-python/
        if isinstance(alpha_in, list):
            # define model evaluation method
            cv = RepeatedKFold(
                    n_splits = n_splits_in, 
                    n_repeats = n_repeats_in, 
                    random_state = random_state_in)
            # define grid
            grid = dict()
            grid['alpha'] = alpha_in
            # define search
            search = HalvingGridSearchCV(self.lasso_reg_,
                grid,
                scoring = 'neg_root_mean_squared_error', 
                            # 'neg_mean_absolute_error', 'r2'],
                refit = False,
                cv = cv,
                n_jobs = n_jobs_in)
            # perform the search
            results = search.fit(X_train, y_train)
            self.lassoCV_results = results
            # self.lassoCV_results = pd.DataFrame(self.lassoCV_results.cv_results_)

            # print(pd.DataFrame(self.lassoCV_results.cv_results_))
            print(search.best_params_)
            # print top 10 performing based on scores
            self.df_lassoCV_rmse = pd.DataFrame(self.lassoCV_results.cv_results_).sort_values(by = 'rank_test_score')[[
                            'param_alpha',
                            'mean_test_score',
                            'std_test_score',
                            'rank_test_score']]

            # redfine alpha as best alpha
            alpha_in = search.best_params_['alpha']
            # write best alpha to self
            self.lasso_alpha_ = search.best_params_['alpha']

            # redfine lasso using update alpha value
            self.lasso_reg_ = Lasso(alpha = alpha_in,
                            max_iter = max_iter_in)
            
            # self.df_lassoCV_rmse = pd.DataFrame(self.lassoCV_results.cv_results_).sort_values(by = 'rank_test_neg_root_mean_squared_error')[[
            #                 'param_alpha',
            #                 'mean_test_neg_root_mean_squared_error',
            #                 'std_test_neg_root_mean_squared_error',
            #                 'rank_test_neg_root_mean_squared_error']]
            # self.df_lassoCV_mae = pd.DataFrame(self.lassoCV_results.cv_results_).sort_values(by = 'rank_test_neg_root_mean_squared_error')[[
            #                 'param_alpha',
            #                 'mean_test_neg_mean_absolute_error',
            #                 'std_test_neg_mean_absolute_error',
            #                 'rank_test_neg_mean_absolute_error']]
            # self.df_lassoCV_r2 = pd.DataFrame(self.lassoCV_results.cv_results_).sort_values(by = 'rank_test_r2')[[
            #                 'param_alpha',
            #                 'mean_test_r2',
            #                 'std_test_r2',
            #                 'rank_test_r2']]
            
            # return(# self.lassoCV_results.cv_results_, 
            #         self.df_lassoCV_rmse[0:10],
            #         self.df_lassoCV_mae[0:10], 
            #         self.df_lassoCV_r2[0:10])
            # summarize (best score and best params aren't available when using
            # more than 1 scoring metric)
            # print('MAE: %.3f' % results.cv_results_.best_score_)
            # print('Config: %s' % results.cv_results_.best_params_)

            #

        else:
            # fit model
            self.lasso_reg_.fit(X_train, y_train)

            # return variables with non-zero coeficients
            # with intercept appended

            coef_int = np.append(self.lasso_reg_.coef_[
                                np.where(np.abs(self.lasso_reg_.coef_) > 10e-20)
                                ], self.lasso_reg_.intercept_)
            feat_int = np.append(self.lasso_reg_.feature_names_in_[
                                np.where(np.abs(self.lasso_reg_.coef_) > 10e-20)
                                ], 'intercept')

            self.df_lasso_features_coef_ = pd.DataFrame({
                'features':  feat_int,
                'coefficients': coef_int,
            })

            # Calculate performance metrics and output as single row pandas DataFrame
            # pandas dataframe including only features selected in lasso regression
            expl_out = X_train[self.df_lasso_features_coef_.drop(
                            self.df_lasso_features_coef_.tail(1).index)['features']]
            # predicted water yield from training data
            ypred_out = self.lasso_reg_.predict(X_train)
            

            # NOTE: 
            # Commenting out k-folds, because not lending itself to progressing through models
            # was only printing to see the results

            #### Apply k-fold cross validation to see how performs across samples
            # # define k-fold model
            # cv = RepeatedKFold(n_splits = n_splits_in,
            #         n_repeats = n_repeats_in,
            #         random_state = random_state_in)

            # #evaluate model
            # scores = cross_validate(self.lasso_reg_,
            #     X_train, y_train, 
            #     scoring = ('neg_root_mean_squared_error', 
            #     'neg_mean_absolute_error', 'r2'), cv = cv, n_jobs = 1,
            #     return_train_score = True,
            #     return_estimator = False)

            # self.lasso_scores_ = scores


            # rmse_train = -self.lasso_scores_['train_neg_root_mean_squared_error']
            # rmse_test = -self.lasso_scores_['test_neg_root_mean_squared_error']
            # mae_train = -self.lasso_scores_['train_neg_mean_absolute_error']
            # mae_test = -self.lasso_scores_['test_neg_mean_absolute_error']
            # print('Mean CV training RMSE (stdev): %.3f (%.3f)' % (np.mean(rmse_train), np.std(rmse_train)))
            # print('Mean CV testing RMSE (stdev): %.3f (%.3f)' % (np.mean(rmse_test), np.std(rmse_test)))
            # print('Mean CV training MAE: %.3f (%.3f)' % (np.mean(mae_train), np.std(mae_train)))
            # print('Mean CV testing MAE: %.3f (%.3f)' % (np.mean(mae_test), np.std(mae_test)))

            #####

            # sample size
            n_k_out = X_train.shape[0]
            # number of features
            n_f_out = self.df_lasso_features_coef_.shape[0] - 1
            # sum of squared residuals
            ssr_out = ssr(ypred_out, y_train)
            # unadjusted R-squared
            r2_out = r2_score(y_train, ypred_out)
            # Adjusted R-squared
            r2adj_out = R2adj(n_k_out, n_f_out, r2_out)
            # Mean absolute error
            mae_out = mean_absolute_error(y_train, ypred_out)
            # Root mean squared error
            rmse_out = mean_squared_error(y_train, ypred_out, squared = False)
            # Akaike Information Criterion
            AIC_out = AIC(n_k_out, ssr_out, n_f_out)
            # Bayesian Information Criterion
            BIC_out = BIC(n_k_out, ssr_out, n_f_out)
            # Mallows' Cp
            # apply linear regression using all explanatory variables
            # this object is used in feature selection just below and
            # results from this will be used to calculate Mallows' Cp
            reg_all = LinearRegression().fit(X_train, y_train)
            # calc sum of squared residuals for all data for use in
            # Mallows' Cp calc
            reg_all_ssr = ssr(y_train, reg_all.predict(X_train))
            M_Cp_out = M_Cp(reg_all_ssr, ssr_out, n_k_out, n_f_out)
            # VIF
            VIF_out = VIF(expl_out)
            # Percent Bias
            percBias_out = PercentBias(ypred_out, y_train)
    
            if timeseries:
                # Nash-Sutcliffe Efficeincy
                NSE_out = NSE(ypred_out, y_train)
                # Kling Gupta Efficiency
                KGE_out = KGE(ypred_out, y_train)

                # # define input dataframe for NSE and KGE function
                # df_tempin = pd.DataFrame({
                #     'y_pred': ypred_out,
                #     'y_obs': y_train,
                #     'ID_in': id_var
                # })

                # # apply function to calc NSE and KGE for all catchments
                # self.df_NSE_KGE_ = NSE_KGE_Apply(df_in = df_tempin)

                # # assign mean NSE and KGE to variable to be added to output metrics
                # NSE_out = np.round(np.mean(self.df_NSE_KGE['NSE']), 4)
                # KGE_out = np.round(np.mean(self.df_NSE_KGE['KGE']), 4)
        
            # write performance metrics to dataframe
            if timeseries:
                df_perfmetr = pd.DataFrame({
                    'n_features': n_f_out,
                    'ssr': ssr_out,
                    'r2': r2_out,
                    'r2adj': r2adj_out,
                    'mae': mae_out,
                    'rmse': rmse_out,
                    'AIC': AIC_out,
                    'BIC': BIC_out,
                    'M_Cp': M_Cp_out,
                    'VIF': [VIF_out],
                    'percBias': percBias_out,
                    'NSE': NSE_out,
                    'KGE': KGE_out
                })
            else:
                df_perfmetr = pd.DataFrame({
                    'n_features': n_f_out,
                    'ssr': ssr_out,
                    'r2': r2_out,
                    'r2adj': r2adj_out,
                    'mae': mae_out,
                    'rmse': rmse_out,
                    'AIC': AIC_out,
                    'BIC': BIC_out,
                    'M_Cp': M_Cp_out,
                    'VIF': [VIF_out],
                    'percBias': percBias_out
                })
    
            # assign performance metric dataframe to self
            self.df_lasso_regr_performance_ = df_perfmetr
            return(df_perfmetr)


    def xgb_regression(self,
                       n_splits_in = 10,
                       n_repeats_in = 1,
                       random_state_in = 100,
                       grid_in = {
                        'n_estimators': [100], # [100, 250, 500], # [10], # 
                        'colsample_bytree': [1], # [0.7, 1], 
                        'max_depth': [6], # [4, 6, 8],
                        'gamma': [0], # [0, 1], 
                        'reg_lambda': [0], # [0, 1, 2]
                        'learning_rate': [0.3] # [0.02, 0.1, 0.3]
                        },
                       timeseries = False,
                       n_jobs_in = -1,
                       halving_factor = 3,
                       dir_save = 'D:/Projects/GAGESii_ANNstuff/Python/Scripts/Learning_Results/xgbreg_learn_model.json',
                       id_var = ''):
        # using info from here as guide: 
        # https://machinelearningmastery.com/lasso-regression-with-python/
        """
        Parameters
        -------------
        n_splits_in: integer
            Nunmber of folds for k-fold CV
        grid_in: dictionary
            holds hyperparameters and a range of values for each hyperparameter
            to consider in tuning.
            Okay to enter only one value for each hyperparameter to simply set
            the hyperparameters to those values
        n_repeats_in: integer
           Number of times to repreat k-fold CV
        random_state_in: integer
            random seed for stochastic processes (helps ensure reproducability)
        timeseries: Boolean
            If set to True, then also calculate NSE, KGE, and % bias metrics
        n_jobs_in: -1 or positive integer
            -1 means run on all available cores
        halving_factor: int or float
            sepcifies portion of candidates chosen for subsequent round in 
            HalvingGridSearchCV (e.g., 3 means 1/3 of candidates are chosen)

        Attributes
        -------------
        xgb_reg_: XGBoost regression object using sklearn API
            output model from lasso regression
        df_xgbcv_results_: pandas dataframe
            output results from cross validation
            This attribute is not created if all items in grid_in are of length 1
        df_xgboost_performance_: pandas DataFrame
            performance metrics for training data
        """
        try:
            X_train, y_train = self.expl_vars_tr_, self.resp_var
        except:
            X_train, y_train = self.expl_vars, self.resp_var

        # define xgboost model
        xgb_reg = xgb.XGBRegressor(
            objective = 'reg:squarederror',
            tree_method = 'hist', # 'gpu_hist',
            verbosity = 1, # 0 = silent, 1 = warning (default), 2 = info, 3 = debug
            sampling_method = 'uniform', # 'gradient_based', # default is 'uniform'
            # nthread = 8 # defaults to maximum number available, only use for less threads
            )


        # if all variables in grid are not of length 1, then perform cross-validation
        if any(type(grid_in[x]) == list for x in grid_in):

            try:
                # define model evaluation method
                cv = RepeatedKFold(
                            n_splits = n_splits_in, 
                            n_repeats = n_repeats_in, 
                            random_state = random_state_in)

                # define gridsearch cross-validation object
                gsCV = HalvingGridSearchCV(
                            estimator = xgb_reg,
                            param_grid = grid_in,
                            factor = halving_factor,
                            scoring = 'neg_root_mean_squared_error',
                            cv = cv, # default = 5
                            return_train_score = False, # True, # default = False
                            random_state = random_state_in,
                            verbose = 2,
                            n_jobs = -1,
                            refit = False
                        )


                # apply gridsearch cross-validation
                time_st = time.perf_counter()
                cv_fitted = gsCV.fit(X_train, y_train)

                time_end = time.perf_counter()
                print(f'\n grid search took {(time_end - time_st)/60: .2f} minutes \n')

                # print results
                self.df_xgbcv_results_ = pd.DataFrame(cv_fitted.cv_results_).sort_values(by = 'rank_test_score')

                # print best parameters
                print(f'\n {cv_fitted.best_params_} \n')

                # redefine grid_in as cv_fitted.best_params_ if it exists
                if isinstance(cv_fitted.best_params_, dict):
                    grid_in = cv_fitted.best_params_
        
            except TypeError:
                print('If cross-validation grid search is desired then enter values into grid as lists.')
                print('For example using brackets, [], around the value(s).')
                print('Even if a single value is entered for a variable, it must be entered as a list')
                print('If cross-validation grid search is not desired, then enter each value as an integer or float')

        # assign parameters used in model to self
        self.xgboost_params_ = pd.DataFrame.from_dict(grid_in, orient = 'index').T

        # apply XGBOOST using best parameters from cross-validation

        # define xgboost model
        xgb_reg = xgb.XGBRegressor(
            objective = 'reg:squarederror',
            tree_method = 'hist', # 'gpu_hist',
            verbosity = 1, # 0 = silent, 1 = warning (default), 2 = info, 3 = debug
            sampling_method = 'uniform', # 'gradient_based', # default is 'uniform'
            # nthread = 4 defaults to maximum number available, only use for less threads
            n_estimators = grid_in['n_estimators'],
            colsample_bytree = grid_in['colsample_bytree'],
            max_depth = grid_in['max_depth'],
            gamma = grid_in['gamma'],
            reg_lambda = grid_in['reg_lambda'],
            learning_rate = grid_in['learning_rate']
        )

        # train model
        xgb_reg.fit(X_train, y_train)

        # return predicted values from training
        ypred_out = xgb_reg.predict(X_train)


        # Save Model and reload model
        # save model
        xgb_reg.save_model(dir_save)

        print(f'model saved at {dir_save}')
        
        # calculate performance metrics
        # sample size
        n_k_out = X_train.shape[0]
        # number of features
        n_f_out = X_train.shape[1]
        # sum of squared residuals
        ssr_out = ssr(ypred_out, y_train)
        # unadjusted R-squared
        r2_out = r2_score(y_train, ypred_out)
        # Adjusted R-squared
        r2adj_out = R2adj(n_k_out, n_f_out, r2_out)
        # Mean absolute error
        mae_out = mean_absolute_error(y_train, ypred_out)
        # Root mean squared error
        rmse_out = mean_squared_error(y_train, ypred_out, squared = False)
        # Akaike Information Criterion
        AIC_out = AIC(n_k_out, ssr_out, n_f_out)
        # Bayesian Information Criterion
        BIC_out = BIC(n_k_out, ssr_out, n_f_out)
        # # Mallows' Cp
        # # apply linear regression using all explanatory variables
        # # this object is used in feature selection just below and
        # # results from this will be used to calculate Mallows' Cp
        # reg_all = LinearRegression().fit(X_train, y_train)
        # # calc sum of squared residuals for all data for use in
        # # Mallows' Cp calc
        # reg_all_ssr = ssr(y_train, reg_all.predict(X_train))
        # M_Cp_out = M_Cp(reg_all_ssr, ssr_out, n_k_out, n_f_out)
        M_Cp_out = np.nan
        # VIF
        VIF_out = VIF(X_train)
        # Percent Bias
        percBias_out = PercentBias(ypred_out, y_train)
 
        if timeseries:
            # # Nash-Sutcliffe Efficeincy
            # NSE_out = NSE(ypred_out, y_train)
            # # Kling Gupta Efficiency
            # KGE_out = KGE(ypred_out, y_train)

            # define input dataframe for NSE and KGE function
            df_tempin = pd.DataFrame({
                'y_pred': ypred_out,
                'y_obs': y_train,
                'ID_in': id_var
            })

            # apply function to calc NSE and KGE for all catchments
            self.df_NSE_KGE_ = NSE_KGE_Apply(df_in = df_tempin)

            # assign mean NSE and KGE to variable to be added to output metrics
            NSE_out = np.round(np.mean(self.df_NSE_KGE_['NSE']), 4)
            KGE_out = np.round(np.mean(self.df_NSE_KGE_['KGE']), 4)
            # recalculate percent bias as mean of individual catchments
            percBias_out = np.round(np.mean(self.df_NSE_KGE_['PercBias']), 4)
            # calculate RMSE from time series
            rmsets_out = np.round(np.mean(self.df_NSE_KGE_['RMSE']), 4)


        # write performance metrics to dataframe
        if timeseries:
            df_perfmetr = pd.DataFrame({
                'n_features': n_f_out,
                'ssr': ssr_out,
                'r2': r2_out,
                'r2adj': r2adj_out,
                'mae': mae_out,
                'rmse': rmse_out,
                'VIF': [VIF_out],
                'percBias': percBias_out,
                'NSE': NSE_out,
                'KGE': KGE_out,
                'RMSEts': rmsets_out
            })
        else:
            df_perfmetr = pd.DataFrame({
                'n_features': n_f_out,
                'ssr': ssr_out,
                'r2': r2_out,
                'r2adj': r2adj_out,
                'mae': mae_out,
                'rmse': rmse_out,
                'VIF': [VIF_out],
                'percBias': percBias_out
            })
            
     
        # write performance metrics to dataframe
        if timeseries:
            df_perfmetr = pd.DataFrame({
                'n_features': n_f_out,
                'ssr': ssr_out,
                'r2': r2_out,
                'r2adj': r2adj_out,
                'mae': mae_out,
                'rmse': rmse_out,
                'AIC': AIC_out,
                'BIC': BIC_out,
                'M_Cp': M_Cp_out,
                'VIF': [VIF_out],
                'percBias': percBias_out,
                'NSE': NSE_out,
                'KGE': KGE_out
            })
        else:
            df_perfmetr = pd.DataFrame({
                'n_features': n_f_out,
                'ssr': ssr_out,
                'r2': r2_out,
                'r2adj': r2adj_out,
                'mae': mae_out,
                'rmse': rmse_out,
                'AIC': AIC_out,
                'BIC': BIC_out,
                'M_Cp': M_Cp_out,
                'VIF': [VIF_out],
                'percBias': percBias_out
            })

        
        # save xgb predictor object to self
        self.xgb_reg_ = xgb_reg


        # assign performance metric dataframe to self
        self.df_xgboost_performance_ = df_perfmetr
        return(df_perfmetr)





        ######################


    def pred_plot(self,
        model_in,
        X_pred,
        y_obs,
        id_color,
        timeseries = False,
        plot_out = False,
        id_var = ''):
        """
        Takes ... add description
        
        Parameters
        ----------
        model_in: an sklearn model like object
            it must have a .fit or .predict method (e.g., LinearRegression().fit())
        X_pred: pd.DataFrame
            features of the same name and type as in X_train, but associated with the time
            and/or conditions associated with the response variable being predicted
        y_obs: numpy array
            Observed response variables to be used for training and validating model
        id_color: numeric or character array type object
            e.g., numpy array or pandas.series
            Variable used to color points in predicted vs observed plots (e.g., 'aggecoregion')
        timeseries: Boolean
            if True, then KGE and NSE are included in performance metrics
        plot_out: Boolean
            if True, then output plots, otherwise, do not
        id_var: pandas series or array
            unique identifier for catchments (e.g., STAID)
        
        Attributes
        ----------
        plots: Outputs plots of predicted vs. observed along with performance metrics
        df_pred_performance_: pandas DataFrame
            A DataFrame holding performance metrics of the applied model (model_in)
        pred_model_coef_: pandas DataFrame
            DataFrame of features and their coefficients
        pred_model_interc_: float
            the intercept from the model used in prediction
        """

        try:
            # output model features, associated coefficeints, and intercept
            # append intercept to features
            features_int = np.append(model_in.coef_, model_in.intercept_)
            # try:
            features_int_names = np.append(model_in.feature_names_in_, 'intercept')
            # except:
            #     features_int_names = np.append([X_pred.name], 'intercept')
            #     X_pred = pd.DataFrame(np.array(X_pred).reshape(-1,1))



            # output to self
            self.df_linreg_features_coef_ = pd.DataFrame({
                'features': features_int_names,
                'coef': features_int
            })

        except:
            print('features and ceofficients unable to be calculated.')
            print('This is likely due to the model type being used (e.g., xgboost regressor).')

        # # return variables with non-zero coeficients
        # self.lasso_features_coef_ = pd.DataFrame({
        #     'features':  self.lasso_reg_.feature_names_in_[
        #                     np.where(np.abs(self.lasso_reg_.coef_) > 10e-20)
        #                     ],
        #     'coefficients': self.lasso_reg_.coef_[
        #                     np.where(np.abs(self.lasso_reg_.coef_) > 10e-20)
        #                     ]
        # })

        # Predict y with input model
        y_pred = model_in.predict(X_pred)
        # Calculate performance metrics and output as single row pandas DataFrame
        # sample size
        n_k_out = X_pred.shape[0]
        # number of features
        # if from lasso equation then use model_in.coef_ to get number of features
        # otherwise, use explanatory variable dataframe
        try:
            n_f_out = len(np.where(np.abs(model_in.coef_) > 0)[0])
        except:
            n_f_out = X_pred.shape[1]
        # sum of squared residuals
        ssr_out = ssr(y_pred, y_obs)
        # unadjusted R-squared
        r2_out = r2_score(y_obs, y_pred)
        # Adjusted R-squared
        r2adj_out = R2adj(n_k_out, n_f_out, r2_out)
        # Mean absolute error
        mae_out = mean_absolute_error(y_obs, y_pred)
        # Root mean squared error
        rmse_out = mean_squared_error(y_obs, y_pred, squared = False)
        # VIF of variables in predictive model
        # if lasso regression, then only use features with non-zero coefficeints
        # else use X_pred directoy
        try:
            self.lasso_reg_.l1_ratio
            X_in = X_pred[self.df_lasso_features_coef_.drop(
                self.df_lasso_features_coef_.tail(1).index)['features']]
            VIF_out = VIF(X_in)
        except:
            VIF_out = VIF(X_pred)
        # Percent Bias
        percBias_out = PercentBias(y_pred, y_obs)

        # If timeseries = True then also calculate NSE and KGE
        if timeseries:
            # # Nash-Sutcliffe Efficeincy
            # NSE_out = NSE(y_pred, y_obs)
            # # Kling Gupta Efficiency
            # KGE_out = KGE(y_pred, y_obs)

            # define input dataframe for NSE and KGE function
            df_tempin = pd.DataFrame({
                'y_pred': y_pred,
                'y_obs': y_obs,
                'ID_in': id_var
            })
                        
            # apply function to calc NSE and KGE for all catchments
            self.df_NSE_KGE_ = NSE_KGE_Apply(df_in = df_tempin)

            # assign mean NSE and KGE to variable to be added to output metrics
            NSE_out = np.round(np.mean(self.df_NSE_KGE_['NSE']), 4)
            KGE_out = np.round(np.mean(self.df_NSE_KGE_['KGE']), 4)
            # recalculate percent bias as mean of individual catchments
            percBias_out = np.round(np.mean(self.df_NSE_KGE_['PercBias']), 4)
            # calculate RMSE from time series
            rmsets_out = np.round(np.mean(self.df_NSE_KGE_['RMSE']), 4)


        # write performance metrics to dataframe
        if timeseries:
            df_perfmetr = pd.DataFrame({
                'n_features': n_f_out,
                'ssr': ssr_out,
                'r2': r2_out,
                'r2adj': r2adj_out,
                'mae': mae_out,
                'rmse': rmse_out,
                'VIF': [VIF_out],
                'percBias': percBias_out,
                'NSE': NSE_out,
                'KGE': KGE_out,
                'RMSEts': rmsets_out
            })
        else:
            df_perfmetr = pd.DataFrame({
                'n_features': n_f_out,
                'ssr': ssr_out,
                'r2': r2_out,
                'r2adj': r2adj_out,
                'mae': mae_out,
                'rmse': rmse_out,
                'VIF': [VIF_out],
                'percBias': percBias_out
            })

        if plot_out:
            # prepare dataframe for plotting performance
            df_in = pd.DataFrame({
                'observed': y_obs,
                'predicted': y_pred,
                'ID': id_color
            })
            # Plot predicted vs observed
            p = (
                    p9.ggplot(data = df_in) +
                    p9.geom_point(p9.aes(x = 'observed', 
                                            y = 'predicted', 
                                            color = 'ID')) +
                    p9.geom_abline(slope = 1) +
                    p9.theme_bw() +
                    p9.theme(axis_text = p9.element_text(size = 14),
                                axis_title = p9.element_text(size = 14),
                                aspect_ratio = 1,
                                legend_text = p9.element_text(size = 14),
                                legend_title = p9.element_text(size = 14))
                )
            print(p)
        # assign performance metric dataframe to self
        self.df_pred_performance_ = df_perfmetr
        return(df_perfmetr)





# %%
