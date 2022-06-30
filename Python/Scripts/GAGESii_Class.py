# %% Intro

# Defining a class to hold and process methods associated with 
# 1. clustering methods
# 2. Feature selection and regression
# 3. Other methods such as standardization and normalization


# %% Import libraries

import numpy as np # matrix operations/data manipulation
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
from sklearn.model_selection import RepeatedKFold # for repeated k-fold validation
from sklearn.model_selection import cross_val_score # only returns a single score (e.g., MAE)
from sklearn.model_selection import cross_validate # can be used to return multiple scores 
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn_extra.cluster import KMedoids
# import statsmodels.formula.api as smf # for ols, aic, bic, etc.
# import statsmodels.api as sm # for ols, aic, bic, etc.
# from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs # plot sfs results
import matplotlib.pyplot as plt # plotting
# from matplotlib import cm
import plotnine as p9
import umap # for umap training and projection
import hdbscan # for hdbscan clustering
# from mpl_toolkits.mplot3d import Axes3D # plot 3d plot
# import plotly as ply # interactive plots
# import plotly.io as pio5 # easier interactive plots
import plotly.express as px # easier interactive plots
import seaborn as sns # easier interactive plots
import pandas as pd # data wrangling
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
                # stdsc.fit_transform(self.clust_vars)
            )
            # give columns names to transformed data
            clust_vars_tr_.columns = self.clust_vars.columns
            # replace transformed vars with untransformed for specified columns
            clust_vars_tr_[not_tr] = self.clust_vars[not_tr]

            self.clust_vars_tr_ = clust_vars_tr_

        if method == 'normalize':
            normsc = MinMaxScaler()
            self.scaler_ = normsc.fit(self.clust_vars)
            self.clust_vars_tr_ = self.scaler_.transform(self.clust_vars)

    
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
        ks_predicted_: list of arrays with the labels of which
            group, k, in which each catchment was placed
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
        self.ks_predicted_ = ks_out
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
        # min_sample = 20
        gm_spantree = True,
        metric_in = 'euclidean',
        # clust_seleps = 0,
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
            for epsilon = 0.5, and clusters that emereved at distances greater than 0.5 will
            be untouched
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

        hd_clusterer = hdbscan.HDBSCAN(min_cluster_size = min_clustsize,
                                # min_samples = 20, 
                                gen_min_span_tree = gm_spantree,
                                metric = metric_in,
                                # cluster_selection_epsilon = clust_seleps,
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

        # plot the spanning tree
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



        
    def umap_reducer(self, 
        nn = 3, 
        mind = 0.1,
        sprd = 1,
        nc = 3,
        color_in = 'blue'):
        
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
                        n_epochs = 200) # 200 for large datasets; 500 small

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


    Attributes
    -------------


    """

    def __init__(self, expl_vars, resp_var, id_var):
        
        # explanatory variables
        self.expl_vars = expl_vars
        # response variable
        self.resp_var = resp_var
        # ID vars (e.g., gauge numbers)
        self.id_var = id_var

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
            self.expl_vars_tr_ = self.scaler_.transform(self.expl_vars)

    def lin_regression_select(self, kmax_in = 30): #, fl, fh):

        """
        Using explanatory and response data provided, explore linear regression 
        using specified method (sequential forward selection , sequential backward 
        selection, or exhuastive selection) for a number of variables up to kmax_in.
        ------------

        Parameters
        -------------
        kmax_in: integer
            maximum number of features considered
        fl: integer
            smallest (or lowest) number of features to consider (e.g., 1)
        fh: integer
            largest (or highest) number of features to consider (e.g, 81)

        Attributes
        -------------
        vif_data_: pandas dataframe
            dataframe holding vif for all variables

        res_metr_: results metrics including MAE, RMSE, R2, R2_adj. , 
            AIC, BIC, Mallows' Cp, and for time series, NSE and Bias. 

        sfscv_out_: results of the cross validation used for features selection
            including feature index, performance for each fold, average performance,
            feature names, information about variation in performance from cv
        sfs_out_: features and coefficients

        """
        try:
            X_train, y_train = self.expl_vars_tr_, self.resp_var
        except:
            X_train, y_train = self.expl_vars, self.resp_var

        # instantiate sklearn linear regression object
        reg = LinearRegression()
        # apply linear regression using all explanatory variables
        # Results from this will be used to calculate Mallows' Cp
        reg_all = reg.fit(X_train, y_train)


        # create sequential selection object
        sfs_obj = SFS(reg_all, 
            k_features = kmax_in, 
            forward = True, # set to False for backward)
            floating = False,
            verbose = 1, # 0 for no output, 1 for number features in set, 2 for more detail
            scoring = 'r2', # 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_median_absolute_error'
            cv = 10
        )
        # for scoring options you can import sklearn and print options using code below
        # import sklearn
        # sklearn.metrics.SCORERS.keys()
        
        sfs_fit = sfs_obj.fit(X_train, y_train)
        
        self.sfs_features_ = sfs_fit.k_feature_names_
        # self.sfs_
        # sfs_coefs = sfs_fit.coef_

        # self.sfs_out_ = pd.DataFrame({
        #     'feature': sfs_features,
        #     'coef': sfs_coefs
        # })
        
        # # save sequential selection results in dataframe
        self.sfscv_out_ = pd.DataFrame.from_dict(sfs_fit.get_metric_dict()).T

        fig = plot_sfs(sfs_fit.get_metric_dict(), kind = 'std_dev')
        print(fig)
        plt.title('Performance of Sequential Forward Selection (w. StdErr')
        plt.grid()
        plt.show()

        # loop through all combinations of features selected by the sfs (or other algorithm)
        # and log the performance metrics

        


        # define sample size n_k, number of features k_k, and rsquared r2_k
        n_k = len(y_train)
        k_k = len(X_train.columns)
        r2_k = r2_score(y_train, self.reg.predict(X_train))

        

        ######## Apply this chunk to each round of models
        # sum of squared residuals
        # ssr = np.sum((self.reg.predict(X_train) - y_train)**2)
        # # define sample size n_k, number of features k_k, and rsquared r2_k
        # n_k = len(y_train)
        # k_k = len(X_train.columns)
        # r2_k = r2_score(y_train, self.reg.predict(X_train))

        # # AIC
        # aic = n_k * np.log(ssr/n_k) + 2 * k_k
        # # BIC
        # bic = n_k * np.log(ssr/n_k) + k_k * np.log(n_k)
        # # Adjusted R2
        # r2adj = 1 - ((n_k - 1)/(n_k - k_k - 1)) * (1 - r2_k)

        # # assign rsquared, adjusted rsquared, AIC, BIC to a dataframe
        # regr_metr = pd.DataFrame({
        #     'r2': [r2_k],
        #     'r2_adj': [r2adj],
        #     'aic': [aic],
        #     'bic': [bic]
        # })
        
        # # assign MLR performance metrics to ouptut dataframe
        # # NOTE to self: may move this so only performance  
        # self.regr_metr_ = regr_metr
        #############

        # # subset features to features chosen by sfs
        # X_train_sfs = sfs_fit.transform(X_train)
        # # perform new linear regression with selected features
        # self.reg_sfs_ = LinearRegression().fit(X_train_sfs, y_train)

        # # retrieve the intercept
        # print('Intercept:') 
        # self.reg.intercept_

        # # retrieve the coeficients
        # print(f' Coefficients:')
        # self.reg.coef_



        # self.results = sm.OLS(y_train, X_train).fit()

        # # VIF dataframe
        # vif_data = pd.DataFrame()
        # vif_data['feature'] = X_train.columns

        # # caclute VIF for each feature
        # vif_data['VIF'] = [vif(X_train.values, i)
        #                     for i in range(len(X_train.columns))]
        
        # # assign VIF dataframe to self object
        # self.vif_data_ = vif_data

        # # calculate Mallows Cp
        # # mallowCp = self.results.ssr/

        # # # assign rsquared, adjusted rsquared, AIC, BIC to a dataframe
        # res_metr = pd.DataFrame({
        #     'r2': [self.results.rsquared],
        #     'r2_adj': [self.results.rsquared_adj],
        #     'aic': [self.results.aic],
        #     'bic': [self.results.bic]
        # })

        # self.res_metr_ = res_metr

    # def lin_regression(self):

        

    def lasso_regression(self,
                        alpha_in = 1,
                        max_iter_in = 1000,
                        n_splits_in = 10,
                        n_repeats_in = 3,
                        random_state_in = 100):
        # using info from here as guide: 
        # https://machinelearningmastery.com/lasso-regression-with-python/
        """
        Parameters
        -------------
        alpha_in: float [0:Inf)
            parameter controlling strength of L1 penalty.
            alpha = 0 same as LinearRegression, but advised not to use alpha = 0
        max_iter_in: integer [0:Inf)
            Maximum number of iterations
            Default is 1000
        n_splits_in: integer
            Nunmber of folds for k-fold CV
        n_repeats_in: integer
            Number of times to repreat k-fold CV
        random_state_in = random seed for stochastic processes (helps ensure reproducability)

        Attributes
        -------------
        lasso_scores_: numpy.ndarray
            scores from cross validation
        features_keep_: pandas DataFrame
            features with non-zero coefficients and the coefficients
        """
        try:
            X_train, y_train = self.expl_vars_tr_, self.resp_var
        except:
            X_train, y_train = self.expl_vars, self.resp_var

        self.lasso_reg = Lasso(alpha = alpha_in,
                            max_iter = max_iter_in).fit(X_train, y_train)

        # define k-fold model
        cv = RepeatedKFold(n_splits = n_splits_in, n_repeats = n_repeats_in, random_state = random_state_in)

        #evaluate model
        scores = cross_validate(self.lasso_reg, 
            X_train, y_train, 
            scoring = ('neg_root_mean_squared_error', 
            'neg_mean_absolute_error', 'r2'), cv = cv, n_jobs = 1,
            return_train_score = True,
            return_estimator = False)

        self.lasso_scores_ = scores

        # return variables with non-zero coeficients
        self.lasso_features_keep_ = pd.DataFrame({
            'features':  self.lasso_reg.feature_names_in_[
                            np.where(np.abs(self.lasso_reg.coef_) > 10e-20)
                            ],
            'coefficients': self.lasso_reg.coef_[
                            np.where(np.abs(self.lasso_reg.coef_) > 10e-20)
                            ]
        })

        rmse_train = -self.lasso_scores_['train_neg_root_mean_squared_error']
        rmse_test = -self.lasso_scores_['test_neg_root_mean_squared_error']
        mae_train = -self.lasso_scores_['train_neg_mean_absolute_error']
        mae_test = -self.lasso_scores_['test_neg_mean_absolute_error']
        print('Mean CV training RMSE (stdev): %.3f (%.3f)' % (np.mean(rmse_train), np.std(rmse_train)))
        print('Mean CV testing RMSE (stdev): %.3f (%.3f)' % (np.mean(rmse_test), np.std(rmse_test)))
        print('Mean CV training MAE: %.3f (%.3f)' % (np.mean(mae_train), np.std(mae_train)))
        print('Mean CV testing MAE: %.3f (%.3f)' % (np.mean(mae_test), np.std(mae_test)))
    
# %%
