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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV # for hyperparameter tuning
from sklearn.model_selection import RepeatedKFold # for repeated k-fold validation
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn_extra.cluster import KMedoids
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
# import umap # umap
# import hdbscan # hdbscan
# from mlxtend.plotting import heatmap # plotting heatmap


# %% load data

# # water yield directory
dir_WY='E:/DataWorking/USGS_discharge/train_val_test'
# dir_WY='E:/GAGES_Work/AnnualWY'
# # explantory var (and other data) directory
dir_expl='E:/Projects/GAGESii_ANNstuff/Data_Out'
# dir_expl='E:/GAGES_Work/ExplVars_Model_In'

# Annual Water yield
df_train_anWY=pd.read_csv(
    f'{dir_WY}/yrs_98_12/annual_WY/Ann_WY_train.csv',
    dtype={"site_no":"string"}
    )
# df_train_anWY=pd.read_csv(
#     f'{dir_WY}/Ann_WY_train.csv',
#     dtype={"site_no":"string"}
#     )

# mean annual water yield
df_train_mnanWY= df_train_anWY.groupby(
    'site_no', as_index=False
).mean().drop(columns=["yr"])

# GAGESii explanatory vars
df_train_expl=pd.read_csv(
    f'{dir_expl}/ExplVars_Model_In/All_ExplVars_Train_Interp_98_12.csv',
    dtype={'STAID': 'string'}
)

# mean GAGESii explanatory vars
df_train_mnexpl=df_train_expl.groupby(
    'STAID', as_index=False
).mean().drop(columns=['year'])

# vars to color plots with (e.g., ecoregion)
df_ID=pd.read_csv(
    f'{dir_expl}/GAGES_idVars.csv',
    dtype={'STAID': 'string'}
)

df_train_ID=df_ID[df_ID.STAID.isin(df_train_expl.STAID)]


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
        self.clust_vars=clust_vars
        self.id_vars=id_vars
        # self.color_vars=color_vars

    def stand_norm(self, method='standardize', not_tr=[]):
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
        """

        if method == 'standardize':
            stdsc=StandardScaler()
            clust_vars_tr_=pd.DataFrame(
                stdsc.fit_transform(self.clust_vars)
            )
            # give columns names to transformed data
            clust_vars_tr_.columns=self.clust_vars.columns
            # replace transformed vars with untransformed for specified columns
            clust_vars_tr_[not_tr]=self.clust_vars[not_tr]

            self.clust_vars_tr_=clust_vars_tr_

        if method == 'normalize':
            normsc=MinMaxScaler()
            self.clust_vars_tr_=normsc.fit_transform(self.clust_vars)

    
    def k_clust(self, 
                method='kmeans', 
                ki=2, kf=20, 
                plot_mean_sil=True, 
                plot_distortion=True,
                kmed_method='alternate'):
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

        """
        # initialize vectors to hold outputs
        distortions=[]
        ks_out=[]
        silhouette_mean_scores=[]
        silhouette_k_scores=[]

        for i in range(ki, kf + 1):
            if method == 'kmeans':
                km=KMeans(n_clusters=i,
                            init='k-means++',
                            n_init=10,
                            max_iter=300,
                            tol=1e-04,
                            random_state=100)

            if method == 'kmedoids':
                km=KMedoids(n_clusters=i,
                            metric='manhattan',
                            # method='pam',
                            method=kmed_method,
                            init='k-medoids++',
                            max_iter=300,
                            random_state=100)
            # if clust_vars_tr_ exists, use them, else use clust_vars directly
            try:
                km_fit=km.fit(self.clust_vars_tr_)

                # calc individual silhouette scores
                silhouette_vals=silhouette_samples(
                    self.clust_vars_tr_, km_fit.labels_, metric='euclidean'
                )                
            except AttributeError:
                km_fit=km.fit(self.clust_vars)

                # calc individual silhouette scores
                silhouette_vals=silhouette_samples(
                    self.clust_vars, km_fit.labels_, metric='euclidean'
                )

            # calc silhouette scores
            mean_sil=np.array(silhouette_vals).mean()
            mean_sil_k=[]
            for j in np.unique(km_fit.labels_):
                mn_sil=np.array(silhouette_vals[km_fit.labels_ == j]).mean()

                mean_sil_k.append(mn_sil)

            # update vectors
            distortions.append(km.inertia_)
            ks_out.append(km_fit.labels_)
            silhouette_mean_scores.append(mean_sil)
            silhouette_k_scores.append(mean_sil_k)
        
        self.distortions_=distortions
        self.ks_predicted_=ks_out
        self.silhouette_mean_=silhouette_mean_scores
        self.silhouette_k_mean_=silhouette_k_scores

        # if plot_mean_sil=true, plot mean silhouette coef. vs number of clusters
        if plot_mean_sil is True:
            data_in=pd.DataFrame({
                'NumberOfk': range(ki, kf+1),
                'Mean_Sil_Coef': silhouette_mean_scores
            })
            plot=(
                p9.ggplot(data=data_in) +
                p9.geom_point(p9.aes(x='NumberOfk', y='Mean_Sil_Coef')) +
                p9.xlab('Number of clusters') + p9.ylab('Mean Silhouette Coeficient') +
                p9.ggtitle(f'Mean Silhouette values - {method}') +
                p9.scale_x_continuous(breaks=range(ki, kf+1)) +
                p9.theme_minimal()
            )

            print(plot)


        # if plot_distortion=True, plot distortion vs number of clusters
        if plot_distortion is True:
            data_in=pd.DataFrame({
                'NumberOfk': range(ki, kf+1),
                'Distortion': self.distortions_
            })
            plot2=(
                p9.ggplot(data=data_in) +
                p9.geom_line(p9.aes(x='NumberOfk', y='Distortion')) +
                p9.geom_point(p9.aes(x='NumberOfk', y='Distortion')) +
                p9.xlab("Number of clusters") + p9.ylab('Distortion') + 
                p9.ggtitle(f'Elbow Plot - {method}') +
                p9.theme_minimal()
            )

            print(plot2)




    def plot_silhouette_vals(self, k=3, method='kmeans', kmed_method='alternate'):
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
            km=KMeans(n_clusters=k,
                        init='k-means++',
                        n_init=10,
                        max_iter=300,
                        tol=1e-04,
                        random_state=100)

        if method == 'kmedoids':
            km=KMedoids(n_clusters=k,
                        metric='manhattan',
                        # method='pam',
                        method=kmed_method,
                        init='k-medoids++',
                        max_iter=300,
                        random_state=100)
        try:
            km_fit=km.fit(self.clust_vars_tr_)
            # calc individual silhouette scores
            silhouette_vals=silhouette_samples(
                self.clust_vars_tr_, km_fit.labels_, metric='euclidean'
            )
            
        except:
            km_fit=km.fit(self.clust_vars)
            # calc individual silhouette scores
            silhouette_vals=silhouette_samples(
                self.clust_vars, km_fit.labels_, metric='euclidean'
            )


        # create dataframe for use in plotting
        self.sil_datain_=pd.DataFrame({
            'k': pd.Series(km_fit.labels_, dtype='category'),
            'Sil_score': silhouette_vals
        }).sort_values(by=['k', 'Sil_score'])
        self.sil_datain_['number']=range(1, len(silhouette_vals)+1, 1)

        # define tic marks
        tics_in=[]
        for i in np.unique(self.sil_datain_.k):
            tics_in.append(
                np.array(self.sil_datain_.number[self.sil_datain_.k == i]).mean()
            )

        # Plot

        plot_out=(
            p9.ggplot(data=self.sil_datain_) +
            p9.geom_bar(p9.aes(x ='number', y= 'Sil_score', color='k'),
                stat='identity') +
            p9.geom_hline(p9.aes(yintercept=np.array(silhouette_vals).mean()),
                linetype='dashed', color='red') +
            p9.scale_x_continuous(breaks=tics_in, 
                labels=np.unique(self.sil_datain_.k)) +
            p9.xlab('Cluster') + p9.ylab('Silhouette Coeficient') +
            p9.coord_flip() +
            p9.theme_minimal()
        )

        print(plot_out)


    def hdbscanner(self,
        min_clustsize=20,
        # min_sample=20
        gm_spantree=True,
        metric_in='euclidean',
        # clust_seleps=0,
        clustsel_meth='leaf'):

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
            for epsilon=0.5, and clusters that emereved at distances greater than 0.5 will
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

        hd_clusterer=hdbscan.HDBSCAN(min_cluster_size=min_clustsize,
                                # min_samples=20, 
                                gen_min_span_tree=gm_spantree,
                                metric=metric_in,
                                # cluster_selection_epsilon=clust_seleps,
                                cluster_selection_method=clustsel_meth)

        # cluster data
        try:
            hd_clusterer.fit(self.clust_vars_tr_)
        except:
            hd_clusterer.fit(self.clust_vars)

        # plot condensed tree
        plt.figure()
        plot2=hd_clusterer.condensed_tree_.plot(select_clusters=True,
                                       selection_palette=sns.color_palette())
        print(plot2)


        # Print number of catchments in each cluster
        clst_lbls=np.unique(hd_clusterer.labels_)

        for i in clst_lbls:
            temp_sum=sum(hd_clusterer.labels_ == i)
            print(f'cluster {i}: sum(clst_lbls == {temp_sum})')

        # assign clusterer object to self
        self.hd_clusterer_=hd_clusterer

        # create dataframe with:
            # cluster labels
            # probabilities (confidence that each point is in the most appropriate cluster)
            # outlier scores (the higher the score, the more likely the point
            # is to be an outlier)

        self.hd_out_=pd.DataFrame({
            'ID': self.id_vars,
            'labels': hd_clusterer.labels_,
            'probabilities': hd_clusterer.probabilities_,
            'outlier_scores': hd_clusterer.outlier_scores_
        })
    
        # Other potential plots of interest

        # plot the spanning tree
        plt.figure()
        plot1=hd_clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                             edge_alpha=0.6,
                                             node_size=1,
                                             edge_linewidth=0.1)
        print(plot1)
        # plot1.show()

        # plot cluster heirarchy
        #clusterer.single_linkage_tree_.plot(cmap='viridis',
        #                                    colorbar=True)
        # condensed tree
        # clusterer.condensed_tree_.plot()
        # look at outliers
        # sns.distplot(clusterer.outlier_scores_,
        #             rug=True)

        # get 90th quantile of outlier scores
        # pd.Series(clusterer.outlier_scores_).quantile(0.9)


        # histogram of cluster probabilities (how strongly staid fits into cluster)
        # plot3=sns.distplot(hd_clusterer.probabilities_,
        #              rug=True)

        # print(plot3)


        
    def umap_reducer(self, 
        nn=3, 
        mind=0.1,
        sprd=1,
        nc=3,
        color_in='blue'):
        
        """
        Parameters
        -------------
        nn: integer; 2 to 100 is reasonable
            n_neighbors: How many points to include in nearest neighbor
            controls how UMAP balances local vs global structure.
            As nn increases focus shifts from local to global structure
            
        mind: float; 
            min_dist: minimum distance allowed to be in the low dimensional representation
                values between 0 and 0.99
            controls how tightly UMAP is allowed to pack points together
            low values of min_dist result in clumpier embeddings which can be useful
            if interested in clustering. Larger values will prevent UMAP from packing
            points together and will focus on the preservation of the broad 
            topological structure.
            
        sprd: integer;
            effective scale of embedding points. In combination with min_dist (mind) this
            determines how clustered/clumped the embedded points are

        nc: integer; 2 to 100 is reasonable
            n_components: number of dimensions (components) to reduce to
        """    
      
        # define umap reducer
        reducer=umap.UMAP(n_neighbors=nn, # 2 to 100
                        min_dist=mind, # 
                        spread= sprd,
                        n_components=nc, # 2 to 100
                        random_state=100,
                        n_epochs=200) # 200 for large datasets; 500 small

        # working input data
        try:
            data_in=self.clust_vars_tr_

        except: 
            data_in=self.clust_vars

        # Scale data to z-scores (number of stdevs from mean)
        # data_in_scld=StandardScaler().fit_transform(data_in)

        # train reducer
        embedding=reducer.fit_transform(data_in)

        # save embeddings as dataframe
        self.df_embedding_=pd.DataFrame.from_dict({
            'Color': color_in,
            'ColSize' : np.repeat(0.1, len(embedding[:, 0]))
            }).reset_index().drop(columns=["index"])

        self.df_embedding_['STAID']=self.id_vars
        for i in range(0, nc):
            self.df_embedding_[f'Emb{i}']=embedding[:, i]

        # Plot embedding

        fig=px.scatter_3d(self.df_embedding_,
                            x='Emb0',
                            y='Emb1',
                            z='Emb2',
                            color='Color',
                            size='ColSize',
                            size_max=10,
                            title=f"nn: {nn}",
                            custom_data=["STAID"]
                            )
        # Edit so hover shows station id
        fig.update_traces(
        hovertemplate="<br>".join([
            "STAID: %{customdata[0]}" 
        ])
        )

        print(fig.show())
        return f'Embedding shape: {embedding.shape}'


# %%
# define list of columns not to transform
# these columns are OHE so already either 0 or 1. 
# for distance metrics, use Manhattan which lends itself to capturing 
not_tr_in=['GEOL_REEDBUSH_DOM_anorthositic', 'GEOL_REEDBUSH_DOM_gneiss',
       'GEOL_REEDBUSH_DOM_granitic', 'GEOL_REEDBUSH_DOM_quarternary',
       'GEOL_REEDBUSH_DOM_sedimentary', 'GEOL_REEDBUSH_DOM_ultramafic',
       'GEOL_REEDBUSH_DOM_volcanic']

test=Clusterer(clust_vars=df_train_mnexpl.drop(columns=['STAID', 'LAT_GAGE', 'LNG_GAGE']),
    id_vars=df_train_mnexpl['STAID'])

test.stand_norm(method='standardize', # 'normalize'
    not_tr=not_tr_in) 

test.k_clust(
    ki=2, kf=20, 
    method='kmeans', 
    plot_mean_sil=True, 
    plot_distortion=True)
    
test.k_clust(
    ki=2, kf=20, 
    method='kmedoids', 
    plot_mean_sil=True, 
    plot_distortion=True,
    kmed_method='alternate')

for i in range(2, 11):
    test.plot_silhouette_vals(k=i)

test.hdbscanner(
    min_clustsize=20,
    gm_spantree=True,
    metric_in=  'euclidean', #'manhattan', #
    clustsel_meth='eom') #'leaf') #

test.umap_reducer(
    nn=10,
    mind=0.01,
    sprd=1,
    nc=20,
    color_in=df_train_ID['AggEcoregion'])
    # df_train_ID['USDA_LRR_Site'])
    # pd.Series(test2.hd_out_.labels, dtype='category')) #df_train_ID['AggEcoregion'])
    # pd.Series(df_train_ID['ECO3_Site'], dtype="category"))# df_train_ID['Class'])

##### 

# new clusetering on umap reduced data
test2=Clusterer(clust_vars=test.df_embedding_.drop(columns=['STAID', 'Color', 'ColSize']),
    id_vars=df_train_mnexpl['STAID'])

test2.k_clust(ki=2, kf=9, 
    method='kmeans', 
    plot_mean_sil=True, 
    plot_distortion=True)

for i in range(2, 9):
    test2.plot_silhouette_vals(k=i)
    
test2.k_clust(
    ki=2, kf=20, 
    method='kmedoids', 
    plot_mean_sil=True, 
    plot_distortion=True,
    kmed_method='alternate')

for i in range(2, 10):
    test2.plot_silhouette_vals(k=i,
        method='kmedoids',
        kmed_method='alternate')

test2.hdbscanner(
    min_clustsize=30,
    gm_spantree=True,
    metric_in=  'euclidean', #'manhattan', #
    clustsel_meth='eom') #'leaf') #

test2.umap_reducer(
    nn=100,
    mind=0.01,
    sprd=1,
    nc=3,
    color_in=df_train_ID['AggEcoregion'])
    # df_train_ID['USDA_LRR_Site'])
    # pd.Series(df_train_ID['ECO3_Site'], dtype="category"))# df_train_ID['Class'])

# %%
# %% define regression class

# Define a regressor class that allows:
# Multiple linear regression with 
# feature selections methods:
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
        self.expl_vars=expl_vars
        # response variable
        self.resp_var=resp_var
        # ID vars (e.g., gauge numbers)
        self.id_var=id_var

    def stand_norm(self, method='standardize', not_tr=[]):
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
            stdsc=StandardScaler()
            expl_vars_tr_=pd.DataFrame(
                stdsc.fit_transform(self.expl_vars)
            )
            # give columns names to transformed data
            expl_vars_tr_.columns=self.expl_vars.columns
            # replace transformed vars with untransformed for specified columns
            expl_vars_tr_[not_tr]=self.expl_vars[not_tr]

            self.expl_vars_tr_=expl_vars_tr_

        if method == 'normalize':
            normsc=MinMaxScaler()
            self.expl_vars_tr_=normsc.fit_transform(self.expl_vars)

    def lin_regression(self):

        """
        Parameters
        -------------


        Attributes
        -------------


        """
        try:
            X_train, y_train=self.expl_vars_tr_, self.resp_var
        except:
            X_train, y_train=self.expl_vars, self.resp_var

        self.reg=LinearRegression().fit(X_train, y_train)

        # retrieve the intercept
        print(self.reg.intercept_)

        # retrieve the coeficients
        print(self.reg.coef_)


    def lasso_regression(self, alpha_in=1):
        # using info from here as guide: 
        # https://machinelearningmastery.com/lasso-regression-with-python/
        """
        Parameters
        -------------
        alpha_in: float [0:Inf)
            parameter controlling strength of L1 penalty.
            alpha=0 same as LinearRegression, but advised not to use alpha=0

        Attributes
        -------------
        

        """
        try:
            X_train, y_train=self.expl_vars_tr_, self.resp_var
        except:
            X_train, y_train=self.expl_vars, self.resp_var

        self.lasso_reg=Lasso(alpha=1).fit(X_train, y_train)

        # define k-fold model
        cv=RepeatedKFold(n_splits=10, n_repeats=3, random_state=100)

        #evaluate model
        scores=cross_val_score(self.lasso_reg, 
            X_train, y_train, 
            scoring='neg_mean_absolute_error', cv=cv, n_jobs=1)

        self.lasso_scores = np.absolute(scores)

        print('Mean MASE: %.3f (%.3f)' % (np.mean(self.lasso_scores), np.std(self.lasso_scores)))
    
# %%
# define list of columns not to transform
# these columns are OHE so already either 0 or 1.

not_tr_in=['GEOL_REEDBUSH_DOM_anorthositic', 'GEOL_REEDBUSH_DOM_gneiss',
       'GEOL_REEDBUSH_DOM_granitic', 'GEOL_REEDBUSH_DOM_quarternary',
       'GEOL_REEDBUSH_DOM_sedimentary', 'GEOL_REEDBUSH_DOM_ultramafic',
       'GEOL_REEDBUSH_DOM_volcanic']

testreg=Regressor(expl_vars=df_train_mnexpl.drop(columns=['STAID']),
    resp_var = df_train_mnanWY['Ann_WY_ft3'],
    id_var = df_train_ID['AggEcoregion'])

testreg.stand_norm(method='standardize', # 'normalize'
    not_tr=not_tr_in) 

testreg.lasso_regression()

# %%
