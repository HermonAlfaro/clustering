class Clustering(CreateData):
    
    def __init__(self,path,idx,clustering_method):
        """
        Class to do clustering, written at begin to use KMeans an AgglomerativeClustering from scikit-learn
        Input: path: path to csv with data
               idx: list of index of the colums to use
               custering_method: 'kmeans' or 'agglomerative_clustering'
        """
        CreateData.__init__(self,path,idx)
        self.nb_cluster = None
        self.ClusterMethod = clustering_method
        self.cl = None
        self.y_pred = None
        self.cluster_labels = None
        
            
    def scaling(self,scaler):
        """
        Input : scaler: clas that do the scaling. Written thinking in use the scaler of scikit-learn,
        but it can be use with the custom class that have the same methods
        """
        sc = scaler()
        self.X = sc.fit_transform(self.X)
        return self.X
    
    def determining_nb_cluster(self,max_nb_cluster):
        """
        method to determine the number of clusters between 1 and max_nb_cluster
        We consider the elbow method for the kmeans
        Input: max_nb_cluster: max number of cluster to do the search
        """
        """
        Doing the elbow method
        """
        nb_clusters = [i for i in range(1,max_nb_cluster+1)]
        wcss = []
        distance = [0.0]
            
        for i in range(len(nb_clusters)):
            cl = KMeans(n_clusters=nb_clusters[i],init='k-means++',random_state=42)
            cl.fit(self.X)
            wcss.append(cl.inertia_)
            
        x1, y1 = nb_clusters[0], wcss[0]
        x2, y2 = nb_clusters[-1], wcss[-1]
        m = (y1-y2)/(x1-x2)
        n = -1/m
            
        for i in range(1,len(nb_clusters)-1):
            xi, yi =  nb_clusters[i], wcss[i]
            x = (1/(m-n))*(-n*xi + m*x1 + yi - y1)
            y = m*(x-x1) + y1
            d = (x-xi)**2 + (y-yi)**2
            d = d**(0.5)
            distance.append(d)
        distance.append(0.0)
            
        # Determining number of cluster that maximize the distance
        max_distance =max(distance)
        self.nb_cluster = nb_clusters[distance.index(max_distance)]
        print(f"Optimal number of clusters by elbow method: {self.nb_cluster}")
            
        if self.ClusterMethod == 'kmeans':
            # Visualizing
            plt.plot(nb_clusters, wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.show()
            
        return self.nb_cluster
        
    def clustering(self,params):
        """
        Input: params: dictionary with the parameters for the specific method
        """
        if self.ClusterMethod == 'kmeans':
            print("Doing the kmeans clustering")
            self.cl = KMeans(n_clusters=self.nb_cluster, init = 'k-means++', random_state=42)
            self.y_pred = self.cl.fit_predict(self.X)
        
        if self.ClusterMethod == 'agglomerative_clustering':
            print("Doing agglomerative hierarchical clustering")
            self.cl = AgglomerativeClustering(n_clusters = self.nb_cluster, **params)
            self.y_pred = self.cl.fit_predict(self.X)
        
        self.cluster_labels = np.unique(self.y_pred)
        print(f"cluster labels : {self.cluster_labels}")
        return self.y_pred
            
    def visualize(self,idx,title,xlabel,ylabel):
        """
        Method to visualize in 2D the clusters
        Input: idx: idex of columns to do the cluster. It has to be a list with two index
        """
        if len(idx) != 2:
            raise ValueError("The len of the list idx has to be 2")
        c = 0
        for i in idx:
            if i in self.idx:
                c += 1
        
        if c != len(idx):
            raise ValueError("These index was not used to do the clustering")
            
        else:
            
            X = self.data.iloc[:,idx].values
            for cluster in self.cluster_labels:
                label = f"cluster {cluster}"
                plt.scatter(X[self.y_pred == cluster,0],X[self.y_pred == cluster,1],label=label)
                if self.ClusterMethod == 'kmeans':
                    plt.scatter(self.cl.cluster_centers_[:,0], self.cl.cluster_centers_[:,1], s=300,c='yellow',
                       label = 'centroids')
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
            plt.show()
            
        if self.ClusterMethod == 'agglomerative_clustering':
            # Dendogram
            mergings = sch.linkage(self.X,method='ward')
            dendogram = sch.dendrogram(mergings)
            plt.title('Dendogram')
            plt.ylabel('Euclidian distances')
            plt.show()
            
    @staticmethod
    def pipeline_clustering(path,idx,clustering_method,scaler,max_nb_cluster,params,title,xlabel,ylabel):
        self = Clustering(path=path,idx=idx,clustering_method=clustering_method)
        if scaler is not None:
            self.scaling(scaler)
        self.determining_nb_cluster(max_nb_cluster)
        self.clustering(params)
        self.visualize(idx,title,xlabel,ylabel)
        
        return self
