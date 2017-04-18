import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def reachDist(df,MinPts):
    """ calculates the reach distance of each point to MinPts around it
    df - dataframe
    MinPts - number of nearest neighbours
    """
    nbrs = NearestNeighbors(n_neighbors=MinPts)
    nbrs.fit(df)
    distancesMinPts, indicesMinPts = nbrs.kneighbors(df)
    for i in range(distancesMinPts.shape[1]):
        distancesMinPts[:,i] = np.amax(distancesMinPts,axis=1)
    return distancesMinPts, indicesMinPts


def lrd(MinPts,knnDistMinPts):
    """ calculates the Local Reachability Density
    knndistMinPts - distances of knn 
    MinPts -  number of nearest neighbours   
    """
    return (MinPts/np.sum(knnDistMinPts,axis=1))


def lof(Ird,MinPts,dsts):
    """ lof calculates lot outlier scores
    Ird - matrix of lrd
    MinPts -  number of nearest neighbours   
    """    
    lof=[]
    for item in dsts:
       tempIrd = np.divide(Ird[item[1:]],Ird[item[0]])
       lof.append(tempIrd.sum()/MinPts)
    return lof


def returnFlag(x):
    if x['Score']>1.3:
       return 1
    else:
       return 0


def run(input_path, output_path="results/results.csv", m=15):

    data = pd.read_csv(input_path, sep=",", header=None)
    # data = pd.read_csv('data/ondr_preprocessed.csv', sep=",", header=None)

    ##### for spark start 
    # from pyspark import SparkContext
    # from pyspark import SparkConf
    # conf = SparkConf().setAppName("LOF")
    # sc = SparkContext(conf=conf)#conf=conf
    # data = sc.textFile("hdfs://hador-cluster/user/shakukse/lamost_small.csv")
    # data = data.map(lambda x: (x.split(",")))
    # data = data.take(data.count())#30000)#data.count())
    # names =  map( lambda x: x[0], data)
    # features = map (lambda x: np.array([float(n) for n in x[1:-1]]) , data)    
    # df1 = pd.DataFrame(names)
    # df2 = pd.DataFrame(features)
    # data = pd.merge(df1, df2,left_index=True,right_index=True)
    ##### for spark end


    columns_number = len(data.columns)
    reachdist, reachindices = reachDist(data.ix[:,1:columns_number] ,m)#,knndist)

    print ("REACH DISTANCES ARE READY")

    irdMatrix = lrd(m,reachdist)
    print ("LDR IS READY")

    lofScores = lof(irdMatrix,m,reachindices)
    print ("LOF IS READY")
    scores= pd.DataFrame(lofScores,columns=['Score'])

    data = pd.DataFrame(data)
    mergedData=pd.merge(data,scores,left_index=True,right_index=True)

    mergedData['flag'] = mergedData.apply(returnFlag, axis=1)
    Outliers = mergedData[(mergedData['flag']==1)]
    Normals = mergedData[(mergedData['flag']==0)]

    print ("Writing all scores, but outliers are " + str(Outliers.shape)) 
    # #### for spark start
    # sc.parallelize(mergedData['Score'].tolist()).saveAsTextFile("hdfs://hador-cluster/user/shakukse/lof_st.csv")
    # sc.parallelize(names).saveAsTextFile("hdfs://hador-cluster/user/shakukse/lof_st_names.csv")
    # #### for spark end
    Outliers.to_csv(output_path+".outliers", header=None, index=False)
    mergedData.to_csv(output_path, header=None, index=False)

    print ("OUTPUT IS WRITTEN")
