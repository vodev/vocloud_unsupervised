from __future__ import print_function
from lof.lof import lof
import sys
import argparse
import json

import numpy as np
import pandas as pandas

from pyspark import SparkContext
from pyspark import SparkConf


def parse_args(argv):
    parser = argparse.ArgumentParser("spark_lof")
    parser.add_argument("config", type=str)
    parsed_args =  parser.parse_args(argv)
    with open(parsed_args.config) as in_config:
        conf = json.load(in_config)
   
    # input_dataset = conf.get("dataset", "test")
    # input_path = conf.get("dataset", "hdfs://hador-cluster/user/shakukse/preprocessed.csv")
    input_path = conf.get("dataset")
    min_pts = conf.get("min_pts", "1.0")
    output = conf.get("output")
    return input_path, min_pts, output


def get_features(x):
	return np.array([float(n) for n in x.split(",")[1:-1]])

def get_features_string(x):
	sentence = x.split(",")[1:-1]
	return ','.join(sentence)	

def get_names(x):
	return x.split(",")[0]


def get_pair(x):
    arr = x.split(",")
    return ( arr[0], arr[1:-1])

def clean_data(data):
    paired = data.map(lambda x: get_pair(x))
    paired.groupByKey().values()

def main (argv):
    input_path, min_pts, output = parse_args(argv)
    if input_path == None:
        raise ValueError('Input path is not given in config file')
    if output == None:
        raise ValueError('Output path is not given in config file')  

    conf_spark = SparkConf().setAppName("Lof")
    sc = SparkContext(conf=conf_spark)    

    data = sc.textFile(input_path)
    features = data.map(get_features)
    features_part = features.take(10000)

    outliers = lof.outliers(min_pts, sc.parallelize(features_part))
    names = data.map(get_names).take(data.count())
    outliers_ready = []
    for i in range(len(outliers)):
        if outliers[i] != None:
            print ("Outlier found! " + str(i))
            # outliers_ready.append({"name": names[i], "lof": outliers[i], "index": i} )#, "instance": instance}#, "index": i}
            outliers_ready.append([names[i], outliers[i], i])#, "instance": instance}#, "index": i}

    sc.parallelize(outliers_ready).saveAsTextFile(output)
    print("saved at " + output)
    # df = pd.DataFrame(outliers)
    # df.to_csv("file:///storage/brno2/home/shakukse/vocloud-unsupervised/lof_out.csv", header=None)
    # print ("FINISHED")
    # print (outliers_ready)

if __name__ == '__main__':
    main(sys.argv[1:])