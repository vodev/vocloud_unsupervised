from __future__ import print_function
import sys
import argparse
import json

import numpy as np
import pandas as pandas
import lof.lof_seq  #, lof_par

def parse_args(argv):
    parser = argparse.ArgumentParser("spark_lof")
    parser.add_argument("config", type=str)
    # parser.add_argument("type", type=str, default="seq")
    parsed_args = parser.parse_args(argv)
    with open(parsed_args.config) as in_config:
        conf = json.load(in_config)
   
    # run_type = parsed_args.type
    input_path = conf.get("dataset")#conf.get(input_dataset)
    min_pts = conf.get("min_pts", "1.0")
    output = conf.get("output")
    return input_path, min_pts, output#, run_type


def main (argv):
    input_path, min_pts, output_path = parse_args(argv)
    if input_path is None:
        raise ValueError('Input path is not given in config file')
    lof.lof_seq.run(input_path, output_path, min_pts)
    print ("Finished LOF!")

if __name__ == '__main__':
    main(sys.argv[1:])