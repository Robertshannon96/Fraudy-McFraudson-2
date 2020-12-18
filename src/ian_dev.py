import numpy as np
import pandas as pd
import pickle

from eda import DataPipeline

if __name__ == '__main__':
    infile = open('model.py', 'rb')
    model = pickle.load(infile)