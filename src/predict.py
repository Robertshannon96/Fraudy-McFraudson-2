import numpy as np
import pandas as pd
import pickle

from r_eda import DataPipeline
from r_model import MyModel

if __name__ == '__main__':
    infile = open('model.pkl','rb')
    model = pickle.load(infile)