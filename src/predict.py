import numpy as np
import pandas as pd

from eda import DataPipeline
from model import MyModel

if __name__ == '__main__':
    infile = open('model.pkl','rb')
    model = pickle.load(infile)