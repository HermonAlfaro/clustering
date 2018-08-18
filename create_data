# Importing the libraries
import pandas as pd

class CreateData:
    
    def __init__(self,path, idx):
        """
        Input: path: path to csv with the data
               idx: list of index of the colums to use  
        """
        self.data = pd.read_csv(path)
        self.X = self.data.iloc[:,idx].values
        self.idx = idx
        
