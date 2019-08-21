import numpy as np
import pandas as pd

class Samples():

    def __init__(self):
        self.df = pd.DataFrame()

    def add(self, predictions, spot, vectors):
        predictions = predictions[0]
        for pi in range(len(predictions)):
            pred = predictions[pi]
            sample = {}
            sample['coord'] = [(pred[0], pred[1])]
            sample['bx'] = [pred[2]]
            sample['by'] = [pred[3]]
            sample['objectness'] = [pred[4]]
            sample['class'] = [pred[5]]
            sample['xyz'] = [spot]
            sample['vectors'] = [vectors.flatten()]

            df = pd.DataFrame(sample, index = [0])
            if len(self.df) == 0:
                self.df = df
            else:
                self.df = self.df.append(df, ignore_index = True)

    def save(self, fileName):
        self.df.to_pickle(fileName)

    def load(self, fileName):
        print("Loading {}".format(fileName))
        self.df = pd.read_pickle(fileName)
        print("Length = {} samples".format(self.df.shape[0]))
        return self.df

    def filter(self, classNumber):
        self.df = self.df[self.df['class']==classNumber]

    def filterGreater(self, field, value):
        self.df = self.df[self.df[field]>value]

    def __getitem__(self, item):
        if len(self.df):
            row = self.df.iloc[item]
            coord = (np.array(row['coord']) - 448//2)/448
            vectors = row['vectors'].reshape(3,3)
            xyz = row['xyz']
            radius = (row['bx'] + row['by'])/4.0/448
            center = xyz + vectors[:,1] * (coord[1] + radius) + vectors[:,2] * (coord[0] + radius)
            return (center, radius)

    def __len__(self):
        if len(self.df):
            return self.df.shape[0]
        else:
            return 0
