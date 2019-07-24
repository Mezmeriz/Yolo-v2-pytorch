import pandas as pd
import numpy as np
from pathlib import Path

class Annotations():

    def __init__(self, source):
        self.source = source
        if Path(source).exists():
            self.df = self.load()
        else:
            self.df = pd.DataFrame()

        print('Current size = {}'.format(self.df.size))



    def save(self):
        self.df.to_pickle(self.source)

    def load(self):
        df = pd.read_pickle(self.source)
        return df

    def add(self, index, center, bbox):
        df = pd.DataFrame({'imageIndex': index, 'xc': center[0], 'yc': center[1], 'bx': bbox[0], 'by': bbox[1]}, index = [0])
        if self.df.size is None:
            self.df = df
        else:
            self.df = self.df.append(df, ignore_index=True)

    def __getitem__(self, item):
        return self.df[self.df['imageIndex']==item]

if __name__ == '__main__':
    anno = Annotations('foo.pkl')
    for ifor in range(10):
        anno.add(ifor, (10,10), np.array([20,20]))

    print(anno[4])
    print(anno.df)
    # anno.save()
