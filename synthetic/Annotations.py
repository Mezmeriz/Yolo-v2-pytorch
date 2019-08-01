import pandas as pd
import numpy as np
from pathlib import Path

class Annotations():

    def __init__(self, source, refresh=False):
        self.source = source
        if Path(source).exists() and not refresh:
            self.df = self.load()
        else:
            self.df = pd.DataFrame()

        print('Current size = {}'.format(self.df.shape[0]))

    def save(self):
        self.df.to_pickle(self.source)

    def load(self):
        df = pd.read_pickle(self.source)
        return df

    def nextImageIndex(self):
        """Used for adding images to the dataset"""
        if len(self.df):
            return np.max(self.df['imageIndex'])+1
        else:
            return 0

    def add(self, index, category, catID,  bbox, center, N):
        xc = bbox[0]*N + center[0]
        yc = bbox[1]*N + center[1]
        bx = (bbox[2]-bbox[0])*N
        by = (bbox[3]-bbox[1])*N
        df = pd.DataFrame({
            'imageIndex': index, 'category': pd.Categorical(category,
                                                            categories =['circle', 'rectangle']),
            'catID': catID, 'xc': xc, 'yc': yc, 'bx': bx, 'by': by}, index = [0])

        if self.df.size is None:
            self.df = df
        else:
            self.df = self.df.append(df, ignore_index=True)

    def __getitem__(self, item):
        return self.df[self.df['imageIndex']==item]

    def __len__(self):
        return np.max(self.df['imageIndex'])+1

class AnnotationsCombined(Annotations):

    def __init__(self, path, subdir = None):
        super().__init__(path)

        if subdir is not None:
            self.df = self.df[self.df['type']==subdir]
        self.imagePool = self.df['imageIndex'].unique()
        self.N = self.imagePool.shape[0]

    def add(self, file, files, annoRow, setType):
        index = 1
        category = 2
        catID = 3
        xc = 4
        yc = 5
        bx = 6
        by = 7

        df = pd.DataFrame({
            'filePrefix': pd.Categorical(file, files),
            'imageIndex': annoRow[index], 'category': pd.Categorical(annoRow[category],
                                                            categories =['circle', 'rectangle']),
            'catID': annoRow[catID], 'xc': annoRow[xc], 'yc': annoRow[yc], 'bx': annoRow[bx], 'by': annoRow[by],
        'type' : pd.Categorical(setType, ['train', 'test', 'val'])}, index = [0])

        if self.df.size is None:
            self.df = df
        else:
            self.df = self.df.append(df, ignore_index=True)

    def checkFile(self, file):
        if self.df.shape[0] == 0 or len(self.df[self.df['filePrefix']==file])==0:
            return False
        else:
            return True

    def addSet(self, annoSource, file, files, indicies, setType):
        for ifor in indicies:
            for annoRow in annoSource[annoSource['imageIndex']==ifor].itertuples(name=None):
                self.add(file, files, annoRow, setType)

    def getImagePrefix(self, item):
        return
    def __getitem__(self, item):
        return self.df[self.df['imageIndex']==self.imagePool[item]]

    def __len__(self):
        return self.N

def annotationFile(rootPath, samplePrefix):
    return rootPath.parent / "annotations" / (samplePrefix + "_anno.pkl")

def imageFileFromIndex(rootPath, samplePrefix, ImageIndex):
    return rootPath.parent / "images" / samplePrefix / (samplePrefix + "_{}.png".format(ImageIndex))
