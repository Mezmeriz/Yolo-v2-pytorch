import numpy as np
import open3d as o3d
import pandas as pd

def pose(xyz, rot = None):
    m = np.identity(4)
    m[0:3,3] = xyz
    if rot is not None:
        m[0:3, 0:3] = rot
    return m

COLOR_BLUE = [0,0,1]
COLOR_GREEN = [0.1, 0.9, 0.1]
COLOR_RED = [1, 0, 0]

class Chains:

    def __init__(self, df, chainNumber = None):
        self.df = df

        if chainNumber is None:
            self.chainNumber = None
            self.dfChain = None
        else:
            self.setChain(chainNumber)

    def setChain(self, chainNumber):
        self.chainNumber = chainNumber
        self.dfChain = self.df[self.df['chain'] == chainNumber]
        self.make()

    def keyIndex(self, key):
        dfKeys = [i for i in self.df.keys()]
        return dfKeys.index(key)

    def make(self):
        df = self.dfChain
        R = 0.015
        obj = []

        for i in range(len(df)):
            center = df.iloc[i, self.keyIndex('centers')]
            tailFlag, headFlag = df.iloc[i, [self.keyIndex('tailFlag'), self.keyIndex('headFlag')]]
            sphere = o3d.create_mesh_sphere(R, 12).transform(pose(center))
            if tailFlag:
                sphere.paint_uniform_color(COLOR_GREEN)
            elif headFlag:
                sphere.paint_uniform_color(COLOR_RED)
            else:
                sphere.paint_uniform_color(COLOR_BLUE)

            sphere.compute_vertex_normals()
            obj.append(sphere)
        self.obj = obj

    def show(self):
        o3d.draw_geometries(self.obj)

    def showInteractive(self):
        self.custom_draw_geometry_with_key_callback(self.obj)

    def save(self, fileName):
        self.df.to_pickle(fileName)

    def custom_draw_geometry_with_key_callback(self, pcd):

        highlightIndex = 0
        self.obj[highlightIndex].paint_uniform_color(COLOR_GREEN)

        def next(vis):
            nonlocal highlightIndex
            highlightIndex += 1
            self.obj[highlightIndex].paint_uniform_color(COLOR_GREEN)
            vis.update_geometry()
            # vis.poll_events()
            vis.update_renderer()

        key_to_callback = {}
        key_to_callback[ord("N")] = next
        o3d.visualization.draw_geometries_with_key_callbacks(pcd, key_to_callback)


def load(fileName):
    print("Loading {}".format(fileName))
    df = pd.read_pickle(fileName)
    print("Length = {} samples".format(df.shape[0]))
    return Chains(df)

if __name__ == '__main__':
    C = load('tmp/chain981.pkl')
    C.setChain(981)
    C.showInteractive()
