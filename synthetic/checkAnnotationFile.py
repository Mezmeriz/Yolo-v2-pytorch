from synthetic.Annotations import Annotations
import numpy as np

anno = Annotations("../data/TwoShapes/annotations/test1_train.pkl")
print(anno.df.head(20))
print(anno.df.shape[0])
print(np.max(anno.df["imageIndex"]))
print(len(anno))
print(anno[35])

x = anno[35]
y = np.vstack([x["xc"], x["yc"], x["bx"], x["by"], x["catID"]]).T
print(y)