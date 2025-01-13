#target = __import__("PoissonJoint.py")
from QLD.PoissonJoint import *
import pandas  as pd
df = pd.read_csv("test/plate.csv")
MLE, lower, upper,  = quantifyInputFromSerialDilution(df, foldDilution=10, maxCellRange=300000, visualize=False)
print(round(MLE), round(lower),round(upper))


