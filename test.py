#target = __import__("PoissonJoint.py")
from QLD.PoissonJoint import *
import pandas  as pd
df = pd.read_csv("test/plate.csv")
sol = quantifyInputFromSerialDilution(df, foldDilution=10)
print(sol)


