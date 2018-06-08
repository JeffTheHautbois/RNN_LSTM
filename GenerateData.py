import pandas as pd
import numpy as np
from numpy import random
#pd generate timestamp obj
rng = pd.date_range (start='1/1/2018', periods=145, freq="D")

cost = []
#hacky generate transcact history
for i in range (0,145):
    if (rng[i].dayofweek == 0):
        price = 2.29 + i/100
        cost.append(round(price,2))
    elif (rng[i].dayofweek == 1):
        price = 10.50 + random.randint(low = 3, high = 7)
        cost.append(round(price,2))
    elif (rng[i].dayofweek == 2):
        price = 15 + i/100
        cost.append(round(price,2))
    elif (rng[i].dayofweek == 3):
        price = 6 + random.uniform(low = 1, high = 2)
        cost.append(round(price,2))
    elif (rng[i].dayofweek == 4):
        price = 31.23 + random.uniform(low = 5, high = 10)
        cost.append(round(price,2))
    elif (rng[i].dayofweek == 5):
        price = 4.12 + i/100
        cost.append(round(price,2))
    elif (rng[i].dayofweek == 6):
        price = 63.52 + random.randint(low = 5,high = 10)
        cost.append(round(price,2))
dft = pd.DataFrame(rng,columns= ["date"])
dft["cost"] = pd.DataFrame(cost)
file_name = "spending.csv"
dft.to_csv(file_name,encoding='utf-8', index=False)