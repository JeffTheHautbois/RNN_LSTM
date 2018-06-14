import pandas as pd
import numpy as np
from numpy import random
#pd generate timestamp obj
length = 144
rng = pd.date_range (start='1/1/2018', periods=length, freq="D")

def generateLunch():
    cost = []
    for i in range (0,length):
        if (rng[i].dayofweek == 0):
            price = 10.50 + random.uniform(low = 0.5, high = 6)
            cost.append(round(price,2))
        elif (rng[i].dayofweek == 1):
            price = 10.50 + random.randint(low = 3, high = 7)
            cost.append(round(price,2))
        elif (rng[i].dayofweek == 2):
            price = 15 + i/100
            cost.append(round(price,2))
        elif (rng[i].dayofweek == 3):
            price = 8 + random.uniform(low = 1, high = 2)
            cost.append(round(price,2))
        elif (rng[i].dayofweek == 4):
            price = 19.99 + random.uniform(low = 5, high = 10)
            cost.append(round(price,2))
        elif (rng[i].dayofweek == 5):
            price = 5.99 + i/100
            cost.append(round(price,2))
        elif (rng[i].dayofweek == 6):
            price = 22.99 + random.uniform(low = 2,high = 8)
            cost.append(round(price,2))
    return cost

def generateBreakfast():
    cost = []
    for i in range (0,length):
        if (rng[i].dayofweek == 0):
            price = 2.68 + random.randint(low=1, high = 2)
            cost.append(round(price,2))
        elif (rng[i].dayofweek == 1):
            price = 2.15 + random.uniform(low = 1, high = 3)
            cost.append(round(price,2))
        elif (rng[i].dayofweek == 2):
            price = 5.10 + random.uniform(low = 1, high = 2)
            cost.append(round(price,2))
        elif (rng[i].dayofweek == 3):
            price = 4.20 + random.uniform(low = 1, high = 2)
            cost.append(round(price,2))
        elif (rng[i].dayofweek == 4):
            price = 7.34 + random.uniform(low = 3, high = 4)
            cost.append(round(price,2))
        elif (rng[i].dayofweek == 5):
            price = 10.23 + random.uniform(low = 0.2, high = 0.4)
            cost.append(round(price,2))
        elif (rng[i].dayofweek == 6):
            price = 0 + random.uniform(low = 2.20,high = 2.80)
            cost.append(round(price,2))
    return cost

#cost = generateLunch()
cost = generateBreakfast()
dft = pd.DataFrame(rng,columns= ["date"])
dft["cost"] = pd.DataFrame(cost)
file_name = "spending_breakfast.csv"
dft.to_csv(file_name,encoding='utf-8', index=False)

#dataframe = pd.read_csv(file_name, engine='python')
#dataset = dataframe.values
#dataset = dataset.astype('float32')