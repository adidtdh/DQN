import torch
import csv
import pandas as pd
import numpy as np
from time import sleep


from torch.utils import data
from torch.utils.data import Dataset, DataLoader

class Investor():
    def __init__(self, cash) -> None:
        self.money = cash
        self.assets = 0
    
    def buy(self, price):
        #print(f"buying currently: price:{price}, money:{self.money}, assets:{self.assets}")
            self.assets += self.money/price
            self.money = 0

    
    def sell(self, price):
        if self.assets != 0:
            self.money += self.assets * price
            self.assets = 0

    def getWealth(self,price):
        return self.money + self.assets*price
    
def zscore(A):
    mean = torch.mean(A[:-1])
    std = torch.std(A[:-1])
    
    if std == 0:
        std = .1

    A = torch.div(torch.subtract(A, mean), std)


    return A


### Data in file
#   Unix Timestamp          Mod by seconds in day and divided by seconds in day
#   Date                    Discarded
#   Symbol                  Discarded
#   Open                    Z score normalized
#   High                    Z score normalized
#   Low                     Z score normalized
#   Close                   Z score normalized
#   Volume                  Z score normalized

class StockMarketDataset(Dataset):
    def __init__(self, csvfile, view_distance) -> None:
        super().__init__()

        self.view_distance = view_distance

        df = pd.read_csv(csvfile)

        self.data = np.flip(df.to_numpy(), axis=0) # flip the values so it starts at the earliest cronologically
        self.data[:,0] = (self.data[:,0] % 86400)/86400 # mod the unix time by seconds in a day and divide by seconds in a day
        self.data = np.delete(self.data, 2, 1) # delete the ETHUSD colummn
        self.data = np.delete(self.data, 1, 1) # delete the date time column


        self.data = self.data.astype(np.float32)

        self.data = torch.tensor(self.data)


    def __len__(self) -> int:
        return len(self.data) - self.view_distance -1
    
    def __getitem__(self, index):
        values = torch.clone(self.data[index:index + self.view_distance+1])

        price = values[len(values)-2][4].item() # the "Close price" is used as the acutal price,  it kinda works

        values[:,1] = zscore(values[:,1]) # z score normalize Open
        values[:,2] = zscore(values[:,2]) # z score normalize High
        values[:,3] = zscore(values[:,3]) # z score normalize Low
        values[:,4] = zscore(values[:,4]) # z score normalize Close
        values[:,5] = zscore(values[:,5]) # z score normalize Volume

        X = values[0:self.view_distance]
        Y = values[self.view_distance][4]

        ### X format
        # Proportion of day
        # Open
        # High
        # Low
        # Close
        # Volume
        return torch.flatten(X),Y, price


#this is not compatable, do not use
class YahooFinaceDataset(Dataset):
    def __init__(self, csvfile, view_distance) -> None:
        super().__init__()

        self.view_distance = view_distance

        df = pd.read_csv(csvfile)

        self.data = df.to_numpy()

        self.data[:,0] = 0 # set entire column to 0 time
        self.data = np.delete(self.data, 5, 1) # delete the adj close colummn


        self.data = self.data.astype(np.float32)

        self.data = torch.tensor(self.data)


    def __len__(self) -> int:
        return len(self.data) - self.view_distance -1
    
    def __getitem__(self, index):
        values = torch.clone(self.data[index:index + self.view_distance+1])

        price = values[len(values)-2][4].item()

        values[:,1] = zscore(values[:,1]) # z score normalize Open
        values[:,2] = zscore(values[:,2]) # z score normalize High
        values[:,3] = zscore(values[:,3]) # z score normalize Low
        values[:,4] = zscore(values[:,4]) # z score normalize Close
        values[:,5] = zscore(values[:,5]) # z score normalize Volume

        X = values[0:self.view_distance]
        Y = values[self.view_distance][4]

        ### X format
        # Proportion of day
        # Open
        # High
        # Low
        # Close
        # Volume
        return torch.flatten(X),Y, price


### Data provided by simulation(x30):
#   Time    Converted from Unixtime to proportion of day complete from 0:00(0.00) to 24:00(1.00)
#   Open
#   Close
#   High
#   Low
#   Volume
#   
### Labels provided by simulation(x1):
#   Time    Converted from Unixtime to proportion of day complete from 0:00(0.00) to 24:00(1.00)
#   Open
#   Close
#   High
#   Low
#   Volume


    

stock_dataset = StockMarketDataset("./data/ETH_1H.csv", 30)
qqq_dataset = YahooFinaceDataset("./data/SPY.csv", 30)

stock_dataloader = DataLoader(stock_dataset, batch_size=64, shuffle=True)
testing_dataloader = DataLoader(stock_dataset, batch_size=1, shuffle=False)

qqq_dataloader = DataLoader(qqq_dataset, batch_size=1, shuffle=False)

def calculate_profit(model, loader):
    rob = Investor(5000)
    model.eval()
    
    with torch.no_grad():
        for x, y, price in loader:
            x = x.squeeze()
            score = model(x).item()
            zprice = x[4].item()

            if score > zprice:
                rob.buy(price)
            else:
                rob.sell(price)
            #print(f"money:{rob.money}, assets:{rob.assets}, wealth{rob.getWealth(price)}")

    model.train()
    return rob.getWealth(price)


