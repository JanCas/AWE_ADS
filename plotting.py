import pandas as pd
import JansPlottingStuff as JPS
import matplotlib.pyplot as plt


ads_file = "ads_20250703_204944.csv" 
num_fins = 210
mw_water = 18 #g/mol
model_file = "test.txt"


if __name__ == "__main__":
    JPS.apply()
    
    model = pd.read_csv("test.txt", sep=r'\s+', names = ["Time", "Weight"])
    model["Time"] = model["Time"] / 3600
    print(model.head())
    ads_test = pd.read_csv(ads_file, parse_dates=["Time",])
    ads_test["timedelta"] = (ads_test["Time"] - ads_test["Time"][0]).dt.total_seconds() / 3600
    ads_test["weight_per_fin"] = (ads_test["Big Scale"] - ads_test["Big Scale"].min()) / num_fins
    ads_test["mol_per_fin"] = ads_test["weight_per_fin"] / mw_water
    ax = ads_test.plot(x="timedelta", y="mol_per_fin")
    model.plot(x="Time", y="Weight", ax=ax)
    plt.show()
