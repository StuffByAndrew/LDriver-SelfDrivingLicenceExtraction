import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "/home/fizzer/Desktop/enph353DrivingLicenceExtraction/experiments/data/"
col_names = "x_origin", "y_origin", "x_velocity", "y_velocity"
def plot_vectors(filename):
    csv_file = open(path + filename, 'r')
    data = pd.read_csv(csv_file)
    x_origins, y_origins = data["x_origin"].tolist(), data["y_origin"].tolist()
    x_velocities, y_velocities = data["x_velocity"].tolist(), data["y_velocity"].tolist()
   
    plt.quiver(x_origins, y_origins, x_velocities, y_velocities, headaxislength=3, headlength=3, width=0.005)
    plt.show()

if __name__ == "__main__":
    try:
        filename = sys.argv[1]
    except IndexError as e:
        try:
            files = sorted(os.listdir(path))
            filename = files[-1]
        except IndexError as e:
            print("No files in directory {}".format(path))
            sys.exit()
    plot_vectors(filename)