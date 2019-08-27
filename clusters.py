#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    fingerprints = pd.read_csv("brown_fingerprints.csv")

    fingerprints.plot(x ='fro', y='lat', kind = 'scatter')

    plt.show()
