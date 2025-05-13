#Read pickled data from a file
import pickle
import pandas as pd
import numpy as np

# Load the data
with open('generated_data/new_GeoFaultBenchmark_Word2VecCos.pkl', 'rb') as f:
    data = np.load(f, allow_pickle=True)

# Display the data
print(data)

#print data type
print(type(data))

# data is of type DataFrame
# dump the data to a csv file
data.to_csv('generated_data/new_GeoFaultBenchmark_Word2VecCos.csv', index=False)
