import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../movie-music-dataset.csv')
df = df.dropna()

category_count_map = {}

for index, row in df.iterrows():
    category = row['Category']
    if category not in category_count_map.keys():
        category_count_map[category] = 1
    else:
        category_count_map[category] += 1

keys = list(category_count_map.keys())
vals = list(category_count_map.values())

# OR

frequency = df['Category'].value_counts().to_dict()
keys = list(frequency.keys())
vals = list(frequency.values())

plt.bar(keys, vals)
plt.xlabel('Category Name')
plt.ylabel('Number of Appearances')
plt.title('Frequency of Category Names')
plt.xticks(rotation=90, ha='right')

plt.show()