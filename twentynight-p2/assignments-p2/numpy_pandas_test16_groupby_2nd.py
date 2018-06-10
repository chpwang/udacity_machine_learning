import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3 
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

# Change False to True for this block of code to see what it does

# groupby() without as_index
if False:
    first_even = example_df.groupby('even').first()
    print(first_even)
    print(first_even['even']) # Causes an error. 'even' is no longer a column in the DataFrame
    
# groupby() with as_index=False
if False:
    first_even = example_df.groupby('even', as_index=False).first()
    print(first_even)
    print(first_even['even']) # Now 'even' is still a column in the DataFrame

filename = 'nyc-subway-weather.csv'
subway_df = pd.read_csv(filename)

## Make a plot of your choice here showing something interesting about the subway data.
## Matplotlib documentation here: http://matplotlib.org/api/pyplot_api.html
## Once you've got something you're happy with, share it on the forums!

# 本来 groupby 会将 "latitude" 和 "longitude" 作为索引值（index），导致它们不再存在于 subway_df 这个 dataFrame 里
# 加上 as_index=False 之后，这两个列仍然会存在于 subway_df 这个 dataFrame 里
# 所以 as_index 的意思就是是否要让「分类项」，此例为  "latitude" 和 "longitude" ，作为索引（index）
data_by_location = subway_df.groupby(["latitude", "longitude"], as_index=False).mean()
print(data_by_location.head())
data_by_location = subway_df.groupby(["latitude", "longitude"]).mean()
print(data_by_location.head())