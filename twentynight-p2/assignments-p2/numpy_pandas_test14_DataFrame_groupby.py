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

# Change False to True for each block of code to see what it does

# Examine DataFrame
if False:
    print(example_df)
    
# Examine groups - 按照某一列的值分组，相同的取值被归为同一组
# 分组后得到的是一个 GroupBy 类型的对象，类似字典类型，每个组对应一个 dataFrame 数据
if False:
    grouped_data = example_df.groupby('even')
    # The groups attribute is a dictionary mapping keys to lists of row indexes
    print(grouped_data.groups)
    
# Group by multiple columns - 按照多列的取值组合则分组，被选中的列，相同的取值组合算同一组
if False:
    grouped_data = example_df.groupby(['even', 'above_three'])
    print(grouped_data.groups)
    
# Get sum of each group
if False:
    grouped_data = example_df.groupby('even')
    print(grouped_data.sum())
    
# Limit columns in result
if False:
    grouped_data = example_df.groupby('even')
    
    # You can take one or more columns from the result DataFrame
    print(grouped_data.sum()['value'])
    
    print('\n') # Blank line to separate results
    
    # You can also take a subset of columns from the grouped data before 
    # collapsing to a DataFrame. In this case, the result is the same.
    print(grouped_data['value'].sum())
    
filename = 'nyc-subway-weather.csv'
subway_df = pd.read_csv(filename)

### Write code here to group the subway data by a variable of your choice, then
### either print out the mean ridership within each group or create a plot.

# Change False to True for each block of code to see what it does

# Standardize each group
if False:
    def standardize(xs):
        return (xs - xs.mean()) / xs.std(ddof=0)
    grouped_data = example_df.groupby('even')
#    print(grouped_data['value'].apply(standardize))
    
# Find second largest value in each group
if False:
    def second_largest(xs):
        sorted_xs = xs.sort(inplace=False, ascending=False)
        return sorted_xs.iloc[1]
    grouped_data = example_df.groupby('even')
#    print(grouped_data['value'].apply(second_largest))

# --- Quiz ---
# DataFrame with cumulative entries and exits for multiple stations
ridership_df = pd.DataFrame({
    'UNIT': ['R051', 'R079', 'R051', 'R079', 'R051', 'R079', 'R051', 'R079', 'R051'],
    'TIMEn': ['00:00:00', '02:00:00', '04:00:00', '06:00:00', '08:00:00', '10:00:00', '12:00:00', '14:00:00', '16:00:00'],
    'ENTRIESn': [3144312, 8936644, 3144335, 8936658, 3144353, 8936687, 3144424, 8936819, 3144594],
    'EXITSn': [1088151, 13755385,  1088159, 13755393,  1088177, 13755598, 1088231, 13756191,  1088275]
})

def get_hourly_en_ex_for_a_station(station_df):
    return station_df.diff()


def get_hourly_entries_and_exits(entries_and_exits):
    '''
    Fill in this function to take a DataFrame with cumulative entries
    and exits and return a DataFrame with hourly entries and exits.
    The hourly entries and exits should be calculated separately for
    each station (the 'UNIT' column).
    
    Hint: Take a look at the `get_hourly_entries_and_exits()` function
    you wrote in a previous quiz, DataFrame Vectorized Operations. If
    you copy it here and rename it, you can use it and the `.apply()`
    function to help solve this problem.
    '''
    data_grouped_by_station = entries_and_exits.groupby("UNIT")
    return data_grouped_by_station["ENTRIESn", "EXITSn"].apply(get_hourly_en_ex_for_a_station)

print(get_hourly_entries_and_exits(ridership_df))
#print(ridership_df.groupby("UNIT").apply(lambda k:k))