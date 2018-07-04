import numpy as np
import pandas as pd

df = pd.DataFrame({
    'a': [4, 5, 3, 1, 2],
    'b': [20, 10, 40, 50, 30],
    'c': [25, 20, 5, 15, 10]
})

# Change False to True for this block of code to see what it does

# DataFrame apply() - use case 2
if False:   
    print(df.apply(np.mean))
    print(df.apply(np.max))
    
df = pd.DataFrame(
    {'a': [4, 0, 3, 1, 2], 'b': [20, 10, 40, 0, 30], 'c': [0, 20, 5, 15, 10]},
    index=[0, 1, 2, 3, 4]
)
    
def return_2nd_largest(s):
    # 用这个 .copy() 函数才能传值，而不是传址 - 如果传址会导致输入被改动，进而过不了 Udacity 的作业检查
    s_temp = s.copy()   
    s_temp[s_temp.idxmax()] = 0
    return s_temp.max()

"""
# Udacity 老师的方法
def second_largest_in_column(column):
    sorted_column = column.sort_values(ascending=False, inplace=False)
    return sorted_column.iloc[1]
"""
    
def second_largest(df):
    '''
    Fill in this function to return the second-largest value of each 
    column of the input DataFrame.
    '''
    return df.apply(return_2nd_largest)

print(df)
print(second_largest(df))
print(df["a"].sort_values(ascending=False))