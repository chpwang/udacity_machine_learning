import pandas as pd

filename = 'nyc-subway-weather.csv'
subway_df = pd.read_csv(filename)

def correlation(x, y):
    '''
    Fill in this function to compute the correlation between the two
    input variables. Each input is either a NumPy array or a Pandas
    Series.
    
    correlation = average of (x in standard units) times (y in standard units)
    
    Remember to pass the argument "ddof=0" to the Pandas std() function!
    '''
    # 将 x 和 y 数列标准化 - standardize ，这样 x 和 y 即使是不同量级，不同单位，也能进行对比了
    stdized_x = (x - x.mean())/x.std(ddof=0) # 调用 std(ddof=0) 可以禁止使用贝塞耳校正系数
    stdized_y = (y - y.mean())/y.std(ddof=0)
    multipled_df = stdized_x * stdized_y
    return multipled_df.mean()

entries = subway_df['ENTRIESn_hourly']
cum_entries = subway_df['ENTRIESn']
rain = subway_df['meanprecipi']
temp = subway_df['meantempi']


print(correlation(entries, rain))
print(correlation(entries, temp))
print(correlation(rain, temp))

print(correlation(entries, cum_entries))