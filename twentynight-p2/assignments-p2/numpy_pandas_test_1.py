import timeit
import pandas as pd

start_time = timeit.default_timer()

daily_engagement = pd.read_csv('daily-engagement-full.csv')

print(len(daily_engagement))
print(len(daily_engagement['acct'].unique()))

end_time = timeit.default_timer()

print('ran for %.2fs'%(end_time - start_time))