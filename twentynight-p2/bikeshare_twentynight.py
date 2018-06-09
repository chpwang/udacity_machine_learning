import time
import pandas as pd
import numpy as np

CITY_DATA = { 'chicago': 'chicago.csv',
              'new york city': 'new_york_city.csv',
              'washington': 'washington.csv' }

pre_PATH = ""

months = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]

weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def get_filters():
    """
    Asks user to specify a city, month, and day to analyze.

    Returns:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    """
    print('Hello! Let\'s explore some US bikeshare data!')
    # TO DO: get user input for city (chicago, new york city, washington). HINT: Use a while loop to handle invalid inputs
    while True:
        try:
            print("Would you like to see data for Chicago, New York City, or Washington?")
            city = input().lower()
        except Exception as e:
            print("Exception occurred: {} \n".format(e))
        else:
            if city in CITY_DATA:
                break
            else:
                print("Invalid Input...Please enter a city name. \n")
                continue

    # TO DO: get user input for month (all, january, february, ... , june)
    while True:
        try:
            print("Would you like to filter by month?")
            print("Type name of the month or type 'all' for no month filter.")
            month = input().lower()
        except Exception as e:
            print("Exception occurred: {} \n".format(e))
        else:
            if month == "all":
                break
            elif month in months:
                break
            else:
                print("Invalid Input...Please enter a name of days of a week. \n")
                continue

    # TO DO: get user input for day of week (all, monday, tuesday, ... sunday)
    while True:
        try:
            print("Would you like to filter by week day?")
            print("Type name of the day(e.g. Sunday) or type 'all' for no day filter.")
            day = input().lower()
        except Exception as e:
            print("Exception occurred: {} \n".format(e))
        else:
            if day == "all":
                break
            elif day in weekdays :
                break
            else:
                message_template = "There's no such day in 「 {} 」 dataset...Please enter another day name. \n"
                print(message_template.format(city))
                continue

    print('-'*40)
    return city, month, day


def load_data(city, month, day):
    """
    Loads data for the specified city and filters by month and day if applicable.

    Args:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    Returns:
        df - Pandas DataFrame containing city data filtered by month and day
    """
    
    # load data file into a dataframe
    df = pd.read_csv(pre_PATH + CITY_DATA[city])

    # convert the Start Time column to datetime
    df['Start Time'] = pd.to_datetime(df['Start Time'])

    # extract month and day of week from Start Time to create new columns
    df['month'] = df['Start Time'].dt.month
    df['day_of_week'] = df['Start Time'].dt.dayofweek


    # convert the Start Time column to datetime
    df['Start Time'] = pd.to_datetime(df['Start Time'])

    # extract month and day of week from Start Time to create new columns
    df['month'] = df['Start Time'].dt.month
    df['day_of_week'] = df['Start Time'].dt.weekday_name


    # filter by month if applicable
    if month != 'all':
        # use the index of the months list to get the corresponding int
        month_code = months.index(month) + 1

        # filter by month to create the new dataframe
        if month_code in df['month'].unique():
            df = df.groupby("month").get_group(month_code)
        else:
            message_template = "There's no {} in 「 {} 」 dataset...Please enter another month name. \n"
            return message_template.format(month.title(), city.title())

    # filter by day of week if applicable
    if day != 'all':
        # filter by day of week to create the new dataframe
        if day.title() in df["day_of_week"].unique():
            df = df.groupby("day_of_week").get_group(day.title())
        else:
            message_template = "There's no {} in 「 {} 」 dataset...Please enter another day name. \n"
            return message_template.format(day.title(), city.title())


    return df


def time_stats(df, month_filter, day_filter):
    """Displays statistics on the most frequent times of travel."""

    print('\nCalculating The Most Frequent Times of Travel...\n')
    start_time = time.time()

    # TO DO: display the most common month
    # if month filer applicable
    if month_filter == 'all':
        # use the index of the months list to get the corresponding int
        month_code = df['month'].mode().iloc[0]
        print("The most common month is: ")
        print("{} \n".format(months[month_code - 1].title()))

    # TO DO: display the most common day of week
    if day_filter == 'all':
        print("The most common day of week is: ")
        print("{} \n".format(df['day_of_week'].mode().iloc[0]))

    # TO DO: display the most common start hour
    df['hour'] = df['Start Time'].dt.hour
    if day_filter == "all":
        print("The most common start hour is: ")
    else:
        print("The most common start hour of {} is: ".format(day_filter))
    print("{} \n".format(df['hour'].mode().iloc[0]))

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def station_stats(df):
    """Displays statistics on the most popular stations and trip."""

    print('\nCalculating The Most Popular Stations and Trip...\n')
    start_time = time.time()

    # TO DO: display most commonly used start station
    print("The most commonly used START station is: ")
    print("「 {} 」\n".format(df['Start Station'].mode().iloc[0]))

    # TO DO: display most commonly used end station
    print("The most commonly used END station is: ")
    print("「 {} 」\n".format(df['End Station'].mode().iloc[0]))

    # TO DO: display most frequent combination of start station and end station trip
    columns_for_grouping = ["Start Station", "End Station"]
    most_freq_trip = df.groupby(columns_for_grouping).size().nlargest(1)   # use .size() and .nlargest() to find the trip
    m_index = most_freq_trip.index                                         # get the stations info from a MultiIndex object
    coordinates = list(zip(*(m_index.labels)))
    start_station = m_index.levels[0][coordinates[0][0]]
    end_station = m_index.levels[1][coordinates[0][1]]

    message_template = "The most frequent trip is from 「 {} 」 to 「 {} 」, \nand there are {} people take this trip."
    print(message_template.format(start_station, end_station, most_freq_trip.iloc[0]))

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def trip_duration_stats(df):
    """Displays statistics on the total and average trip duration."""

    print('\nCalculating Trip Duration...\n')
    start_time = time.time()

    # TO DO: display total travel time
    h, m, s = seconds_to_minutes_and_hours(df["Trip Duration"].sum())
    print("The TOTAL travel time using share bikes is: {} hours {} minutes {} seconds. \n".format(h, m, s))

    # TO DO: display mean travel time
    h, m, s = seconds_to_minutes_and_hours(round(df["Trip Duration"].mean(), 0))
    print("The MEAN travel time using share bikes is: {} hours {} minutes {} seconds.".format(h, m, s))
    print()

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def user_stats(df, city_filter):
    """Displays statistics on bikeshare users."""

    print('\nCalculating User Stats...\n')
    start_time = time.time()

    # TO DO: Display counts of user types
    grouped_df = df.groupby("User Type")
    size_of_g_df = grouped_df.size()
    print("There are {} users in total: ".format(size_of_g_df.sum()))
    for i in range(grouped_df.ngroups):
        print("{}: {}".format(size_of_g_df.index[i], size_of_g_df[i]))
    print("\n")

    # TO DO: Display counts of gender
    if city_filter == "washington":
        print("Dataset of Washington doesn't contain Gender and Birth info")
    else:
        grouped_df = df.groupby("Gender")
        size_of_g_df = grouped_df.size()
        for i in range(grouped_df.ngroups):
            print("{}: {}".format(size_of_g_df.index[i], size_of_g_df[i]))
        print("\n\n")

        # TO DO: Display earliest, most recent, and most common year of birth
        print("The earliest year of birth of the users is: {} \n".format(df["Birth Year"].min()))
        print("The most recent year of birth is: {} \n".format(df["Birth Year"].max()))
        print("The most common year of birth is: {} \n".format(df["Birth Year"].mode().iloc[0]))
    
    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)

def seconds_to_minutes_and_hours(seconds):
    m, s = divmod(int(seconds), 60)    # .divmod() - a single division to produce both the quotient and the remainder
    h, m = divmod(m, 60)
    return h, m, s


def main():
    while True:
        city, month, day = get_filters()
        df = load_data(city, month, day)

        if isinstance(df, str):    # check if df is string - error message
            print(df)
        else:
            time_stats(df, month, day)
            station_stats(df)
            trip_duration_stats(df)
            user_stats(df, city)

        restart = input('\nWould you like to restart? Enter yes or no.\n')
        if restart.lower() != 'yes':
            break


if __name__ == "__main__":
	main()

 