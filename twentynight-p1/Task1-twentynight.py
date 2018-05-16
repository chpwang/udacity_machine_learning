# coding=utf-8
"""
下面的文件将会从csv文件中读取读取短信与电话记录，
你将在以后的课程中了解更多有关读取文件的知识。
"""
import csv
with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)


"""
任务1：
短信和通话记录中一共有多少电话号码？每个号码只统计一次。
输出信息：
"There are <count> different telephone numbers in the records.
"""

# codes by twentynight - 2018-05-16 23:53:15

tel_numbers = []
message_template = "There are {} different telephone numbers in the records."

# count the number of tel in texts
for i in range(len(texts)):
    tel_numbers.append(texts[i][0])
    tel_numbers.append(texts[i][1])

# count the number of tel in calls
for i in range(len(calls)):
    tel_numbers.append(calls[i][0])
    tel_numbers.append(calls[i][1])

set_of_tel_numbers = set(tel_numbers)


# output message showing the number of tel numbers in records
print(message_template.format(len(set_of_tel_numbers)))