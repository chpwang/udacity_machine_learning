
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
任务2: 哪个电话号码的通话总时间最长? 不要忘记，用于接听电话的时间也是通话时间的一部分。
输出信息:
"<telephone number> spent the longest time, <total time> seconds, on the phone during
September 2016.".

提示: 建立一个字典，并以电话号码为键，通话总时长为值。
这有利于你编写一个以键值对为输入，并修改字典的函数。
如果键已经存在于字典内，为键所对应的值加上对应数值；
如果键不存在于字典内，将此键加入字典，并将它的值设为给定值。
"""

# codes by twentynight - 2018-05-16 23:55:15

"""
# Task 2
"""

calls_dict = dict()
message_template = "{} spent the longest time, {} seconds, on the phone during September 2016."

# count the seconds of each tel numbers spent and store as a dictionary - {key: tel_num, value: time}
for i in range(len(calls)):
    tel_num1 = calls[i][0]
    tel_num2 = calls[i][1]
    spent_time = calls[i][0][3]
    calls_dict[tel_num1] = calls_dict.get(tel_num1, 0) + int(spent_time)
    calls_dict[tel_num2] = calls_dict.get(tel_num2, 0) + int(spent_time)

# find the tel number that spent longest time
tel_num_with_longest_time = max(calls_dict.keys(), key=(lambda k: calls_dict[k]))


# print(len(calls_dict.items()))
# print(tel_num_with_longest_time)
print(message_template.format(tel_num_with_longest_time, calls_dict[tel_num_with_longest_time]))












