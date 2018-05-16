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
任务3:
(080)是班加罗尔的固定电话区号。
固定电话号码包含括号，
所以班加罗尔地区的电话号码的格式为(080)xxxxxxx。

第一部分: 找出被班加罗尔地区的固定电话所拨打的所有电话的区号和移动前缀（代号）。
 - 固定电话以括号内的区号开始。区号的长度不定，但总是以 0 打头。
 - 移动电话没有括号，但数字中间添加了
   一个空格，以增加可读性。一个移动电话的移动前缀指的是他的前四个
   数字，并且以7,8或9开头。
 - 电话促销员的号码没有括号或空格 , 但以140开头。

输出信息:
"The numbers called by people in Bangalore have codes:"
 <list of codes>
代号不能重复，每行打印一条，按字典顺序排序后输出。

第二部分: 由班加罗尔固话打往班加罗尔的电话所占比例是多少？
换句话说，所有由（080）开头的号码拨出的通话中，
打往由（080）开头的号码所占的比例是多少？

输出信息:
"<percentage> percent of calls from fixed lines in Bangalore are calls
to other fixed lines in Bangalore."
注意：百分比应包含2位小数。
"""


# codes by twentynight - 2018-05-17 04:57:56


"""
# Task 3
"""
"""
# Part 1
"""
import re

codes_list = []
message_template1 = "The numbers called by people in Bangalore have codes:"
tel_num_bangalore = re.compile(r"\(080\)")
tel_num_fixline = re.compile(r"\((\d+)\)")
tel_num_telemarketers = re.compile(r"(140).+")
tel_num_mobile = re.compile(r"(\d+)\s(\d+)")


for i in range(len(calls)):
    num_of_caller = calls[i][0]
    num_of_callee = calls[i][1]
    code_value = None

    # see if it's from Bangalore
    if tel_num_bangalore.match(num_of_caller):
        # see if it's fixed line
        code_value = tel_num_fixline.match(num_of_callee)

        if not code_value:
            # see if it's from mobile phone
            code_value = tel_num_mobile.match(num_of_callee)

        elif not code_value:
            # see if it's from Telemarketers
            code_value = tel_num_telemarketers.match(num_of_callee)

    # add code to codes_list if it exists
    if code_value:
        codes_list.append(code_value.group(1))

# sort alphabetically
codes_list_sorted = list(set(codes_list))
codes_list_sorted.sort()

print(message_template1)
for code in codes_list_sorted:
    print(code)


"""
# Part 2
"""
message_template2 = "{} percent of calls from fixed lines in Bangalore are calls to other fixed lines in Bangalore."

ratio_of_bangalore_callee = round(codes_list.count("080")/len(codes_list), 2)

print(message_template2.format(ratio_of_bangalore_callee))




"""
# test output
print(len(codes_list_sorted))
print(len(codes_list))
print("080" in codes_list)
"""

















