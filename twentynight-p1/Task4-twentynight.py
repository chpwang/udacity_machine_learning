
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
任务4:
电话公司希望辨认出可能正在用于进行电话推销的电话号码。
找出所有可能的电话推销员:
这样的电话总是向其他人拨出电话，
但从来不发短信、接收短信或是收到来电


请输出如下内容
"These numbers could be telemarketers: "
<list of numbers>
电话号码不能重复，每行打印一条，按字典顺序排序后输出。
"""

# codes by twentynight - 2018-05-17 05:10:50

"""
# Task 4
"""
callers_tel = []
callees_tel = []
text_senders_tel = []
text_receivers_tel = []
message_template = "These numbers could be telemarketers: "

# store the tel numbers of callers and callees respectively
for i in range(len(calls)):
    callers_tel.append(calls[i][0])
    callees_tel.append(calls[i][1])

# store the tel numbers of text senders and receivers respectively
for i in range(len(texts)):
    text_senders_tel.append(texts[i][0])
    text_receivers_tel.append(texts[i][1])


set_of_callers_tel = set(callers_tel)
set_of_callees_tel = set(callees_tel)

set_of_text_senders_tel = set(text_senders_tel)
set_of_text_receivers_tel = set(text_receivers_tel)


# filter out callees from callers
telemarketer_suspects = set_of_callers_tel.difference(set_of_callees_tel)

# filter out text senders from the result above
telemarketer_suspects = telemarketer_suspects.difference(set_of_text_senders_tel)

# filter out text receivers from the result above
telemarketer_suspects = telemarketer_suspects.difference(set_of_text_receivers_tel)

print(message_template)

tsus = list(telemarketer_suspects)
tsus.sort()

for tel_num in tsus:
    print(tel_num)



"""

print(len(callers_tel))
print(len(set_of_callers_tel))
print(len(set_of_callees_tel))
print(len(texts))
print(len(set_of_text_senders_tel))
print(len(set_of_text_receivers_tel))
"""


"""
### 老师们的做法：

#这里使用set数据类型的话可以更加简洁的完成：

#创建最后结果和需要移除的变量
result = set()
remove = set()

#遍历calls列表
for call in calls:
    result.add(call[0])
    remove.add(call[1])

#遍历texts列表
for text in texts:
    remove.add(text[0])
    remove.add(text[1])
#进行条件筛选
result = result - remove

print("These numbers could be telemarketers: "+ "\n" + "\n".join(sorted(result)))
"""