# Use an import statement at the top
import random

word_file = "words.txt"
word_list = []

#fill up the word_list
with open(word_file,'r') as words:
	for line in words:
		# remove white space and make everything lowercase
		word = line.strip().lower()
		# don't include words that are too long or too short
		if 3 < len(word) < 8:
			word_list.append(word)

# Add your function generate_password here
# It should return a string consisting of three random words 
# concatenated together without spaces
print(len(word_list))
def generate_password():
    new_password = ""
    for i in range(3):
        randomly_index = random.randint(0, len(word_list)-1)
        new_password += word_list[randomly_index]
    
    return new_password


"""
# Udacity 的 generate_password() 函数（写法一）
def generate_password():
    return random.choice(word_list) + random.choice(word_list) + random.choice(word_list)

"""

"""
# Udacity 的 generate_password() 函数（写法二）
def generate_password():
    return ''.join(random.sample(word_list,3))

"""


    
# test your function
print(generate_password())