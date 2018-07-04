
def create_groups(items, num_groups):
    groups = []
    try:
        size = len(items) // num_groups
    except:
        print("WARNING: Returning empty list. Please use a nonzero number.")
    else:
        for i in range(0, len(items), size):
            groups.append(items[i:i + size])

    print("{} groups returned.".format(num_groups))
    return groups
        

print("Creating 6 groups...")
for group in create_groups(range(32), 6):
    print(list(group))

print("\nCreating 0 groups...")
for group in create_groups(range(32), 0):
    print(list(group))