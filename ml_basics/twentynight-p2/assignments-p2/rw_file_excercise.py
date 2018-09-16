def create_cast_list(filename):
    cast_list = []
    #use with to open the file filename
    with open(filename) as f:
        #use the for loop syntax to process each line
        for line in f:
            #and add the actor name to cast_list
            cast_list.append(line.split(",")[0])

    return cast_list

cast_list = create_cast_list('flying_circus_cast.txt')
for actor in cast_list:
    print(actor)