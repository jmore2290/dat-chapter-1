import csv


# Exercise_3 Question 1 
alist = []
blist = []

with open('repos/rock.csv', 'rb') as csvfile:
          spamreader = csv.reader(csvfile, delimiter = ',', quotechar = "'")
          for row in spamreader:
              if row[2] == '1981':
                 blist.append(row[2])
              
print len(blist)

#Run the following code separately from the top code.

import csv
#Exercise_3 Question 2
alist = []
blist = []

# Next block reads in file and puts columns 0 and 6 of each row in a list called alist.  However, there are some non-numerical 
# entries in some rows of column 6.  Therefore, I found out what those non-numerical values were and omitted them from the list
# formation.

with open('repos/rock.csv', 'rb') as csvfile:
          spamreader = csv.reader(csvfile, delimiter = ',', quotechar = "'")
          for row in spamreader:
              if "Stills" in row[6] or "One Scotch" in row[6] or "Somewhere" in row[6] or "Girls" in row[6]:
                 continue
              else:      
                 alist.append([row[0],row[6]])        

                        
# In order to make the sorted() function actually work on alist.  I had to convert 'playcount' value in each row entry from string
#  to int.
for row in alist[1:]:
    row[1] = int(row[1])

# And finally this line will print out the top 20 songs by playcount.

print sorted(alist[1:], reverse=True, key=lambda count: count[1])         
