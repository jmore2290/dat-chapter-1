
def listConvert(alist =[],*args):
    list5 = []
    count = 0
    for x in alist:
        mhoper = str(alist[count])
        if x % 33 == 0 :
           list5.append(x)
           count = count + 1
            
        elif "2" in mhoper:
               list5.append(x)
               count = count + 1
        else:
           count = count + 1
    return list5
