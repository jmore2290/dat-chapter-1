rlist = []
qlist = []
summ = 0

def fibonaccifourmil():
    count = 2
    holder = 1
    rlist.append(holder)
    rlist.append(count)
    summ = count + holder
    rlist.append(summ)
    while summ < 4000000:
        holder = count
        count = summ
        summ = count + holder
        if summ < 4000000:
           rlist.append(summ)
    return rlist


def checkevens(alist = []):
    for x in alist:
        if x%2 == 0:
            qlist.append(x)
    return qlist







print sum(checkevens(fibonaccifourmil()))
