Presented in solution.py is the solution to Lab 4: exercise_2; summing the even numbers in the Fibonacci sequence as long as they do not exceed four million.  The problem is broken up into 3 methods.  Finbonaccifourmil() returns a list of all the numbers in the Fibonacci sequence that are less than four million.  

Meanwhile, checkevens(alist =[]) check the output of fibonaccifourmil() for any even numbers contained within the list.  It then returns a list of those numbers.

The last statement, print sum(checkevens(fibonaccifourmil())) merely sums the list that is returned from the function checkevens(alist=[])
