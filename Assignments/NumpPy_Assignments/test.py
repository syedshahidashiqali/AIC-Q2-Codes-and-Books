import numpy as np

#task3
def function3():
    #extract those numbers from given array. those are must exist in 5,7 Table
    #example [35,70,105,..]
    a = np.arange(1, 100*10+1).reshape((100,10))
    x = a[(a % 5 == 0) & (a % 7 == 0)]
    print(x)
    """
    Expected Output:
     [35,  70, 105, 140, 175, 210, 245, 280, 315, 350, 385, 420, 455,
       490, 525, 560, 595, 630, 665, 700, 735, 770, 805, 840, 875, 910,
       945, 980] 
    """


#task4     >>>>>>>>NNNNNNN>>>>>>>>>>
def function4():
    #Swap columns 1 and 2 in the array arr.
   
    arr = np.arange(9).reshape(3,3)
    
    # arr[:, [0,1]] = arr[:, [1,0]]
    return arr[:, [1, 0, 2]]
    """
    Expected Output:
          array([[1, 0, 2],
                [4, 3, 5],
                [7, 6, 8]])
    """ 
# arr = np.arange(9).reshape(3,3)
# print(arr)
# print(arr[:, [1,0]])
# arr = arr.swapaxes(1,0)
# print(arr)
# a = function4()
# print(a)



#task5
def function5():
    #Create a null vector of size 20 with 4 rows and 5 columns with numpy function
   
    z = np.zeros((4,5), dtype="int32")
  
    return z
    """
    Expected Output:
          array([[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]])
    """ 


#task6
def function6():
    # Create a null vector of size 10 but the fifth and eighth value which is 10,20 respectively
   
    arr = np.zeros(10); arr[4] =10; arr[7] = 20
    return arr





#task7
def function7():
    #  Create an array of zeros with the same shape and type as X. Dont use reshape method
    x = np.arange(4, dtype=np.int64)
    
    return np.zeros_like(x)

    """
    Expected Output:
          array([0, 0, 0, 0], dtype=int64)
    """ 
# a = function7()
# print(a)



#task8
def function8():
    # Create a new array of 2x5 uints, filled with 6.
    
    x = np.arange(10,dtype="uint32").reshape(2,5)
    x[:] = 6
    return x

    """
     Expected Output:
              array([[6, 6, 6, 6, 6],
                     [6, 6, 6, 6, 6]], dtype=uint32)
    """ 

# x = np.full((2,5), 6)
# print(x)
# x[:] = 6

#task9
def function9():
    # Create an array of 2, 4, 6, 8, ..., 100.
    
    a = np.arange(2,101, 2)
  
    return a

    """
     Expected Output:
              array([  2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,  26,
                    28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,  52,
                    54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,  78,
                    80,  82,  84,  86,  88,  90,  92,  94,  96,  98, 100])
    """ 


#task10         >>>>>>>> VVVV IMPORTANT  >>>>>>>>>>
def function10():    # arr - brr
    # Subtract the 1d array brr from the 2d array arr, such that each item of brr subtracts 
    # ..from respective row of arr.
    
    arr = np.array([[3,3,3],[4,4,4],[5,5,5]])
    brr = np.array([1,2,3])
    # subt = # write your code here 
  
    return subt

    """
     Expected Output:
               array([[2 2 2]
                      [2 2 2]
                      [2 2 2]])
    """ 

"""
arr =                               |   brr =
([                                  |   ([1,2,3])
[3,3,3],                            |
[4,4,4],                            |
[5,5,5]                             |
])                                  |
"""
# each item of brr subtracts from respective row of arr
arr = np.array([[3,3,3],[4,4,4],[5,5,5]])
brr = np.array([1,2,3])
# subt = np.subtract(arr, brr[:,None])
# print(subt)
# print(brr[:,None])    # none is used to add new axis, 1d will become 2d, np.newaxis=1

#task11
def function11():
    # Replace all odd numbers in arr with -1 without changing arr.
    
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ans = np.where(arr % 2 == 1, -1, arr).copy()
  
    return ans

    """
     Expected Output:
              array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])
    """ 
# arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# ans = np.where(arr % 2 == 1, -1, arr).copy()
# print(ans)
# print(arr)


#task12
def function12():
    # Create the following pattern without hardcoding. Use only numpy functions and the below input array arr.
    # HINT: use stacking concept
    
    arr = np.array([1,2,3])
    ans = np.hstack(((np.repeat(arr,3,axis=0)), (np.tile(arr,3))))
  
    return ans

    """
     Expected Output:
              array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
    """ 
# a = function12()
# print(a)

#task13
def function13():
    # Set a condition which gets all items between 5 and 10 from arr.
    
    
    arr = np.array([2, 6, 1, 9, 10, 3, 27])
    ans = arr[(arr > 5) & (arr < 10)]
  
    return ans

    """
     Expected Output:
              array([6, 9])
    """ 




##################################################  Dont Remove any comments #############################################
#task14
def function14():
    # Create an 8X3 integer array from a range between 10 to 34 such that the difference between each element is 1 and then Split the array into four equal-sized sub-arrays.
    # Hint use split method
    
    
    arr = numpy.arange(10, 34, 1).reshape(8,3) #write reshape code
    ans = np.split(arr, 4) #write your code here 
  
    return ans

    """
     Expected Output:
       [array([[10, 11, 12],[13, 14, 15]]), 
        array([[16, 17, 18],[19, 20, 21]]), 
        array([[22, 23, 24],[25, 26, 27]]), 
        array([[28, 29, 30],[31, 32, 33]])]
    """ 
# arr = np.arange(10, 34, 1).reshape(8,3) #write reshape code
# ans = np.split(arr, 4)
# print(ans)

#task15       >>>>>>>>>>>>>>>Not Done >>>>>>>>>>>>>>>>>>
def function15():
    #Sort following NumPy array by the second column
    
    
    arr = np.array([[ 8,  2, -2],[-4,  1,  7],[ 6,  3,  9]])
    # ans = #write your code here 
  
    return ans

    """
     Expected Output:
           array([[-4,  1,  7],
                   [ 8,  2, -2],
                   [ 6,  3,  9]])
    """ 
#Sort following NumPy array by the second column
# arr = np.array([[ 8,  2, -2],[-4,  1,  7],[ 6,  3,  9]])
# print(arr)
# print("=====================")
# ans=arr[arr[:, 1].argsort()]
# print(ans)


"""     >>>>> What is wrong I am doing ? >>>>>>>
Actual Output:
   array[[-4  1 -2]
         [ 6  2  7]
         [ 8  3  9]]
"""

##################################################

#task16
def function16():
    #Write a NumPy program to join a sequence of arrays along depth.
    
    
    x = np.array([[1], [2], [3]])
    y = np.array([[2], [3], [4]])
    ans = np.dstack((x,y)) #write your code here 
  
    return ans

    """
     Expected Output:
                [[[1 2]]

                 [[2 3]]

                 [[3 4]]]
    """

#Task17
def function17():
    # replace numbers with "YES" if it divided by 3 and 5
    # otherwise it will be replaced with "NO"
    # Hint: np.where
    arr = np.arange(1,10*10+1).reshape((10,10))
    return np.where((arr % 3 == 0) & (arr % 5 == 0), "YES", "NO")   # Write Your Code HERE

#Excpected Out
"""
array([['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],
       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO']],
      dtype='<U3')
"""


#Task18     <<<<<<<<< Important >>>>>>>>>>>>>
def function18():
    # count values of "students" are exist in "piaic"
    piaic = np.arange(100)
    students = np.array([5,20,50,200,301,7001])
    x = np.count_nonzero(np.intersect1d(piaic, students)) # Write you code Here
    return x

    #Expected output: 3




# Task19
def function19():
    #Create variable "X" from 1,25 (both are included) range values
    #Convert "X" variable dimension into 5 rows and 5 columns
    #Create one more variable "W" copy of "X" 
    #Swap "W" row and column axis (like transpose)
    # then create variable "b" with value equal to 5
    # Now return output as "(X*W)+b:

    X = np.arange(1,26).reshape(5,5)  # Write your code here
    W = np.copy(X).transpose()  # Write your code here 
    b = 5  # Write your code here
    output = (X*W)+b   # Write your code here

    #expected output
    """
    array([[  6,  17,  38,  69, 110],
       [ 17,  54, 101, 158, 225],
       [ 38, 101, 174, 257, 350],
       [ 69, 158, 257, 366, 485],
       [110, 225, 350, 485, 630]])
    """



#Task20
def fucntion20():
    #apply fuction "abc" on each value of Array "X"
    x = np.arange(1,11)
    def xyz(x):
        return x*2+3-2

    return xyz(x) #Write your Code here
#Expected Output: array([ 3,  5,  7,  9, 11, 13, 15, 17, 19, 21])
