import numpy as np

#Ask how many rows do they want in their Matrix
A_Matrix_Num_Rows = int(input("How many rows do you want in your input matrix: "))
A_matrix_Num_Columns = int(input("How many columns do you want in your input matrix: "))

A_matrix_rows = []

i = 1

while i <= A_Matrix_Num_Rows:
    
    if i % 10 == 1 and i % 100 != 11:
        ordinal_indicator = "st"
    elif i % 10 == 2 and i % 100 != 12:
        ordinal_indicator = "nd"
    elif i % 10 == 3 and i % 100 != 13:
        ordinal_indicator = "rd"
    else:
        ordinal_indicator = "th"

    #Input each row in the matrix
    A_matrix_row_input = input("Please input " + str(i) + ordinal_indicator + " array: ")

    #Check the length of the row
    len_row = len(A_matrix_row_input.split())

    #Show an error if the length of row is not the same as length that the user defined
    if len_row != A_matrix_Num_Columns:
        print("You input the wrong number of columns. Please try again")
        continue

    #Append each row input to the A_matrix_rows
    A_matrix_row_list = [int(x) for x in A_matrix_row_input.split()]
    A_matrix_rows.append(A_matrix_row_list)

    i += 1

#Convert the list to a numpy matrix
A_matrix = np.array(A_matrix_rows)
R_matrix = np.array(A_matrix_rows)


# Print the resulting matrix
print("A Matrix:")
print(A_matrix)


# Number of rows and columns in the augmented matrix
num_rows, num_cols = R_matrix.shape

row_column_diff = num_rows - num_cols


#Turn the R Matrix into a Float since rows will be divided
R_matrix = R_matrix.astype(float)

# Set the display precision to 2 decimal places
np.set_printoptions(precision=2)

#Check for Dependent Columns
Dependent_Columns = []

#Check for Redundant Rows
Null_Rows = []

R_matrix_row = 0
R_matrix_column = 0

# PFind the Row Reduced Echelon Form using Gaussian Elimination
while R_matrix_row + 1 <= num_rows:

    #For skipping of columns
    skip_column = False
    last_column = False

    #Make sure everything in the matrix is rounded off to 2 decimal places
    R_matrix = np.round(R_matrix, 2)

    #Generate ID's on every array in the matrix
    IDs = np.arange(0, num_rows)

    #Create a column vector of ID's
    IDs_column = IDs.reshape(-1, 1)

    #Concatenate ID's column to the matrix
    R_matrix_with_IDs = np.hstack((R_matrix, IDs_column))

    print ("---------------------")
    print ("                     ")
    print ("                     ")
    print ("                     ")
    print ("                     ")
    print ("                     ")
    print ("                     ")

    print("Matrix with ID")
    print(R_matrix_with_IDs)
    print ("                     ")
    print ("                     ")
    print ("                     ")
    print ("                     ")
    print ("                     ")



    

    print(R_matrix_column)
    print(R_matrix_row)
    print("Checking number: " + str(abs(R_matrix[R_matrix_row,R_matrix_column])))
    if abs(R_matrix[R_matrix_row,R_matrix_column]) == 0:

        #Check if this is the last row of the matrix
        if num_rows == R_matrix_row + 1 and num_cols == R_matrix_column + 1:
            Dependent_Columns.append(R_matrix_column)
            break

        if num_rows == R_matrix_row + 1:
            cols_to_check = num_cols - R_matrix_column
            Dependent_Columns.append(R_matrix_column)

            for i in range(cols_to_check):
                col_no_to_check = R_matrix_column + 1
                print("Checking Number: " + str(R_matrix[R_matrix_row, col_no_to_check]))
 
                if R_matrix[R_matrix_row, col_no_to_check] == 0:
                    last_column = True
                    Dependent_Columns.append(col_no_to_check)
                    break
                else:
                    skip_column = True
                    break

        if skip_column == True:
            R_matrix_column = col_no_to_check
            continue

        if last_column == True:
            break



        print("The pivot element is not in this row. Swapping current row with another row that is non-zero")
        print ("                     ")
        print ("                     ")
        print ("                     ")

        

        i = 1
        
        rows_to_check = num_rows - R_matrix_row
        
        for i in range(rows_to_check):
            row_no_to_check = R_matrix_row + i
            print("Checking Number: " + str(R_matrix[row_no_to_check, R_matrix_column]))
            if R_matrix[row_no_to_check, R_matrix_column] == 0:

                if num_rows == row_no_to_check + 1:
                    
                    if num_cols == R_matrix_column + 1:
                        Dependent_Columns.append(R_matrix_column)
                        last_column = True
                        break
                    else:
                        Dependent_Columns.append(R_matrix_column)
                        skip_column = True
                        break
                else:
                    print("skipping row...")
                    continue
            else:
                print("Swapping rows " + str(R_matrix_row) + " with row " + str(row_no_to_check))
                R_matrix[R_matrix_row], R_matrix[row_no_to_check]  = R_matrix[row_no_to_check].copy(), R_matrix[R_matrix_row].copy()

                #Update ID Numbers
                R_matrix_with_IDs = np.hstack((R_matrix, IDs_column))

                print(R_matrix)
                break
    
    #If the iteration at the very end of the matrix, the loop will stop prematurely
    print(skip_column)
    if skip_column == True:
        R_matrix_column += 1
        continue

    if last_column == True:
        break


    print("Matrix column" + str(R_matrix_column))
    print("Matrix Row" + str(R_matrix_row))

    # Simplify the rows by dividing the rows by its respective Pivot Element
    R_matrix[R_matrix_row] = R_matrix[R_matrix_row] / R_matrix[R_matrix_row,R_matrix_column]
    R_matrix_with_IDs[R_matrix_row, :-1] = R_matrix[R_matrix_row] / R_matrix[R_matrix_row,R_matrix_column]
    print(str(R_matrix[R_matrix_row]) + " divided by: " + str(R_matrix[R_matrix_row, R_matrix_column]))

    #Make sure everything in the matrix is rounded off to 2 decimal places
    R_matrix = np.round(R_matrix, 2)
    R_matrix_with_IDs = np.round(R_matrix_with_IDs, 2)

    #Assigning the Pivot Element
    pivot_element = R_matrix[R_matrix_row, R_matrix_column]
    print("pivot elemnt is: " + str(pivot_element))


    #Extract the other rows that is not the pivot row
    row_to_exclude = R_matrix_row
    other_rows= [R_matrix_with_IDs[i] for i in range(len(R_matrix)) if i != row_to_exclude]
    other_rows = np.array(other_rows)

    #Extract the values that is at the same column as the Pivot Element
    other_column_values = other_rows[:, R_matrix_column]
    
    print("row " + str(R_matrix_row + 1) + " :" + str(other_column_values))
    
    #Perform Gaussian Elimination
    for target_value in range(len(other_column_values)):
        
        print("Performing Gaussian Eliminaion")

        #Find Multiple
        common_multiple = pivot_element*other_column_values[target_value]*-1
        if common_multiple == 0:
            continue
       
        
        print("Common Multiple: " + str(common_multiple))
        print("actual value: " + str(other_column_values[target_value]))
        extracted_row = other_rows[target_value, :]
        ID_value = extracted_row[-1]
        other_row_without_id = extracted_row[:-1]

        #Solve for the multiplier 
        multiplier = common_multiple / pivot_element
        second_multiplier = common_multiple / other_column_values[target_value]
        multiplier = round(multiplier, 2)
        second_multiplier = round(second_multiplier, 2)
        print(multiplier)
        print(second_multiplier)

        #Perform Elementary Row Operations
        product_row = R_matrix[R_matrix_row] * multiplier
        second_product_row = other_row_without_id * second_multiplier

        product_row = np.round(product_row, 2)
        second_product_row = np.round(second_product_row,2)

        #Round off checking
        print(product_row)
        print(second_product_row)

        subtracted_row = product_row - second_product_row
        other_rows[target_value] = np.hstack((subtracted_row, ID_value))

    #Stack the Pivot row along with the reduced rows
    R_matrix = np.vstack((R_matrix_with_IDs[R_matrix_row, :], other_rows))
    
    #Extract IDs column for reordering of rows
    IDs = R_matrix[:, -1]

    #Sort the IDs
    Sorted_IDs = np.argsort(IDs)

    #Reorder the rows
    R_matrix = R_matrix[Sorted_IDs]
    R_matrix = R_matrix[:, :-1]

    #Present the Resulting Matrix after the nth iteration
    print("**********************************")
    print("**********************************")
    print("        RESULTING MATRIX          ")
    print(R_matrix)


    if R_matrix_column + 1 == num_cols:
        print("Its working")
        if row_column_diff > 0:
            i = 1
            while i <= row_column_diff:
                Null_Rows.append(num_rows - i)
                print(Null_Rows)
                i += 1
        break
    R_matrix_column = R_matrix_column + 1
    
    if R_matrix_row + 1 == num_rows:
        print("Its working")
        if row_column_diff < 0:
            print("There are more columns than rows")
            i = 1
            while i <= abs(row_column_diff):
                Dependent_Columns.append(num_cols - i)
                print(Dependent_Columns)
                i += 1
        break
    R_matrix_row = R_matrix_row + 1

    #Informing the user that the algorithm will now proceed into looking for a new pivot element
    print("Switching from Column " + str(R_matrix_column) + " to " + str(R_matrix_column +1))
 

print("**********************************")
print("**********************************")
print("         RESULTING MATRIX         ")
print(R_matrix)


#Check all rows if there is a row full of zeroes
i = 0

for i in range(num_rows):
    if i == num_rows:
        break
    extracted_row = R_matrix[i]
    all_zeroes = np.all(extracted_row == 0)
    if all_zeroes:
        Null_Rows.append(i)
        print("Removing row number " + str(i + 1))
    
    

#Delete the rows that are full of zeroes
R_matrix = np.delete(R_matrix, Null_Rows, axis=0)
        
#Show the modified matrix
print ("                     ")
print ("                     ") 
print("modified matrix:")
print(R_matrix) 
print ("                     ")
print ("                     ")    
 
# Identify values in the matrix that are negative zero
negative_zeroes = (R_matrix == -0)

# Replace them with just zeroes
R_matrix[negative_zeroes] = 0

#This is to show the Strang Decomposition formula as well as the matrices of A, C, and R
print("                                  ")
print("                                  ")
print("                                  ")
print("                                  ")
print("                                  ")
print("    THE STRANG DECOMPOSITION      ")
print("__________________________________")
print("                                  ")
print("                                  ")
print("            A = C R               ")
print("  A : ")
print("                                  ")
print(A_matrix)
print("                                  ")
print("  R : ")
print("                                  ")
print(R_matrix)

#Find Matrix C by deleting dependent columns from matrix A
Num_rows = A_matrix.shape[0]
R_num_rows, R_num_columns = R_matrix.shape

#This is the combined list of columns that have to be removed
cols_to_delete_set = set(Dependent_Columns)
cols_to_delete = list(cols_to_delete_set)


#If there is no colummn that needs to be removed, Matrix C would be the same as Matrix A
if not cols_to_delete:
    C_matrix = A_matrix
else:
    C_matrix = np.delete(A_matrix, cols_to_delete, axis=1)


#Print Matrix C
print("                                  ")
print("  C : ")
print("                                  ")
print(C_matrix)

