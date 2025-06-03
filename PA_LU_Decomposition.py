import numpy as np


def GET_MATRIX_ROWS_COLUMNS(matrix):
    # Number of rows and columns in the augmented matrix
    num_rows, num_cols = matrix.shape

    return num_rows, num_cols


def GENERATE_ID_COLUMNS(matrix):
    
    num_rows, num_cols  = GET_MATRIX_ROWS_COLUMNS(matrix)

    
    #Generate ID's on every array in the matrix
    IDs = np.arange(0, num_rows)

    #Create a column vector of ID's
    IDs_column = IDs.reshape(-1, 1)

    #Concatenate ID's column to the matrix
    matrix = np.hstack((matrix, IDs_column))

    print(matrix)
    print(IDs_column)

    return matrix, IDs_column



def REORDER_ROWS_MATRIX(matrix, column_order):

    print(matrix)
    matrix = np.array(matrix)

    #Extract First column for reordering of rows
    column = matrix[:, column_order]

    #Sort the IDs
    column_sorted = np.argsort(column)[::-1]

    #Reorder the rows
    matrix = matrix[column_sorted]
    IDS_columns = matrix[:,-1]

    return matrix, IDS_columns


def CREATE_A_MATRIX(Matrix_Num_Rows, Matrix_Num_Columns):
    
    matrix_rows = []
    i = 1

    while i <= Matrix_Num_Rows:
        
        if i % 10 == 1 and i % 100 != 11:
            ordinal_indicator = "st"
        elif i % 10 == 2 and i % 100 != 12:
            ordinal_indicator = "nd"
        elif i % 10 == 3 and i % 100 != 13:
            ordinal_indicator = "rd"
        else:
            ordinal_indicator = "th"

        #Input each row in the matrix
        matrix_row_input = input("Please input " + str(i) + ordinal_indicator + " array: ")

        #Check the length of the row
        len_row = len(matrix_row_input.split())

        #Show an error if the length of row is not the same as length that the user defined
        if len_row != Matrix_Num_Columns:
            print("You input the wrong number of columns. Please try again")
            continue

        #Append each row input to the A_matrix_rows
        matrix_row_list = [int(x) for x in matrix_row_input.split()]
        matrix_rows.append(matrix_row_list)

        i += 1

    #Convert the list to a numpy matrix
    matrix = np.array(matrix_rows)  
      
    return matrix


def CREATE_PLU_MATRIX(A_matrix):
    
    A_matrix, IDs_column = GENERATE_ID_COLUMNS(A_matrix)
    A_matrix, IDs_column = REORDER_ROWS_MATRIX(A_matrix, 0)

    #Number of rows and columns in the augmented matrix

    num_rows, num_cols = GET_MATRIX_ROWS_COLUMNS(A_matrix)

    #Create P_Matrix
    P_matrix = np.zeros((num_rows, num_rows))
    L_matrix = np.zeros((num_rows, num_rows))
    U_matrix = A_matrix

    for columns in range(num_rows):
        P_matrix[IDs_column[columns], columns] = 1

    for columns in range(num_rows):
        L_matrix[columns, columns] = 1

    #Preparation for Gaussian Elimination


    #Turn the R Matrix into a Float since rows will be divided
    U_matrix = U_matrix.astype(float)
    U_matrix = U_matrix[:, :-1]

    # Set the display precision to 2 decimal places
    np.set_printoptions(precision=2)


    return P_matrix, L_matrix, U_matrix


def GAUSSIAN_ELIMINATION_PLU(matrix):

    matrix_row = 0
    matrix_column = 0

    num_rows, num_cols = GET_MATRIX_ROWS_COLUMNS(matrix)
    P_matrix, L_matrix, matrix = CREATE_PLU_MATRIX(matrix)

    row_column_diff = num_rows - num_cols

    # Find the Lower and Upper Row Echelon Form using Gaussian Elimination
    while matrix_row + 1 <= num_rows:

        #For skipping of columns
        skip_column = False
        last_column = False

        #Make sure everything in the matrix is rounded off to 2 decimal places
        matrix = np.round(matrix, 2)

        matrix_with_IDs, IDs_column = GENERATE_ID_COLUMNS(matrix)
        print(matrix)

        print ("---------------------")
        print ("                     ")
        print ("                     ")
        print ("                     ")
        print ("                     ")
        print ("                     ")
        print ("                     ")

        print("Matrix with ID")
        print(matrix_with_IDs)
        print ("                     ")
        print ("                     ")
        print ("                     ")
        print ("                     ")
        print ("                     ")


        if num_rows == matrix_row + 1:
            break
        

        print(matrix_column)
        print(matrix_row)
        print("Checking number: " + str(abs(matrix[matrix_row,matrix_column])))
        if abs(matrix[matrix_row,matrix_column]) == 0:

            #Check if this is the last row and column of the matrix
            if num_rows == matrix_row + 1 and num_cols == matrix_column + 1:
                break

            if num_rows == matrix_row + 1:
                break



            print("The pivot element is not in this row. Swapping current row with another row that is non-zero")
            print ("                     ")
            print ("                     ")
            print ("                     ")

            

            i = 1
            
            rows_to_check = num_rows - matrix_row
            
            for i in range(rows_to_check):
                row_no_to_check = matrix_row + i
                print("Checking Number: " + str(matrix[row_no_to_check, matrix_column]))
                if matrix[row_no_to_check, matrix_column] == 0:

                    if num_rows == row_no_to_check + 1:
                        
                        if num_cols == matrix_column + 1:
                            last_column = True
                            break
                        else:
                            skip_column = True
                            break
                    else:
                        print("skipping row...")
                        continue
                else:
                    print("Swapping rows " + str(matrix_row) + " with row " + str(row_no_to_check))
                    matrix[matrix_row], matrix[row_no_to_check]  = matrix[row_no_to_check].copy(), matrix[matrix_row].copy()
                    
                    #Swap the columns on the P_matrix and L_matrix as well
                    P_matrix[:, [row_no_to_check, matrix_row]] = P_matrix[:, [matrix_row, row_no_to_check]]
                    L_matrix[:, [row_no_to_check, matrix_row]] = L_matrix[:, [matrix_row, row_no_to_check]]

                    #Update ID Numbers
                    matrix_with_IDs = np.hstack((matrix, IDs_column))

                    print(matrix)
                    break
        
        #If the iteration at the very end of the matrix, the loop will stop prematurely
        print(skip_column)
        if skip_column == True:
            matrix_column += 1
            continue

        if last_column == True:
            break


        print("Matrix column" + str(matrix_column))
        print("Matrix Row" + str(matrix_row))


        #Make sure everything in the matrix is rounded off to 2 decimal places
        matrix = np.round(matrix, 2)
        matrix_with_IDs = np.round(matrix_with_IDs, 2)

        #Assigning the Pivot Element
        pivot_element = matrix[matrix_row, matrix_column]
        print("pivot elemnt is: " + str(pivot_element))


    

        #Extract rows that are above the pivot row
        excluded_rows = [matrix_with_IDs[i] for i in range(0, matrix_row + 1)]

        #Extract the other rows that is below the pivot row
        other_rows = [matrix_with_IDs[i] for i in range(matrix_row + 1, len(matrix))]
        other_rows = np.array(other_rows)

        other_rows_ID = other_rows[:, -1]
        print(other_rows_ID)

        #Extract the values that is at the same column as the Pivot Element
        other_column_values = other_rows[:, matrix_column]
        
        print("row " + str(matrix_row + 1) + " :" + str(other_column_values))

        
        #Perform Gaussian Elimination
        for target_value in range(len(other_column_values)):
            
            print("Performing Gaussian Eliminaion")

            #Find Multiple
            multiplier = other_column_values[target_value] / pivot_element *-1
            
            target_value_ID = int(other_rows_ID[target_value])

            print(target_value_ID)
            
            if multiplier == 0:
                continue
            
            L_matrix[target_value_ID, matrix_column] = multiplier
            print(L_matrix)

            
            print("Multiple: " + str(multiplier))

            extracted_row = other_rows[target_value, :]
            ID_value = extracted_row[-1]
            extracted_row = extracted_row[:-1]





            #Perform Elementary Row Operations
            product_row = matrix[matrix_row] * multiplier

            product_row = np.round(product_row, 2)
            extracted_row = np.round(extracted_row,2)

            #Round off checking
            print(product_row)
            print(extracted_row)

            subtracted_row = product_row + extracted_row
            other_rows[target_value] = np.hstack((subtracted_row, ID_value))

        #Stack the Pivot row along with the reduced rows
        matrix = np.vstack((excluded_rows, other_rows))
        
        #Extract IDs column for reordering of rows
        IDs = matrix[:, -1]

        #Sort the IDs
        Sorted_IDs = np.argsort(IDs)

        #Reorder the rows
        matrix = matrix[Sorted_IDs]

        U_rows, U_columns = GET_MATRIX_ROWS_COLUMNS(matrix)
        column_diff = abs(num_cols - U_columns)
        print(column_diff)
        matrix = matrix[:, :-column_diff]

        #Present the Resulting Matrix after the nth iteration
        print("**********************************")
        print("**********************************")
        print("        RESULTING MATRIX          ")
        print(matrix)


        if matrix_column + 1 == num_cols:
            print("Its working")
            if row_column_diff > 0:
                i = 1
                while i <= row_column_diff:
                    i += 1
            break
        matrix_column = matrix_column + 1
        
        if matrix_row + 1 == num_rows:
            print("Its working")
            if row_column_diff < 0:
                print("There are more columns than rows")
                i = 1
                while i <= abs(row_column_diff):
                    i += 1
            break
        matrix_row = matrix_row + 1

        #Informing the user that the algorithm will now proceed into looking for a new pivot element
        print("Switching from Column " + str(matrix_column) + " to " + str(matrix_column +1))


    return matrix, P_matrix, L_matrix

#Ask how many rows do they want in their Matrix
A_Matrix_Num_Rows = int(input("How many rows do you want in your input matrix: "))
A_matrix_Num_Columns = int(input("How many columns do you want in your input matrix: "))

A_matrix = CREATE_A_MATRIX(A_Matrix_Num_Rows,A_matrix_Num_Columns)
A_unordered_matrix = A_matrix

print(A_matrix)

U_Matrix, P_Matrix, L_Matrix = GAUSSIAN_ELIMINATION_PLU(A_matrix)


#This is to show the Strang Decomposition formula as well as the matrices of A, C, and R
print("                                  ")
print("                                  ")
print("                                  ")
print("                                  ")
print("                                  ")
print("         LU DECOMPOSITION         ")
print("__________________________________")
print("                                  ")
print("                                  ")
print("           P A = L U              ")
print("  P : ")
print("                                  ")
print(P_Matrix)
print("                                  ")
print("  A : ")
print("                                  ")
print(A_unordered_matrix)
print("                                  ")
print("  L : ")
print("                                  ")
print(L_Matrix)
print("                                  ")
print("  U : ")
print("                                  ")
print(U_Matrix)