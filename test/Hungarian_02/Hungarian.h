#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>
#include <list>
#include <string>
#include <type_traits>

namespace Munkres {

    /* Utility function to print Matrix */
    template<template <typename, typename...>   class Container, 
                                                typename T,
                                                typename... Args>
    typename std::enable_if<!std::is_convertible<Container<T, Args...>, std::string>::value &&
                    !std::is_constructible<Container<T, Args...>, std::string>::value,
                    std::ostream&>::type
    operator<<(std::ostream& os, const Container<T, Args...>& con);

    /* Handle negative elements if present. If allowed = true, add abs(minval) to 
    * every element to create one zero. Else throw an exception */
    template<typename T>
    void handle_negatives(std::vector<std::vector<T>>& matrix, bool allowed);
    
    /* Ensure that the matrix is square by the addition of dummy rows/columns if necessary */
    template<typename T>
    void pad_matrix(std::vector<std::vector<T>>& matrix);

    /* For each row of the matrix, find the smallest element and subtract it from every 
    * element in its row.  
    * For each col of the matrix, find the smallest element and subtract it from every 
    * element in its col. Go to Step 2. */
    template<typename T>
    void step1(std::vector<std::vector<T>>& matrix, int& step);

    inline void clear_covers(std::vector<int>& cover);

    /* Find a zero (Z) in the resulting matrix.  If there is no starred zero in its row or 
    * column, star Z. Repeat for each element in the matrix. Go to Step 3. */
    template<typename T>
    void step2(const std::vector<std::vector<T>>& matrix, 
                    std::vector<std::vector<int>>& M, 
                    std::vector<int>& RowCover,
                    std::vector<int>& ColCover, 
                    int& step);
    
    /* Cover each column containing a starred zero.  If K columns are covered, the starred 
    * zeros describe a complete set of unique assignments. */
    void step3(const std::vector<std::vector<int>>& M, 
            std::vector<int>& ColCover,
            int& step);

    template<typename T>
    void find_a_zero(int& row, 
                    int& col,
                    const std::vector<std::vector<T>>& matrix,
                    const std::vector<int>& RowCover,
                    const std::vector<int>& ColCover);
    
    bool star_in_row(int row, const std::vector<std::vector<int>>& M);

    void find_star_in_row(int row, int& col, const std::vector<std::vector<int>>& M);

    /* Find a noncovered zero and prime it.  If there is no starred zero in the row containing
    * this primed zero, Go to Step 5.  Otherwise, cover this row and uncover the column 
    * containing the starred zero. Continue in this manner until there are no uncovered zeros
    * left. Save the smallest uncovered value and Go to Step 6. */
    template<typename T>
    void step4(const std::vector<std::vector<T>>& matrix, 
            std::vector<std::vector<int>>& M, 
            std::vector<int>& RowCover,
            std::vector<int>& ColCover,
            int& path_row_0,
            int& path_col_0,
            int& step);

    void find_star_in_col(int c, int& r, const std::vector<std::vector<int>>& M);

    void find_prime_in_row(int r, int& c, const std::vector<std::vector<int>>& M);

    void augment_path(std::vector<std::vector<int>>& path, 
                                                int path_count, 
                                                std::vector<std::vector<int>>& M);

    void erase_primes(std::vector<std::vector<int>>& M);

    /* Construct a series of alternating primed and starred zeros as follows.  
    * Let Z0 represent the uncovered primed zero found in Step 4.  Let Z1 denote the 
    * starred zero in the column of Z0 (if any). Let Z2 denote the primed zero in the 
    * row of Z1 (there will always be one).  Continue until the series terminates at a 
    * primed zero that has no starred zero in its column.  Unstar each starred zero of 
    * the series, star each primed zero of the series, erase all primes and uncover every 
    * line in the matrix.  Return to Step 3. */
    void step5(std::vector<std::vector<int>>& path, 
            int path_row_0, 
            int path_col_0, 
            std::vector<std::vector<int>>& M, 
            std::vector<int>& RowCover,
            std::vector<int>& ColCover,
            int& step);

    template<typename T>
    void find_smallest(T& minval, 
                    const std::vector<std::vector<T>>& matrix, 
                    const std::vector<int>& RowCover,
                    const std::vector<int>& ColCover);
    
    /* Add the value found in Step 4 to every element of each covered row, and subtract it 
    * from every element of each uncovered column.  Return to Step 4 without altering any
    * stars, primes, or covered lines. Notice that this step uses the smallest uncovered 
    * value in the cost matrix to modify the matrix. */
    template<typename T>
    void step6(std::vector<std::vector<T>>& matrix, 
            const std::vector<int>& RowCover,
            const std::vector<int>& ColCover,
            int& step);

    /* Calculates the optimal cost from mask matrix */
    template<template <typename, typename...> class Container,
            typename T,
            typename... Args>
    T output_solution(const Container<Container<T,Args...>>& original,
                    const std::vector<std::vector<int>>& M);
    
    /* Main function of the algorithm */
    template<template <typename, typename...> class Container,
            typename T,
            typename... Args>
    typename std::enable_if<std::is_integral<T>::value, T>::type // Work only on integral types
    hungarian(const Container<Container<T,Args...>>& original,
            bool allow_negatives = true);    

} // namespace Munkres

#endif  // HUNGARIAN_H
