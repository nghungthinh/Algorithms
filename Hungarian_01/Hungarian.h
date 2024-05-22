#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#include <iostream>
#include <Eigen/Dense>
#include <vector>

const int INF = std::numeric_limits<int>::max();

class HungarianAlgorithm
{
public:
    // Constructor
    HungarianAlgorithm(const std::vector<std::vector<double>> &values);

    // Solve function to get the assignment and matching matrix
    void solve(std::vector<int> &assignment, Eigen::MatrixXd &matching_assignment, double &total_cost);

private:
    int _rows;                          // Number of original rows in the cost matrix
    int _cols;                          // Number of original columns in the cost matrix
    int _square_matrix_size;            // Size of the square matrix after padding
    Eigen::MatrixXd _cost_matrix;       // Original cost matrix (padded to square if necessary)
    Eigen::MatrixXd _reduced_cost_matrix; // Reduced cost matrix

    std::vector<int> _u;                // Potential for workers
    std::vector<int> _v;                // Potential for jobs
    std::vector<int> _assignment;       // Current assignment
    std::vector<int> _path;             // Path for augmenting

    // Function to create a reduced cost matrix
    Eigen::MatrixXd Reduced_Cost_Matrix(const Eigen::MatrixXd &matrix);

    // Function to optimize the assignment
    void Assignment_Optimize();

    // Function to update the labels
    void Update_Labels(int delta, const std::vector<bool> &visited, std::vector<int> &minValues);

    // Function to augment the path
    void Augment_Path(int assignedJob);

    // Function to extract the final assignment
    void Extract_Assignment(std::vector<int> &assignment_results, Eigen::MatrixXd &matching_assignment, double &total_cost);

    // Function to convert vector<vector<double>> to Eigen::MatrixXd(x,x)
    Eigen::MatrixXd create_CostMatrix(const std::vector<std::vector<double>> &values);
};

#endif // HUNGARIAN_H
