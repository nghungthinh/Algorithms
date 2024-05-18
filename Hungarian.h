#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#include <Eigen/Dense>
#include <vector>

const int INF = std::numeric_limits<int>::max();

class HungarianAlgorithm {
public:
    HungarianAlgorithm(const Eigen::MatrixXd cost_matrix);
    // HungarianAlgorithm();
    void solve(std::vector<int>& assignment, Eigen::MatrixXd &matching_assignment);

private:
    Eigen::MatrixXd _cost_matrix;
    Eigen::MatrixXd _reduced_cost_matrix;
    std::vector<int> _u, _v, _assignment, _path;
    int _rows, _cols;

    void Assigment_Optimize();
    void Update_Labels(int delta, const std::vector<bool>& visited, std::vector<int>& minValues);
    void Augment_Path(int assignedJob);
    void Extract_Assignment(std::vector<int> &assignment_results, Eigen::MatrixXd &matching_assignment);
    Eigen::MatrixXd Reduced_Cost_Matrix(Eigen::MatrixXd cost_matrix);

};

#endif // HUNGARIAN_H
