#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#include <Eigen/Dense>
#include <vector>

class HungarianAlgorithm {
public:
    HungarianAlgorithm(const Eigen::MatrixXd& costMatrix);
    Eigen::MatrixXd reduceMatrix(const Eigen::MatrixXd& matrix);
    std::vector<int> solve();

private:
    void step2b();
    void step3();
    bool step4();
    void step5();
    void step6();

    Eigen::MatrixXd costMatrix_;
    Eigen::MatrixXd maskMatrix_;
    std::vector<bool> rowCover_;
    std::vector<bool> colCover_;
    std::vector<std::pair<int, int>> path_;
    int pathRow0_, pathCol0_;
    int numRows_;
    int numCols_;
};

#endif // HUNGARIAN_H
