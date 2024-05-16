#include "Hungarian.h"
#include <iostream>
#include <limits>
#include <algorithm>

HungarianAlgorithm::HungarianAlgorithm(const Eigen::MatrixXd& costMatrix)
    : costMatrix_(costMatrix),
      maskMatrix_(Eigen::MatrixXd::Zero(costMatrix.rows(), costMatrix.cols())),
      rowCover_(costMatrix.rows(), false),
      colCover_(costMatrix.cols(), false),
      pathRow0_(-1),
      pathCol0_(-1),
      numRows_(costMatrix.rows()),
      numCols_(costMatrix.cols()) {}

Eigen::MatrixXd HungarianAlgorithm::reduceMatrix(const Eigen::MatrixXd& matrix) {
    Eigen::MatrixXd reduced = matrix;

    // Reduce Rows
    for (int i = 0; i < reduced.rows(); i++) {
        reduced.row(i) -= Eigen::VectorXd::Constant(reduced.cols(), reduced.row(i).minCoeff());
    }

    // Reduce Columns
    for (int j = 0; j < reduced.cols(); j++) {
        reduced.col(j) -= Eigen::VectorXd::Constant(reduced.rows(), reduced.col(j).minCoeff());
    }
    return reduced;
}

std::vector<int> HungarianAlgorithm::solve() {
    // Step 1: Reduce the cost matrix
    costMatrix_ = reduceMatrix(costMatrix_);

    // Steps 2-6: Execute the Hungarian algorithm
    step2b();
    bool done = false;
    while (!done) {
        step3();
        done = step4();
        if (!done) {
            step5();
            step6();
        }
    }

    // Extract the assignment
    std::vector<int> assignment(numRows_, -1);
    for (int i = 0; i < numRows_; ++i) {
        for (int j = 0; j < numCols_; ++j) {
            if (maskMatrix_(i, j) == 1) {
                assignment[i] = j;
                break;
            }
        }
    }
    return assignment;
}

void HungarianAlgorithm::step2b() {
    for (int i = 0; i < maskMatrix_.rows(); ++i) {
        for (int j = 0; j < maskMatrix_.cols(); ++j) {
            if (costMatrix_(i, j) == 0 && !rowCover_[i] && !colCover_[j]) {
                maskMatrix_(i, j) = 1;
                rowCover_[i] = true;
                colCover_[j] = true;
            }
        }
    }
    rowCover_ = std::vector<bool>(numRows_, false);
    colCover_ = std::vector<bool>(numCols_, false);
}

void HungarianAlgorithm::step3() {
    for (int i = 0; i < numRows_; ++i) {
        for (int j = 0; j < numCols_; ++j) {
            if (maskMatrix_(i, j) == 1) {
                colCover_[j] = true;
            }
        }
    }
}

bool HungarianAlgorithm::step4() {
    int row = -1, col = -1;
    bool done = false;
    while (!done) {
        for (int i = 0; i < numRows_; ++i) {
            for (int j = 0; j < numCols_; ++j) {
                if (costMatrix_(i, j) == 0 && !rowCover_[i] && !colCover_[j]) {
                    row = i;
                    col = j;
                    maskMatrix_(i, j) = 2;
                    if (std::find(maskMatrix_.col(j).data(), maskMatrix_.col(j).data() + numRows_, 1) != maskMatrix_.col(j).data() + numRows_) {
                        int starRow = std::distance(maskMatrix_.col(j).data(), std::find(maskMatrix_.col(j).data(), maskMatrix_.col(j).data() + numRows_, 1));
                        rowCover_[i] = true;
                        colCover_[j] = false;
                        row = starRow;
                        break;
                    } else {
                        pathRow0_ = row;
                        pathCol0_ = col;
                        return false;
                    }
                }
            }
            if (row != -1 && col != -1) break;
        }
        if (row == -1) done = true;
    }
    return done;
}

void HungarianAlgorithm::step5() {
    bool done = false;
    int count = 0;
    path_.push_back({pathRow0_, pathCol0_});

    while (!done) {
        int row = -1, col = -1;
        for (int i = 0; i < numRows_; ++i) {
            if (maskMatrix_(i, path_.back().second) == 1) {
                row = i;
                break;
            }
        }

        if (row != -1) {
            path_.push_back({row, path_.back().second});
        } else {
            done = true;
        }

        if (!done) {
            for (int j = 0; j < numCols_; ++j) {
                if (maskMatrix_(path_.back().first, j) == 2) {
                    col = j;
                    break;
                }
            }
            path_.push_back({path_.back().first, col});
        }
    }

    for (auto& p : path_) {
        if (maskMatrix_(p.first, p.second) == 1) {
            maskMatrix_(p.first, p.second) = 0;
        } else if (maskMatrix_(p.first, p.second) == 2) {
            maskMatrix_(p.first, p.second) = 1;
        }
    }

    rowCover_ = std::vector<bool>(numRows_, false);
    colCover_ = std::vector<bool>(numCols_, false);
    for (int i = 0; i < numRows_; ++i) {
        for (int j = 0; j < numCols_; ++j) {
            if (maskMatrix_(i, j) == 2) {
                maskMatrix_(i, j) = 0;
            }
        }
    }
    path_.clear();
}

void HungarianAlgorithm::step6() {
    double minVal = std::numeric_limits<double>::max();
    for (int i = 0; i < numRows_; ++i) {
        for (int j = 0; j < numCols_; ++j) {
            if (!rowCover_[i] && !colCover_[j]) {
                if (costMatrix_(i, j) < minVal) {
                    minVal = costMatrix_(i, j);
                }
            }
        }
    }

    for (int i = 0; i < numRows_; ++i) {
        for (int j = 0; j < numCols_; ++j) {
            if (rowCover_[i]) costMatrix_(i, j) += minVal;
            if (!colCover_[j]) costMatrix_(i, j) -= minVal;
        }
    }
}
int main() {
    Eigen::MatrixXd costMatrix(4, 4);
    costMatrix << 82, 83, 69, 92,
                  77, 37, 49, 92,
                  11, 69, 5, 86,
                   8, 9, 98, 23;

    HungarianAlgorithm hungarian(costMatrix);

    Eigen::MatrixXd reducedMatrix = hungarian.reduceMatrix(costMatrix);
    std::cout << "Reduced Cost Matrix:\n" << reducedMatrix << std::endl;

    std::vector<int> assignment = hungarian.solve();
    std::cout << "Assignments:" << std::endl;
    for (size_t i = 0; i < assignment.size(); ++i) {
        std::cout << "Row " << i << " assigned to Column " << assignment[i] << std::endl;
    }

    return 0;
}