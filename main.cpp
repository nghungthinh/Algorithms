#include <iostream>
#include <Eigen/Dense>
#include "Hungarian.h"

int main()
{
    Eigen::MatrixXd test(3, 3);
    Eigen::MatrixXd matching_assignment(3, 3);

    // test << 2500, 4000, 3500,
    //         4000, 6000, 3500,
    //         2000, 4000, 2500;
    test << 1500, 4000, 4500,
            2000, 6000, 3500,
            2000, 4000, 2500;
    std::vector<int> assigment;

    HungarianAlgorithm hungarian(test);
    hungarian.solve(assigment, matching_assignment);

    std::cout << matching_assignment << "\n";    

    return 0;
}