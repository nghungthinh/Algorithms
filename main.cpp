#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "Hungarian.h"

int main()
{
    std::vector<std::vector<double>> cost_values = {{85, 12, 36, 83, 50, 96, 12, 1},
                                                    {84, 35, 16, 17, 40, 94, 16, 52},
                                                    {14, 16, 8, 53, 14, 12, 70, 50},
                                                    {73, 83, 19, 44, 83, 66, 71, 18},
                                                    {36, 45, 29, 4, 61, 15, 70, 47},
                                                    {7, 14, 11, 69, 57, 32, 37, 81},
                                                    {9, 65, 38, 74, 87, 51, 86, 52},
                                                    {52, 40, 56, 10, 42, 2, 26, 36},
                                                    {85, 86, 36, 90, 49, 89, 41, 74},
                                                    {40, 67, 2, 70, 18, 5, 94, 43},
                                                    {85, 12, 36, 83, 50, 96, 12, 1},
                                                    {84, 35, 16, 17, 40, 94, 16, 52},
                                                    {14, 16, 8, 53, 14, 12, 70, 50},
                                                    {73, 83, 19, 44, 83, 66, 71, 18},
                                                    {36, 45, 29, 4, 61, 15, 70, 47},
                                                    {7, 14, 11, 69, 57, 32, 37, 81},
                                                    {9, 65, 38, 74, 87, 51, 86, 52},
                                                    {52, 40, 56, 10, 42, 2, 26, 36},
                                                    {85, 86, 36, 90, 49, 89, 41, 74},
                                                    {40, 67, 2, 70, 18, 5, 94, 43}};


    HungarianAlgorithm hungarian(cost_values);

    std::vector<int> assignment;
    Eigen::MatrixXd matching_assignment;
    double total_cost;

    hungarian.solve(assignment, matching_assignment, total_cost);

    std::cout << "Assignment: ";
    for (const auto& val : assignment)
        std::cout << val << " ";
    std::cout << std::endl;

    std::cout << "Matching Assignment Matrix:\n" << matching_assignment << std::endl;

    std::cout << "Total optimized cost: " << total_cost << std::endl;

    /* @details , Test more case */
    std::vector<std::vector<std::vector<double>>> tests;
    
    tests.push_back({{25,40,35},
                     {40,60,35},
                     {20,40,25}});
    
    tests.push_back({{64,18,75},
                     {97,60,24},
                     {87,63,15}});
    
    tests.push_back({{80,40,50,46}, 
                     {40,70,20,25},
                     {30,10,20,30},
                     {35,20,25,30}});
    
    tests.push_back({{10,19,8,15},
                     {10,18,7,17},
                     {13,16,9,14},
                     {12,19,8,18},
                     {14,17,10,19}});
    
    {
        for (auto& m: tests) {
        HungarianAlgorithm hungarian_temp(m);

        std::vector<int> assignment;
        Eigen::MatrixXd matching_assignment;
        double total_cost;

        hungarian_temp.solve(assignment, matching_assignment, total_cost);

        std::cout << "Assignment: ";
        for (const auto& val : assignment)
            std::cout << val << " ";
        std::cout << "\n";

        std::cout << "Matching Assignment Matrix:\n" << matching_assignment << "\n";

        std::cout << "Total optimized cost: " << total_cost << "\n";
        std::cout << "----------------------------------------------------------\n";
    }
    }
    return 0;
}
