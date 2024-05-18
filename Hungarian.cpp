#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "Hungarian.h"


HungarianAlgorithm::HungarianAlgorithm(Eigen::MatrixXd cost_matrix)
{
    this->_rows = cost_matrix.rows();
    this->_cols = cost_matrix.cols();
    this->_cost_matrix = cost_matrix;
    this->_u = std::vector<int>(_rows + 1, 0);
    this->_v = std::vector<int>(_cols + 1, 0);
    this->_assignment = std::vector<int>(this->_cols + 1, 0);
    this->_path = std::vector<int>(this->_cols + 1, 0);
    this->_reduced_cost_matrix = Reduced_Cost_Matrix(cost_matrix);
}

void HungarianAlgorithm::solve(std::vector<int>& assignment, Eigen::MatrixXd &matching_assignment)
{
    Assigment_Optimize();
    Extract_Assignment(assignment, matching_assignment);
}

Eigen::MatrixXd HungarianAlgorithm::Reduced_Cost_Matrix(Eigen::MatrixXd reduced_matrix)
{
    // Reduce Rows
    for (int i = 0; i < reduced_matrix.rows(); i++) {
        reduced_matrix.row(i) -= Eigen::VectorXd::Constant(reduced_matrix.cols(), reduced_matrix.row(i).minCoeff());
    }

    // Reduce Columns
    for (int j = 0; j < reduced_matrix.cols(); j++) {
        reduced_matrix.col(j) -= Eigen::VectorXd::Constant(reduced_matrix.rows(), reduced_matrix.col(j).minCoeff());
    }
    return reduced_matrix;
}

void HungarianAlgorithm::Assigment_Optimize()
{
    std::cout << this->_reduced_cost_matrix << "\n";
    for (int worker = 1; worker <= _rows; ++worker)
    {
        int cost = 0;
        int assignedJob = 0;
        std::vector<bool> visited(this->_cols + 1, false);
        std::vector<int> min_Values(this->_cols + 1, INF);
        this->_assignment[0] = worker;

        while (true)
        {
            visited[assignedJob] = true;
            int delta = INF;
            int next_Job = 0;
            int currentWorker = _assignment[assignedJob];

            for ( int job = 1; job <= _cols; ++job)
            {
                if(!visited[job])
                {
                    cost = this->_reduced_cost_matrix(currentWorker - 1, job - 1) - this->_u[currentWorker] - this->_v[job];

                    if ( cost < min_Values[job] )
                    {
                        min_Values[job] = cost;
                        this->_path[job] = assignedJob;
                    }

                    if ( min_Values[job] < delta )
                    {
                        delta = min_Values[job];
                        next_Job = job;
                    }
                }
                std::cout << "Job: " << job << std::endl;
                std::cout << "Cost: " << this->_reduced_cost_matrix(currentWorker - 1, job - 1) << " - " << this->_u[currentWorker] << " - " << this->_v[job] << " = " << cost << std::endl;
                std::cout << "Min Values[" << job << "]: " << min_Values[job] << std::endl;
                std::cout << "Path[" << job << "]: " << this->_path[job] << std::endl;
            }

            std::cout << "Delta: " << delta << std::endl;
            std::cout << "Next Job: " << next_Job << std::endl;
            Update_Labels(delta, visited, min_Values);
            assignedJob = next_Job;
            
            if (this->_assignment[assignedJob] == 0)
                break;
        }
        Augment_Path(assignedJob);
        std::cout << "-----------------------------------------------------------\n";
    }
}

void HungarianAlgorithm::Update_Labels(int delta, const std::vector<bool>& visited, std::vector<int>& minValues)
{
    std::cout << "vissted = [" << visited[0] << "-" << visited[1] << "-" << visited[2] << "-" << visited[3] << "]\n";
    for( int job = 0; job <= this->_cols; job++)
    {
        if(!visited[job])
        {
            this->_u[job] += delta;
            this->_v[job] -= delta;
        } else {
            minValues[job] -= delta;
        }
        
        std::cout << "u[" << this->_assignment[job] << "]: " << this->_u[this->_assignment[job]] << std::endl;
        std::cout << "v[" << job << "]: " << this->_v[job] << std::endl;
        std::cout << "Min Values[" << job << "]: " << minValues[job] << std::endl;
    }
}

void HungarianAlgorithm::Augment_Path(int assignedJob)
{
    while (true)
    {
        int previous_Job = this->_path[assignedJob];
        this->_assignment[assignedJob] = this->_assignment[previous_Job];
        assignedJob = previous_Job;

        std::cout << "Assigned Job: " << assignedJob << std::endl;
        std::cout << "Previous Job: " << previous_Job << std::endl;
        std::cout << "Assignment[" << assignedJob << "]: " << this->_assignment[assignedJob] << std::endl;

        if (assignedJob == 0) break;
    }
}

void HungarianAlgorithm::Extract_Assignment(std::vector<int> &assignment_results, Eigen::MatrixXd &matching_assignment)
{
    assignment_results.resize(this->_rows);
    matching_assignment = Eigen::MatrixXd::Constant(this->_rows, this->_cols, 0);
    for (int job = 1; job <= this->_cols; job++)
    {
        assignment_results[this->_assignment[job] - 1] = job - 1;
        matching_assignment(this->_assignment[job] - 1, job - 1) = 1;
        std::cout << "Final Assignment - Worker: " << this->_assignment[job] - 1 << ", Job: " << job - 1 << std::endl;
    }
}