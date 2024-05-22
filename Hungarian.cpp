#include "Hungarian.h"

HungarianAlgorithm::HungarianAlgorithm(const std::vector<std::vector<double>> &values)
{
    Eigen::MatrixXd cost_matrix = create_CostMatrix(values);
    // Handle non-square matrix by adding dummy rows or columns
    int max_size = std::max(cost_matrix.rows(), cost_matrix.cols());
    Eigen::MatrixXd square_matrix = Eigen::MatrixXd::Constant(max_size, max_size, 0);
    square_matrix.block(0, 0, cost_matrix.rows(), cost_matrix.cols()) = cost_matrix;

    this->_rows = cost_matrix.rows();
    this->_cols = cost_matrix.cols();
    this->_square_matrix_size = max_size;
    this->_cost_matrix = square_matrix;
    this->_u = std::vector<int>(_square_matrix_size + 1, 0);
    this->_v = std::vector<int>(_square_matrix_size + 1, 0);
    this->_assignment = std::vector<int>(_square_matrix_size + 1, 0);
    this->_path = std::vector<int>(_square_matrix_size + 1, 0);
    this->_reduced_cost_matrix = Reduced_Cost_Matrix(square_matrix);
}

void HungarianAlgorithm::solve(std::vector<int> &assignment, Eigen::MatrixXd &matching_assignment, double &total_cost)
{
    Assignment_Optimize();
    Extract_Assignment(assignment, matching_assignment, total_cost);
}

Eigen::MatrixXd HungarianAlgorithm::Reduced_Cost_Matrix(const Eigen::MatrixXd &matrix)
{
    Eigen::MatrixXd reduced_matrix = matrix;
    // Reduce Rows
    for (int i = 0; i < reduced_matrix.rows(); i++)
    {
        reduced_matrix.row(i) -= Eigen::VectorXd::Constant(reduced_matrix.cols(), reduced_matrix.row(i).minCoeff());
    }

    // Reduce Columns
    for (int j = 0; j < reduced_matrix.cols(); j++)
    {
        reduced_matrix.col(j) -= Eigen::VectorXd::Constant(reduced_matrix.rows(), reduced_matrix.col(j).minCoeff());
    }
    return reduced_matrix;
}

void HungarianAlgorithm::Assignment_Optimize()
{
    for (int worker = 1; worker <= _square_matrix_size; ++worker)
    {
        int cost = 0;
        int assignedJob = 0;
        std::vector<bool> visited(this->_square_matrix_size + 1, false);
        std::vector<int> min_Values(this->_square_matrix_size + 1, INF);
        this->_assignment[0] = worker;

        while (true)
        {
            visited[assignedJob] = true;
            int delta = INF;
            int next_Job = 0;
            int currentWorker = _assignment[assignedJob];
            for (int job = 1; job <= _square_matrix_size; ++job)
            {
                if (!visited[job])
                {
                    cost = this->_reduced_cost_matrix(currentWorker - 1, job - 1) - this->_u[currentWorker] - this->_v[job];

                    if (cost < min_Values[job])
                    {
                        min_Values[job] = cost;
                        this->_path[job] = assignedJob;
                    }

                    if (min_Values[job] < delta)
                    {
                        delta = min_Values[job];
                        next_Job = job;
                    }
                }
            }

            Update_Labels(delta, visited, min_Values);
            assignedJob = next_Job;

            if (this->_assignment[assignedJob] == 0)
                break;
        }
        Augment_Path(assignedJob);
    }
}

void HungarianAlgorithm::Update_Labels(int delta, const std::vector<bool> &visited, std::vector<int> &minValues)
{
    for (int job = 0; job <= this->_square_matrix_size; job++)
    {
        if (!visited[job])
        {
            this->_u[this->_assignment[job]] += delta;
            this->_v[job] -= delta;
        }
        else
        {
            minValues[job] -= delta;
        }
    }
}

void HungarianAlgorithm::Augment_Path(int assignedJob)
{
    while (true)
    {
        int previous_Job = this->_path[assignedJob];
        this->_assignment[assignedJob] = this->_assignment[previous_Job];
        assignedJob = previous_Job;

        if (assignedJob == 0)
            break;
    }
}

void HungarianAlgorithm::Extract_Assignment(std::vector<int> &assignment_results, Eigen::MatrixXd &matching_assignment, double &total_cost)
{
    assignment_results.resize(this->_rows);
    matching_assignment = Eigen::MatrixXd::Constant(this->_rows, this->_cols, 0);
    total_cost = 0.0;
    for (int job = 1; job <= this->_square_matrix_size; job++)
    {
        if (this->_assignment[job] <= this->_rows)
        { 
            // Ignore dummy rows
            int worker = this->_assignment[job] - 1;
            int actual_job = job - 1;
            assignment_results[worker] = actual_job;
            if (job <= this->_cols) {
                matching_assignment(worker, actual_job) = 1;
                total_cost += this->_cost_matrix(worker, actual_job);
            }
        }
    }

}

Eigen::MatrixXd HungarianAlgorithm::create_CostMatrix(const std::vector<std::vector<double>> &values) {
    size_t rows = values.size();
    size_t cols = values[0].size();
    Eigen::MatrixXd matrix(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix(i, j) = values[i][j];
        }
    }
    return matrix;
}