#include <iostream>
#include <random>
#include <chrono>
#include <Eigen/Sparse>
#include <Eigen/PardisoSupport>

using namespace Eigen;
typedef Eigen::Triplet<double> T;
int main()
{
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  int rows = 10000;
  int cols = 10000;

  std::vector<T> tripletList;
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
    {
      auto v_ij = dist(gen); //generate random number
      if (v_ij < 0.1)
      {
        tripletList.push_back(T(i, j, v_ij)); //if larger than treshold, insert it
      }
    }
  SparseMatrix<double> X(rows, cols);
  X.setFromTriplets(tripletList.begin(), tripletList.end()); //create the matrix

  VectorXd b = VectorXd::Zero(rows);
  b(0) = 1.0e-5;

  std::cout << "start to computation" << std::endl;

  auto t_start = std::chrono::system_clock::now();
  PardisoLU<SparseMatrix<double>> solver;
  solver.compute(X);
  auto x = solver.solve(b);
  if (solver.info() == Eigen::Success)
    std::cout << "successed." << std::endl;
  else
    std::cerr << "failed." << std::endl;
  for (int i = 0; i < 5; i++)
    std::cout << x[i] << std::endl;
  auto t_end = std::chrono::system_clock::now();

  SparseLU<SparseMatrix<double>> solver2;
  solver2.compute(X);
  auto x2 = solver2.solve(b);
  if (solver2.info() == Eigen::Success)
    std::cout << "successed." << std::endl;
  else
    std::cerr << "failed." << std::endl;
  for (int i = 0; i < 5; i++)
    std::cout << x2[i] << std::endl;
  auto t_end2 = std::chrono::system_clock::now();

  auto t_dur = t_end - t_start;
  auto t_dur2 = t_end2 - t_end;
  auto t_sec = std::chrono::duration_cast<std::chrono::seconds>(t_dur).count();
  auto t_sec2 = std::chrono::duration_cast<std::chrono::seconds>(t_dur2).count();
  std::cout << "PARDISOLU Computation time: " << t_sec << " sec \n";
  std::cout << "EigenLU Computation time: " << t_sec2 << " sec \n";
}