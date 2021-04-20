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

  int rows = 1000000;
  int cols = 1000000;

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
  auto XXT = X * X.transpose();
  //  std::cout << "XXT: " << XXT << std::endl;

  PardisoLU<SparseMatrix<double>> solver;
  ConjugateGradient<SparseMatrix<double>> solver2;
  VectorXd b = VectorXd::Zero(rows);
  b(0) = 1;

  //  std::cout << "b: " << b << std::endl;

  auto t_start = std::chrono::system_clock::now();
  solver.compute(XXT);
  auto x = solver.solve(b);
  if (solver.info() == Eigen::Success)
    std::cout << "successed." << std::endl;
  else
    std::cerr << "failed." << std::endl;
  auto t_end = std::chrono::system_clock::now();

  solver2.compute(XXT);
  auto x2 = solver2.solve(b);
  if (solver2.info() == Eigen::Success)
    std::cout << "successed." << std::endl;
  else
    std::cerr << "failed." << std::endl;
  auto t_end2 = std::chrono::system_clock::now();

  std::cout << "#iterations:     " << solver2.iterations() << std::endl;
  std::cout << "estimated error: " << solver2.error() << std::endl;

  //  std::cout << "x:" << x << std::endl;
  //  std::cout << "x2:" << x2 << std::endl;

  auto t_dur = t_end - t_start;
  auto t_dur2 = t_end2 - t_end;
  auto t_sec = std::chrono::duration_cast<std::chrono::seconds>(t_dur).count();
  auto t_sec2 = std::chrono::duration_cast<std::chrono::seconds>(t_dur2).count();
  std::cout << "x Computation time: " << t_sec << " sec \n";
  std::cout << "x2 Computation time: " << t_sec2 << " sec \n";
}