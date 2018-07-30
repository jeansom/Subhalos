#include <iostream>
#include <cstdio>
#include <cmath>

// Linear interpolation following MATLAB linspace
std::vector<double> LinearSpacedArray(double a, double b, std::size_t N)
{
  double h = (b - a) / static_cast<double>(N-1);
  std::vector<double> xs(N);
  std::vector<double>::iterator x;
  double val;
  for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) {
    *x = val;
  }
  return xs;
}

std::vector<double> LogSpacedArray(double a, double b, std::size_t N)
{
  double h = (b - a) / static_cast<double>(N-1);
  std::vector<double> xs(N);
  std::vector<double>::iterator x;
  double val;
  for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) {
    *x = pow(10, val);
  }
  return xs;
}
