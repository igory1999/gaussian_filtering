#include <iostream>
#include <cmath>
#include <vector>
#include <mpi.h>
#include <adios2.h>


using namespace std;

class Cube
{
  vector<double> data;
  int size;
  inline void check_bounds(int i, int j, int k, string msg) const
  {
    if(i < 0 || i > size - 1 || j < 0 || j > size - 1 || k < 0 || k > size - 1)
      {
	cerr << "out of bounds in " << msg << endl;
	cerr << "i=" << i << " j=" << j << " k=" << k << " size = " << size << endl;
	exit(1);
      }
  }
public:
  Cube(int size):  size(size) { data = vector<double>(size*size*size, 0); }
  Cube(Cube &other): size(other.size), data(other.data) {}
  inline void set(int i, int j, int k, double v)
  {
    check_bounds(i, j, k, "set");
    int index = i*size*size + j*size + k;
    data[index] = v;
  }

  inline void set(Cube& v)
  {
    data = v.data;
    size = v.size;
  }
  
  inline double get(int i, int j, int k) const
  {
    check_bounds(i, j, k, "get");
    int index = i*size*size + j*size + k;
    return data[index];
  }
  inline int get_size() const
  {
    return size;
  }
  void print() const
  {
    for(int i = 0; i < size; ++i)
    {
      for(int j = 0; j < size; ++j)
	{
	  for(int k = 0; k < size; ++k)
	    {
	      cout << "data[" << i << "][" << j << "]["
		   << k << "] = " << get(i, j, k)  << endl;
	    }
	}
    }    
  }
  void print_nonzeros() const
  {
    for(int i = 0; i < size; ++i)
    {
      for(int j = 0; j < size; ++j)
	{
	  for(int k = 0; k < size; ++k)
	    {
	      double v = get(i, j, k);
	      if(v != 0.0)
		cout << "data[" << i << "][" << j << "]["
		     << k << "] = " << v  << endl;
	    }
	}
    }    
  }

  double * get_data()
  {
    return data.data();
  }
  
};

void generate_kernel(double sigma, Cube & kernel)
{
  double sum = 0.0;
  int size = kernel.get_size();
  
  for(int i = 0; i < size; ++i)
    {
      for(int j = 0; j < size; ++j)
	{
	  for(int k = 0; k < size; ++k)
	    {
	      double r2 = ((size - 1)/2 - i)*((size - 1)/2 - i) +
		((size - 1)/2 - j)*((size - 1)/2 - j) + ((size - 1)/2 - k)*((size -1)/2 - k);
	      kernel.set(i, j, k, exp(-r2/2/sigma/sigma)/2/sigma/sigma/M_PI);
	      sum += kernel.get(i, j, k);
	    }
	}
    }


  for(int i = 0; i < size; ++i)
    {
      for(int j = 0; j < size; ++j)
	{
	  for(int k = 0; k < size; ++k)
	    {
	      kernel.set(i, j, k, kernel.get(i, j, k)/sum);
	    }
	}
    }  
}


void apply_kernel(Cube &data, const Cube &kernel)
{
  int data_size = data.get_size();
  int kernel_size = kernel.get_size();
  int kernel_window_size = (kernel_size - 1)/2;
  Cube tmp(data);

#pragma omp target teams distribute parallel for collapse(3)
  for(int i = 0; i < data_size; ++i)
    {
      for(int j = 0; j < data_size; ++j)
	{
	  for(int k = 0; k < data_size; ++k)
	    {
	      double v = 0;
	      for(int ii = -kernel_window_size; ii <= kernel_window_size; ++ii)
		{
		  for(int jj = -kernel_window_size; jj <= kernel_window_size; ++jj)
		    {
		      for(int kk = -kernel_window_size; kk <= kernel_window_size; ++kk)
			{
			  if(i + ii < 0 || i + ii >= data_size ||
			     j + jj < 0 || j + jj >= data_size ||
			     k + kk < 0 || k + kk >= data_size)
			    {
			      v += kernel.get(ii + kernel_window_size,
					      jj + kernel_window_size,
					      kk + kernel_window_size) * data.get(i, j, k);

			    }
			  else
			    {
			      v += kernel.get(ii + kernel_window_size,
					      jj + kernel_window_size,
					      kk + kernel_window_size) * data.get(i + ii, j + jj, k + kk);
			    }
			}
		    }
		}
	      tmp.set(i, j, k, v);
	    }
	}
    }
  data.set(tmp);
}

int main(int argc, char ** argv)
{
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int rank, comm_size, wrank;

  MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
  const unsigned int color = 2;
  MPI_Comm comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &comm);

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);


  adios2::ADIOS ad ("adios2.xml", comm, adios2::DebugON);
  adios2::IO writer_io = ad.DeclareIO("data");
  adios2::Engine writer =
    writer_io.Open("data.bp",
		   adios2::Mode::Write, comm);

  adios2::Variable<double> var;

  const int window_size = 5;
  const double sigma = 1.0;
  Cube kernel(2*window_size + 1);
  generate_kernel(sigma, kernel);
  //cube.print();

  Cube data(50);
  data.set(10, 20, 30, 5.3);
  //data.print_nonzeros();
  //cout << "==============" << endl;
  apply_kernel(data, kernel);
  //data.print_nonzeros();

  size_t L = data.get_size();

  var =
    writer_io.DefineVariable<double> ("data",
      { L, L, L },
      { 0, 0, 0 },
      { L, L, L } );

  writer.BeginStep ();
  writer.Put<double> (var, data.get_data());
  writer.EndStep();

  writer.Close();
  MPI_Finalize();
}
