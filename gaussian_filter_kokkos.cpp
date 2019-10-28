#include <iostream>
#include <cmath>
#include <vector>
#include <mpi.h>
#include <adios2.h>
#include <iostream>

#include <Kokkos_Core.hpp>

typedef Kokkos::View<double***>  ViewMatrixType;

void generate_gaussian(int l, double sigma, ViewMatrixType g)
{
     Kokkos::parallel_for(
			   Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {l,l,l}), 
			   KOKKOS_LAMBDA (int i, int j, int k)
			   {
			     double r2 = ((l - 1)/2 - i)*((l - 1)/2 - i) +
			       ((l - 1)/2 - j)*((l - 1)/2 - j) + ((l - 1)/2 - k)*((l -1)/2 - k);
			     g(i, j, k) = exp( - r2/2/sigma/sigma)/2/sigma/sigma/M_PI;
			   }
			   );
      double sum = 0;
      Kokkos::parallel_reduce(
			   Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {l,l,l}), 
			   KOKKOS_LAMBDA (int i, int j, int k, double & local_sum)
			   {
			     local_sum += g(i, j, k);
			   },
			   sum
			   );      
      std::cout<<sum<<std::endl;
            Kokkos::parallel_for(
			   Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {l,l,l}), 
			   KOKKOS_LAMBDA (int i, int j, int k)
			   {
			     g(i, j, k) /= sum;
			   }
			   );
}

void init_data(ViewMatrixType::HostMirror d, int d_size)
{
     d(10, 20, 30) = 5.3;
     d(0,0,0) = 10.5;
     d(0,40,40) = - 11.3;
     std::cout << d(13,32,1) << std::endl;
     std::cout << d(10,20,30) << " is " << 5.3 << " in init ?" << std::endl;
}


void apply_kernel(ViewMatrixType data, ViewMatrixType result, int d_size,
		  ViewMatrixType kernel, int k_size)
{
  int w = (k_size - 1)/2;
  Kokkos::parallel_for(
		       Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},
							      {d_size, d_size, d_size}), 
		       KOKKOS_LAMBDA (int i, int j, int k)
		       {
			 result(i, j, k) = 0.0;
			 int ip, jp, kp;
			 for(int ii = -w; ii <= w; ++ii)
			   {
			     for(int jj = -w; jj <= w;  ++jj)
			       {
				 for(int kk = -w; kk <= w; ++kk)
				   {
				     ip = i + ii;
				     jp = j + jj;
				     kp = k + kk;
				     if(ip >= 0  &&  ip < d_size &&
					jp >= 0  &&  jp < d_size &&
					kp >=0 && kp < d_size )
				       {
					 result(i, j, k) += kernel(w + ii, w + jj, w + kk) *
					   data(ip, jp, kp);
				       }
				   }
			       }
			   }
		       }
		       );
}

void to_adios2(ViewMatrixType::HostMirror data, MPI_Comm *comm, std::string stream_name)
{
  adios2::ADIOS ad ("adios2.xml", *comm, adios2::DebugON);
  adios2::IO writer_io = ad.DeclareIO(stream_name);
  adios2::Engine writer =
    writer_io.Open(stream_name + ".bp",
		   adios2::Mode::Write, *comm);

  adios2::Variable<double> var;

  size_t L0 = data.extent(0);
  size_t L1 = data.extent(1);
  size_t L2 = data.extent(2);  

  var =
    writer_io.DefineVariable<double> ("data",
      { L0, L1, L2 },
      { 0, 0, 0 },
      { L0, L1, L2 } );

  writer.BeginStep ();
  writer.Put<double> (var, data.data());
  writer.EndStep();

  writer.Close();  
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
  
  int L = 50;
  int gw = 5;
  int l = 2*gw + 1;
  double sigma = 0.1;
  
  Kokkos::initialize( argc, argv );
  {
    ViewMatrixType g("gaussian", l, l, l);
    ViewMatrixType data("data", L, L, L);
    ViewMatrixType result("result", L, L, L);
    ViewMatrixType::HostMirror h_data = Kokkos::create_mirror_view( data );
    ViewMatrixType::HostMirror h_g = Kokkos::create_mirror_view( g );    
    init_data(h_data, L);
    Kokkos::deep_copy(data, h_data);
    generate_gaussian(l, sigma,  g);
    //Kokkos::deep_copy(h_g, g);
    //Kokkos::deep_copy(result, data);
    apply_kernel(data, result, L, g, l);
    Kokkos::deep_copy(h_data, result);
    to_adios2(h_data, &comm, "data");
    //to_adios2(h_g, &comm, "g");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}

