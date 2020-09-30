#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>

using namespace std;
const int sz = 32;
const int sy = 32;
const int sx = 32; 
const int method_max = 4;


float myfunc(int x, int y, int z, int method){
//
  switch (method){
    case 0:
      if(x==0)
        return 1;
      return x*8;
      break;
    case 1:
      return y;
      break;
    case 2:
      return z;
      break;
    case 3:
      return x+y;
      break;
    case 4:
      return (x+y)*2*((z+1)/float(sz));
      break;
    default:
      return x;
      break;
    }
    return 0;
}



int main(int argc, char *argv[]){

  int method = 0;
  if (argc > 1) method = atoi(argv[1]);
  ofstream fout;
  string foname("Bucky.raw");
  fout.open(foname.c_str());
  if (!fout) {std::cout << "file open failed" << std::endl; return 1;}
  for (int z = 0; z < sz; z++)
    for (int y = 0; y < sy; y++)
      for (int x = 0; x < sx; x++)
        fout << (unsigned char)(myfunc(x, y, z, method));
  if (!fout) {std::cout << "file write failed" << std::endl; return 1;}
  fout.close();
  return 0;
}
