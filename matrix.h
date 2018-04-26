#include<iostream>
#include<fstream>
#include<vector>
#include<sstream>
#include<string>
#include<algorithm>
#include<math.h>
#include<limits.h>

using namespace std;
/* to find the inverse of matrix */
void getCofactor(vector< vector<double> >& A, vector< vector<double> >& temp,int p, int q, int n)
{
    int i = 0, j = 0;
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            if (row != p && col != q)
            {
                temp[i][j++] = A[row][col];
                if (j == n - 1)
                {
                    j = 0;
                    i++;
                }
            }
        }
    }
}
 
/* to find the determinant of the matrix */
double determinant(vector< vector<double> >& A, int n)
{
    double D = 0; 
    if (n == 1)
        return A[0][0];
 
    vector< vector<double> > temp;
    for(int i = 0; i < A.size(); i++)
    {
      vector<double> t;
      for(int j = 0; j < A.size(); j++)
        t.push_back(0.0);
      temp.push_back(t);
    }
    int sign = 1; 
 
    for (int f = 0; f < n; f++)
    {
        getCofactor(A,temp,0, f,n);
        D += sign * A[0][f] * determinant(temp,n - 1);
        sign = -sign;
    }
    return D;
}
 
/* to find the adjoint of the matrix */
void adjoint(vector< vector<double> >& A, vector< vector<double> >& sw_adjoint)
{
    if (A.size() == 1)
    {
        sw_adjoint[0][0] = 1;
        return;
    }
    vector< vector<double> > temp;
    for(int i = 0; i < A.size(); i++)
    {
      vector<double> t;
        for(int j = 0; j < A.size(); j++)
        t.push_back(0.0);
      temp.push_back(t);
    }
          
    int sign = 1;
 
    for (int i=0; i<A.size(); i++)
    {
        for (int j=0; j<A.size(); j++)
        {
            getCofactor(A,temp,i, j, A.size());
            sign = ((i+j)%2==0)? 1: -1;
            sw_adjoint[j][i] = (sign)*(determinant(temp,A.size()-1));
        }
    }
}
 
/* to find the inverse of the matrix */
bool inverse(vector< vector<double> >& sw, vector< vector<double> >& sw_inverse, vector< vector<double> >& sw_adjoint)
{
    double det = determinant(sw,sw.size());
    if (det == 0)
    {
        cout << "Singular matrix, can't find its inverse";
        return false;
    }
    adjoint(sw, sw_adjoint);
    for (int i=0; i<sw.size(); i++)
        for (int j=0; j<sw.size(); j++)
            sw_inverse[i][j] = sw_adjoint[i][j]/det;
    return true;
}

void transpose(vector< vector<double> >& x, vector< vector<double> >& x_transpose)
{
    for(int i = 0; i < x[0].size(); i++)
    {
        vector<double> temp;
        for(int j = 0; j < x.size(); j++)
        {
            temp.push_back(x[j][i]);
        }
        x_transpose.push_back(temp);
    }
}
 
void multiply_matrix(vector< vector<double> >& x, vector< vector<double> >& y, vector< vector<double> >& product)
{
    for(int i = 0; i < x.size(); i++)
    {
        vector<double> temp;
        for(int j = 0; j < y[0].size(); j++)
        {
            temp.push_back(0.0);
        }
        product.push_back(temp);
    }

    for(int i = 0; i < x.size(); i++)
    {
        for(int j = 0; j < y[0].size(); j++)
        {
            for(int k = 0; k < x[0].size(); k++)
            {
                product[i][j] += x[i][k] * y[k][j];
            }
        }
    }
} 
 
void print_matrix(vector< vector<double> > mat)
{
  for(int i = 0; i < mat.size(); i++)
  {
    for(int j = 0; j < mat[i].size(); j++)
      cout << mat[i][j] << " ";
    cout << endl;
  }
  
  cout << endl;
}

