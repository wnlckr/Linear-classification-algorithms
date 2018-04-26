#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<math.h>
#include<limits.h>
#include<float.h>
#include<algorithm>
#include<iomanip>
#include "matrix.h"

using namespace std;

vector< vector<double> > dataset;
vector< vector<double> > test_dataset;
vector< vector<double> > x;
vector< vector<double> > test_x;
vector< vector<double> > x_0;
vector< vector<double> > x_1;
vector< vector<double> > test_x0;
vector< vector<double> > test_x1;
vector< vector<double> > x_transpose;
vector< vector<double> > x_mean;
vector< vector<double> > sigma;
vector< vector<double> > sigma_inverse;
vector< vector<double> > sigma_cofactors;
vector< vector<double> > sigma_adjoint;
vector<double> y;
vector<double> test_y;
vector<double> predicted_y;
vector<double> u0;
vector<double> u1;


void load_data(char* filename, vector< vector<double> >& dataset, vector< vector<double> >& x, vector<double>& y, vector< vector<double> >& x_0, vector< vector<double> >& x_1)
{
    ifstream infile(filename);
    string line;
    double data;
    int count = 0;
    while( getline(infile,line))
    {
        stringstream ss(line);
        vector<double> temp;
        while( ss >> data)
        {
                temp.push_back(data);
                if( ss.peek() == ',')
                    ss.ignore();
        }
        dataset.push_back(temp);
        ss.str("");
    }
    for(int i = 0; i < dataset.size(); i++)
    {
        vector<double> temp;
        temp.assign(dataset[0].size()-1, 0.0);
        x.push_back(temp); 
    }
    for( int i = 0; i < dataset.size(); i++)
    {
        vector<double> temp;
        for(int j = 0; j < dataset[i].size()-1 ; j++)
        {
                x[i][j] = dataset[i][j];
        }
        y.push_back(dataset[i][dataset[i].size() -1 ]);
    }

    for(int i = 0; i < x.size(); i++)
    {
        vector<double> temp;
        for(int j = 0; j < x[0].size(); j++)
        {
            temp.push_back(x[i][j]);
        }
        if( y[i] == 0)
            x_0.push_back(temp);
        else x_1.push_back(temp);
    }
}
 
void print_dataset(vector< vector<double> >&x, vector<double>& y)
{
    for(int i = 0; i < x.size(); i++)
    {
        for(int j = 0; j < x[0].size(); j++)
            cout << x[i][j] << " ";
        cout << "       " << y[i];
        cout << endl;
    }
    cout << endl;
    cout << endl;

    cout << "dataset belonging to positive class, y[i] == 0): " << endl;
    for(int i = 0; i < x_0.size(); i++)
    {
        for(int j = 0; j < x_0[0].size(); j++)
            cout << x_0[i][j] << " " ;
        cout << endl;
    }
    cout << endl << endl;
    cout << "dataset belonging to negative class, y[i] == 1) : " << endl;
    for(int i = 0; i < x_1.size(); i++)
    {
        for(int j = 0; j < x_1[0].size(); j++)
            cout << x_1[i][j] << " ";
        cout << endl;
    }
    cout << endl << endl;
}

double calculate_phi(vector<double> y)
{
    /* phi is the fraction of positive training examples i.e fraction of training examples for which y[i] == 1 */
    int n = y.size();
    double phi = 0.0;
    int count = 0;
    for(int i = 0; i < y.size(); i++)
        if( y[i] == 1)
            count++;
    phi = (double)count/n;
    return phi;
}

/* calculate the mean values of dataset */
double calculate_mean(vector< vector<double> >& x,vector<double>& u)
{
    for(int i = 0; i < x[0].size(); i++)
    {
        double sum = 0.0;
        for(int j = 0; j < x.size(); j++)
            sum += x[j][i];
        sum = sum/x.size();
        u.push_back(sum);
    }
}


/* calculate sigma, where sigma is the covariance of the dataset matrix x */
double calculate_sigma(vector< vector<double> > x, vector<double> y, vector<double> u0, vector<double> u1)
{
    /* mean center the matrix */
    for(int i = 0; i < x.size(); i++)
    {
        vector<double> temp;
        for(int j = 0; j < x[0].size(); j++)
        {
            if( y[i] == 1)
              temp.push_back(x[i][j] - u1[j]);
            else temp.push_back(x[i][j] - u0[j]);
        }
        x_mean.push_back(temp);
    }

    /* transpose the matrix */
    transpose(x_mean,x_transpose);
    /* multiply the matrices */
    multiply_matrix(x_transpose,x_mean,sigma);
    /* find the covariance matrix */
    for(int i = 0; i < sigma.size(); i++)
    {
        for(int j = 0; j < sigma[i].size(); j++)
            sigma[i][j] = (sigma[i][j]/(y.size()));
    }
}

double calculate_px_py(vector<double> x_i , vector<double> u, vector< vector<double> > sigma)
{
    int n = 1;
    double pi = 3.14;
    double det = determinant(sigma,sigma.size());
    int dim = sigma.size();
    double sigma_denom = pow(det,0.5);
    double pi_denom = pow( 2*pi, 0.5*dim);
    double denom = pi_denom * sigma_denom;
    inverse(sigma, sigma_inverse,sigma_adjoint);
    vector<double> xi;
    for(int i = 0; i < x_i.size(); i++)
       xi.push_back(x_i[i] - u[i]);
    
    vector<double> product;
    product.assign(xi.size(),0.0);
    
    for(int i = 0; i < sigma_inverse.size(); i++)
    {
        for(int j = 0; j < sigma_inverse[0].size(); j++)
        {
            product[i] += ( sigma_inverse[i][j] * xi[j]);
        }
    }

    double sum = 0.0;
    for(int i = 0;i < product.size(); i++)
        sum += product[i]* xi[i];

    double e = 2.718281828;
    double exponent = pow(e, -0.5 * sum);

    double result = (1.0/denom) * exponent;
    return result;
}

double calculate_py(double y, double phi)
{
  if( y == 1)
    return phi;
  else return (1- phi);
}

int main()
{
    /* load the training dataset */
    load_data("train.txt",dataset,x,y,x_0,x_1);

    double phi = calculate_phi(y);
    calculate_mean(x_0,u0);
    calculate_mean(x_1,u1);

    cout << "mu1: " << endl;
    for(int i = 0; i < u0.size(); i++)
        cout << u0[i] << " ";
    cout << endl;

    cout << "mu2: " << endl;
    for(int i = 0; i < u1.size(); i++)
        cout << u1[i] << " ";
    cout << endl;

    calculate_sigma(x,y,u0,u1);
 
    for(int i = 0; i < sigma.size(); i++)
    {
       vector<double> temp;
       for(int j = 0; j < sigma.size(); j++)
          temp.push_back(0.0);
       sigma_cofactors.push_back(temp);
    }
 
    for(int i = 0; i < sigma.size(); i++)
    {
      vector<double> temp;
      for(int j = 0; j < sigma.size(); j++)
        temp.push_back(0.0);
      sigma_adjoint.push_back(temp);
    }

    for(int i = 0; i < sigma.size(); i++)
    {
      vector<double> temp;
      for(int j = 0; j < sigma.size(); j++)
        temp.push_back(0.0);
      sigma_inverse.push_back(temp);
    }
 
    /* prediction */

    // load the test dataset 
    load_data("test.txt",test_dataset,test_x,test_y,test_x0,test_x1);
  
    for(int i = 0; i < test_x.size(); i++)
    {
      double px0_0 = calculate_px_py(test_x[i], u0, sigma)*calculate_py(0, phi) ;
      double px0_1 = calculate_px_py(test_x[i], u1, sigma)*calculate_py(1, phi) ;
      
     double pc1_x = px0_1/(px0_0 + px0_1);
     double pc0_x = px0_0/(px0_0 + px0_1);

     if( pc1_x > 0.5)
         predicted_y.push_back(1);
     else predicted_y.push_back(0);
    } 
 
    cout << "test data " << endl << endl;
    for(int i = 0; i < test_x.size(); i++)
    {
        for(int j = 0; j < test_x[0].size(); j++)
            cout << test_x[i][j] << " " ;
        cout << "       " << test_y[i] << "     " << predicted_y[i] << endl;
    }
    int correct = 0;
    int incorrect = 0;   
    int tp = 0;
    int fp = 0;
    int tn = 0; 
    int fn = 0;
    for(int i = 0; i < predicted_y.size(); i++)
    {
        if( test_y[i] == predicted_y[i])
        correct++;
        else incorrect++;
        if( test_y[i] == 1 && predicted_y[i] == 1)
            tp++;
        if( test_y[i] == 0 && predicted_y[i] == 0)
            tn++;
        if( test_y[i] == 1 && predicted_y[i] == 0)
            fn++;
        if( test_y[i] == 0 && predicted_y[i] == 1)
            fp++;
    }
   
    cout << "tp : " << tp << endl;
    cout << "fp : " << fp << endl;
    cout << "tn : " << tn << endl;
    cout << "fn : " << fn << endl;
    cout << "correct predictions: " << correct << endl;
    
    cout << "incorrect predictions: " << incorrect << endl;  
    double precision = (double)tp/(tp + fp);
    double recall = (double)tp/(tp + fn);
    double accuracy = (double)correct/test_y.size();

    cout << "precision: " << precision << endl;
    cout << "recall: " << recall << endl;
    cout << "accuracy: " << accuracy * 100 << "%" << endl;
    
    return 0;
}
