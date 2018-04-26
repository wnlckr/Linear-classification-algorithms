#include<iostream>
#include<fstream>
#include<vector>
#include<sstream>
#include<string>
#include<algorithm>
#include<math.h>
#include<limits.h>
#include "matrix.h"

using namespace std;

vector< vector<double> > x;
vector<double> y;
vector< vector<double> > dataset;
vector<double> mean_values;
vector< vector<double> > x_0;
vector< vector<double> > x_1;
vector<double> mean_0;
vector<double> mean_1;
vector< vector<double> > x_0_transpose;
vector< vector<double> > x_1_transpose;
vector< vector<double> > x_0_product;
vector< vector<double> > x_1_product;
vector< vector<double> > sw;
vector< vector<double> > sw_inverse;
vector< vector<double> > sw_cofactors;
vector< vector<double> > sw_adjoint;
vector< vector<double> > sb;
vector< vector<double>  > mean;
vector< vector<double> >mean_transpose;
vector< vector<double> >w;
vector< vector<double> >w_transpose;
vector<double> r;
vector<double> r_original;
vector< vector<double> > hold;
vector< vector<double> > x_transpose;
vector< vector<double> > inverse_transpose;

/* test */

vector< vector<double> > test_dataset;
vector< vector<double> > test_x;
vector< vector<double> > test_x_0;
vector< vector<double> > test_x_1;
vector<double> test_y;
vector<double> predicted_y;
vector< vector<double> > test_x_transpose;


double find_info(vector<double> final_values, double threshold)
{
  int first_split_pos = 0;
  int first_split_neg = 0;
  
  int second_split_pos = 0;
  int second_split_neg = 0;
  
  int first_split = 0;
  int second_split = 0;
  
  for(int i = 0; i < final_values.size(); i++)
  {
    if( final_values[i] <= threshold)
    {
      if( y[i] == 0)
        first_split_neg++;
      else first_split_pos++;
      first_split++;
    }
    else 
    {
      if( y[i] == 0)
        second_split_neg++;
      else second_split_pos++;
      second_split++;
    }
  }
  
  double first_pos_ratio = (double)first_split_pos/first_split;
  
  double first_neg_ratio = (double)first_split_neg/first_split;
  
  double second_pos_ratio = (double)second_split_pos/second_split;
  
  double second_neg_ratio = (double)second_split_neg/second_split;
  
  double first,second;
  first = first_pos_ratio * log(max(first_pos_ratio,0.001));
  second = first_neg_ratio * log(max(first_neg_ratio,0.001));

  double first_entropy = -(first + second);

  first = second_pos_ratio * log(max(second_pos_ratio,0.001));
  second = second_neg_ratio * log(max(second_neg_ratio,0.001));
  
  
  double second_entropy = -(first + second);
  
  int tot = y.size();
  
  double info = ((double)(first_split_pos + first_split_neg)/tot) * first_entropy + ((double)(second_split_pos + second_split_neg)/tot) * second_entropy;
  
  return info;
}  

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

void mean_centre(vector< vector<double> >& x, vector<double>& mean)
{
    for(int i = 0; i < x.size(); i++)
    {
        for(int j = 0; j < x[i].size(); j++)
        {
            x[i][j] -= mean[j];
        }
    } 
}

void covariance(vector< vector<double> >& x, int size)
{
    for(int i = 0; i < x.size(); i++)
        for(int j = 0; j < x[i].size(); j++)
            x[i][j] = ((x[i][j])/size);
}

void calculate_sb(vector<double> mean_0, vector<double> mean_1)
{
      /* between class scatter matrix */
    
    vector<double> mean_temp;
    for(int i = 0; i < mean_0.size(); i++)
    {
       mean_temp.push_back(mean_0[i] - mean_1[i]);
    }
    mean.push_back(mean_temp);
    
    transpose(mean, mean_transpose);
    
    multiply_matrix(mean_transpose, mean, sb);

}

void calculate_sw(vector< vector<double> >& x_0, vector< vector<double> >& x_1, vector<double> mean_0, vector<double> mean_1)
{
    /* mean centering the data */
    
    mean_centre(x_0,mean_0);
    mean_centre(x_1,mean_1);
    
    /* transpose the matrix */
    
    transpose(x_0,x_0_transpose);
    transpose(x_1,x_1_transpose);

 
    /* multiplication of the matrix */
 
    multiply_matrix(x_0_transpose, x_0, x_0_product);   
    multiply_matrix(x_1_transpose, x_1, x_1_product);
    
    /* calculate covariance */

    covariance(x_0_product,x_0.size());
    covariance(x_1_product, x_1.size());  

    /* within class scatter matrix */

    for(int i = 0; i < x_0_product.size(); i++)
    {
        vector<double> temp;
        for(int j = 0; j <x_0_product[0].size(); j++)
        {
            double sum = x_0_product[i][j] + x_1_product[i][j];
            temp.push_back(sum);
        }
        sw.push_back(temp);
    }

}
    
void calculate_w( vector< vector<double> >& sw, vector< vector<double> >& mean)
{
    for(int i = 0; i < sw.size(); i++)
    {
        vector<double> temp;
       for(int j = 0; j < sw.size(); j++)
       {
          temp.push_back(0.0);
       }
       sw_cofactors.push_back(temp);
    }
 
    for(int i = 0; i < sw.size(); i++)
    {
      vector<double> temp;
      for(int j = 0; j < sw.size(); j++)
        temp.push_back(0.0);
      sw_adjoint.push_back(temp);
    }
 
    for(int i = 0; i < sw.size(); i++)
    {
      vector<double> temp;
      for(int j = 0; j < sw.size(); j++)
        temp.push_back(0.0);
      sw_inverse.push_back(temp);
    }
    
    inverse(sw, sw_inverse, sw_adjoint);
 
    multiply_matrix(sw_inverse,mean_transpose,w);
    vector< vector<double> >temp_hold;
    transpose(w,temp_hold);
    
    for(int i = 0; i < w.size(); i++)
      w[i].clear();
      
    w.clear();
    
    for(int i = 0; i< temp_hold.size(); i++)
      w.push_back(temp_hold[i]);
    
    double length = 0.0;
    
    for(int i = 0; i < w.size(); i++)
      for(int j = 0; j < w[i].size(); j++)
      length += (w[i][j] * w[i][j]);
      
    length = sqrt(length);
    
    for(int i = 0; i < w.size(); i++)
      for(int j = 0; j < w[i].size(); j++)
        w[i][j] = w[i][j]/length;
        

    cout << "w: " << endl;
    print_matrix(w);

}

void predict(vector< vector<double> >& w,vector< vector<double> >& test_x, vector< vector<double> >& test_x_transpose, double threshold)
{
    for(int i = 0; i < test_x.size(); i++)
    {
      vector<double> temp;
      for(int j = 0; j < test_x[i].size(); j++)
      {
        temp.push_back(test_x[i][j]);
      }
      hold.push_back(temp);
      
      for(int i = 0; i < hold[0].size(); i++)
      {
          vector<double> temp_hold;
          for(int j = 0; j < hold.size(); j++)
          {
              temp_hold.push_back(hold[j][i]);
          }
          test_x_transpose.push_back(temp_hold);
      }
       
      double value = 0.0;
      vector< vector<double> > value_matrix;
      multiply_matrix(w,test_x_transpose, value_matrix);
      value = value_matrix[0][0];
      for(int i = 0; i < hold.size(); i++)
        hold[i].clear();
      hold.clear();
      
      for(int i = 0; i < test_x_transpose.size(); i++)
        test_x_transpose[i].clear();
      test_x_transpose.clear();
      if( value <= threshold)
        predicted_y.push_back(1);
      else predicted_y.push_back(0);

     }
}

double calculate_threshold()
{
    for(int i = 0; i < x.size(); i++)
    {
      vector<double> temp;
      for(int j = 0; j < x[i].size(); j++)
      {
        temp.push_back(x[i][j]);
      }
      hold.push_back(temp);
      
      for(int i = 0; i < hold[0].size(); i++)
      {
          vector<double> temp_hold;
          for(int j = 0; j < hold.size(); j++)
          {
              temp_hold.push_back(hold[j][i]);
          }
          x_transpose.push_back(temp_hold);
      }
       
      double value = 0.0;
      vector< vector<double> > value_matrix;
      multiply_matrix(w,x_transpose,value_matrix);

      value = value_matrix[0][0];
      for(int i = 0; i < hold.size(); i++)
        hold[i].clear();
      hold.clear();
      
      for(int i = 0; i < x_transpose.size(); i++)
        x_transpose[i].clear();
      x_transpose.clear();
      r.push_back(value);
      r_original.push_back(value);
     }
     sort(r.begin(), r.end());
        /* calculate entropy */
   
   // entropy of the total data
   int pos = 0;
   int neg = 0;
   
   for(int i = 0; i < y.size(); i++)
    if( y[i] == 0)
      neg++;
    else pos++;
    
   int tot = y.size();
   double pos_ratio = ((double)pos/tot);
   double neg_ratio = ((double)neg/tot);
 
   double first,second;
     
   double entropy_total = -(pos_ratio * log(pos_ratio) + neg_ratio * log(neg_ratio));
   double threshold = 0.0;
   double max_gain = 0.0;
   for(int i = 0;i <  x.size() - 1; i++)
   {
      double threshold_i = (r[i] + r[i+1])/2;
      double info = find_info(r_original,threshold_i);
      double gain = entropy_total - info;
      if( gain > max_gain)
      {
        max_gain = gain;
        threshold = threshold_i;
      }
   }
   return threshold;
}
   
int main()
{
    load_data("train.txt",dataset,x,y,x_0,x_1);
    
    /* find the mean values for each class */
    calculate_mean(x,mean_values);
    calculate_mean(x_0,mean_0);
    calculate_mean(x_1,mean_1);

    /* calculate within class scatter matrix */ 
    calculate_sw(x_0,x_1,mean_0,mean_1);
    
    /* calculate between class scatter matrix */
    calculate_sb(mean_0, mean_1); 
    
    /* calculate w */
    calculate_w(sw,mean);
    
    /* calculate threshold */
    double threshold = calculate_threshold();
    cout << "threshold: " << threshold << endl;
    
    /* load test data */
    load_data("test.txt", test_dataset,test_x, test_y, test_x_0, test_x_1);
    
    /* predict target values for test data */
    predict(w,test_x,test_x_transpose,threshold);
    
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
