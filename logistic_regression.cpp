#include<iostream>
#include<vector>
#include<algorithm>
#include<fstream>
#include<sstream>
#include<limits.h>
#include<float.h>
#include<math.h>

using namespace std;

vector< vector<double> >dataset;
vector< vector<double> > x;
vector<double> y;
vector<double> min_array;
vector<double> max_array;
vector<double> range_array;
vector<double> beta;
vector<double> predicted_y;
vector< vector<double> > test_dataset;
vector< vector<double> > test_x;
vector<double> test_y;
vector<double> test_min_array;
vector<double> test_max_array;
vector<double> test_range_array;

double sigmoid(double z)
{
    double ans;
    double e = 2.718281828;
    ans = 1.0/(1.0 + pow(e,-z));
    return ans;
}

void predict(vector<double> beta, vector< vector<double> > x)
{
    for(int i = 0; i < x.size(); i++)
    {
        double z = 0.0;
        for(int j = 0; j < x[0].size(); j++)
        {
            z += beta[j] * x[i][j];
        }
        double hxi = sigmoid(z);
        if( hxi >= 0.5)
            predicted_y.push_back(1);
        else predicted_y.push_back(0);
    }
}

double cost_function( vector<double>& beta, vector< vector<double> >& x, vector<double>& y,double alpha)
{
    double cost = 0.0;
    double diff = 0.0;
    for(int i = 0; i< x.size(); i++)
    {
        double z = 0.0;
        for(int j = 0; j < x[0].size(); j++)
        {
            z += beta[j]* x[i][j];
        }
        double hxi = sigmoid(z);
        double txi = y[i]*log(hxi) + (( 1 - y[i]) * log( 1 - hxi));
        cost += txi;
        diff = hxi - y[i];
        for(int j = 0; j < x[0].size(); j++)
        {
            beta[j] = beta[j] - alpha* (x[i][j] * diff)/ x.size();
        }
    }
    return -cost;
}

int gradient_descent(vector< vector<double> >& x, vector<double>& y, vector<double>& beta)
{
    double alpha = 5; // learning rate
    double converge_change = 0.0001;

    double cost = cost_function(beta,x,y,alpha);
    cout << "initial cost: " << cost << endl;

    double change_cost= 1;
    int num_iter = 1;

    while( change_cost > converge_change)
    {
        double old_cost = cost;
        cost = cost_function(beta,x,y,alpha);
        cout << endl;
        cout << "cost: " << cost << endl;
        cout << endl;
        cout << "new beta value: " << endl;

        for(int i = 0; i < beta.size(); i++)
            cout << beta[i] << " ";
        cout << endl;

        change_cost = old_cost - cost;
        num_iter += 1;
    }
    return num_iter;
}

void load_data(char* filename, vector< vector<double> >& dataset, vector< vector<double> >& x, vector<double>& y)
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
        temp.assign(dataset[0].size(), 0.0);
        x.push_back(temp); 
        x[i][0] = 1;
    }
    for( int i = 0; i < dataset.size(); i++)
    {
        vector<double> temp;
        for(int j = 1; j < dataset[i].size() ; j++)
        {
                x[i][j] = dataset[i][j-1];
        }
        y.push_back(dataset[i][dataset[i].size() -1 ]);
    }
}
  
void normalize_data(vector< vector<double> >& x, vector<double>& min_array, vector<double>& max_array, vector<double>& range_array)
{
    /* normalize the data */

    // find the minimum and maximum values in the dataset for each attribute
    for(int i = 0; i < x[0].size(); i++)
    {
        double min_value = DBL_MAX;
        double max_value = DBL_MIN;
        for(int j = 0; j < x.size(); j++)
        {
            if( x[j][i] < min_value)
                min_value = x[j][i];
            if( x[j][i] > max_value)
                max_value = x[j][i];
        }
        min_array.push_back(min_value);
        max_array.push_back(max_value);
        range_array.push_back(max_value - min_value);
    }

    // debugging purposes
    
    cout << "min values: " << endl;
    for(int i = 0; i < min_array.size(); i++)
        cout << min_array[i] << " ";
    cout << endl;

    cout << "max values: " << endl;
    for(int i = 0; i < max_array.size(); i++)
        cout << max_array[i] << " ";
    cout << endl;

    cout << "range values: " << endl;
    for(int i = 0; i < range_array.size(); i++)
        cout << range_array[i] << " ";
    cout << endl;

    // Normalize the data

    for(int i = 0; i < x.size(); i++)
    {
        for(int j = 1; j < x[0].size(); j++)
            x[i][j] = (x[i][j] - min_array[j])/ range_array[j];
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
}

int main()
{
    /* load training data */
    load_data("train.txt",dataset,x,y);

    /* normalize data */
    normalize_data(x,min_array,max_array,range_array);

    /* print the dataset for debugging purposes */
    print_dataset(x,y);
    
    /* initial beta values */

    for(int i = 0; i < x[0].size(); i++)
        beta.push_back(0.0);

    /* train the logistice regression model */
    int num = gradient_descent(x,y,beta);
    cout << num << endl;

    /* prediction of values */

    /*load test data*/
    string test_data = "test.txt";
    load_data("test.txt",test_dataset,test_x,test_y);

    /* normalize data */
    normalize_data(test_x,test_min_array, test_max_array, test_range_array);
    predict(beta,test_x);

    int correctly_predicted = 0;
    int incorrect_prediction = 0;

    int true_positive = 0;
    int false_positive = 0;
    int true_negative = 0;
    int false_negative = 0;

    for(int i = 0; i < test_x.size(); i++)
    {
        if( test_y[i] == predicted_y[i])
            correctly_predicted++;
        else incorrect_prediction++;
        if( test_y[i] == 0 && predicted_y[i] == 0)
            true_negative++;
        if( test_y[i] == 1 && predicted_y[i] == 1)
            true_positive++;
        if( test_y[i] == 0 && predicted_y[i] == 1)
            false_positive++;
        if( test_y[i] == 1 && predicted_y[i] == 0)
            false_negative++;
    }
    cout <<"true_positive: " << true_positive << endl;
    cout << "true_negative: " << true_negative << endl;
    cout << "false_positive: " << false_positive << endl;
    cout << "false_negative: " << false_negative << endl;

    cout << "correctly predicted: " << correctly_predicted << endl;
    cout << "incorrectly predicted: " << incorrect_prediction << endl;
    double precision = (double)true_positive/(true_positive + false_positive);
    cout << "precision : " << precision << endl;
    double recall = (double)true_positive/(true_positive + false_negative);
    cout << "recall : " << recall << endl;
    double accuracy = (double)correctly_predicted/test_x.size();
    cout << "accuracy: " << accuracy*100 << "%" << endl;

    return 0;
}
