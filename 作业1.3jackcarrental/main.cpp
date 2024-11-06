#include<bits/stdc++.h>

#define rep(i,a,b) for(int i=a;i<=b;i++)
#define min(a,b) ((a<b)?a:b)
#define max(a,b) ((a>b)?a:b)

const double eps = 1e-4;
double fac[22];

void init_fac(){
    fac[0]=1.0;
    for (int i=1;i<21;i++)
        fac[i]=fac[i-1]*i;
}

double possion_prob(double lamb,int n){
    return exp(-lamb)*powf(lamb,n) / fac[n];
}

const double lambda_rent[2] = {3,4};
const double lambda_ret[2] = {3,2};
const double car_income = 10;
const double move_spend = 2;
const double gamma = 0.9;
double trans_prob[2][21][21];
double R[2][21];

void day_prob(int s,int garage){
    rep(r,0,20){
        double rent_prob = possion_prob(lambda_rent[garage],r);
       // printf("%d %.6f\n",r,rent_prob);
        int out_number = min(r,s);
        R[garage][s] += car_income*rent_prob*out_number;
        rep(ret,0,20){
            double ret_prob = possion_prob(lambda_ret[garage],ret);
            int s_next = min(s-out_number+ret,20);
            trans_prob[garage][s][s_next] += rent_prob*ret_prob;
        }
    }
}

double V[21][21],V1[21][21];
int action[21][21];

double value_calculate(int i,int j,int a){
    if (a>i) a = i;
    if (a<-j) a = -j;
    int n=min(20,i-a);
    int m=min(20,j+a);
    double tmp_v = -abs(a)*move_spend;
    rep(i1,0,20)
        rep(j1,0,20){
            tmp_v+=trans_prob[0][n][i1]*trans_prob[1][m][j1]*(R[0][n]+R[1][m]+gamma*V[i1][j1]);
    }
    return tmp_v;
}


double policy_evaluate(){
    double delta = 0;
    rep(i,0,20)
        rep(j,0,20){
            double old_v = V[i][j];
            int a = action[i][j]; //[-5,5]
            double tmp_v = value_calculate(i,j,a);
            delta = max(delta,tmp_v-old_v);
            V[i][j] = tmp_v;
        }
    return delta;
}

int main(){
    init_fac();
    rep(i,0,20){
        day_prob(i,0);
        day_prob(i,1);
    }
    rep(i,0,20) printf("%.2f ",possion_prob(3,i));
    puts("");
    rep(i,0,20) printf("%.2f ",R[0][i]);
    puts("");
    for (int turns = 0;turns<10000000;turns++){
        double delta = 0;
        rep(i,0,20)
            rep(j,0,20){
                int best_action = 0;
                double best_action_v = 0;
                rep(a,-5,5){
                    if (a>i || a<-j) continue;
                    double value = value_calculate(i,j,a);
                    if (value > best_action_v)
                        best_action_v = value,best_action = a;
                }
                action[i][j] = best_action;
                delta = max(delta,abs(V[i][j]-best_action_v));
                V1[i][j] = best_action_v;
            }
        if (delta < eps) 
        {
            printf("use %d turns\n",turns);
            break;
        }        
        rep(i,0,20)
            rep(j,0,20)
                V[i][j] = V1[i][j];
    }
    rep(i,0,20){
        rep(j,0,20) printf("%.2f ",V[i][j]);
        puts("");
    }
    rep(i,0,20){
        rep(j,0,20) printf("%4d",action[i][j]);
        puts("");
    }
    return 0;
}