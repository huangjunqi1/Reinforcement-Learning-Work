#include<bits/stdc++.h>
using namespace std;
#define mp make_pair
const int Size = 5;
const int dx[4] = {-1,0,1,0};
const int dy[4] = {0,1,0,-1};
const double gamma = 0.9;
const double eps = 1e-6;
pair<int,int>toP[Size][Size][4];
int reward[Size][Size][4];
double value[Size][Size],newValue[Size][Size];
void init(double **value){
    for (int i=0;i<Size;i++)
        for (int j=0;j<Size;j++)
            value[i][j] = 0;
}
int main(){
    for (int i=0;i<Size;i++)  //确定每个动作的目的地与价值
        for (int j=0;j<Size;j++)
            for (int d=0;d<4;d++){
                if (i == 0 && j == 1){  //A
                    toP[i][j][d] = mp(4,1);
                    reward[i][j][d] = 10;
                    continue;
                }
                if (i == 0 && j == 3){  //B
                    toP[i][j][d] = mp(2,3);
                    reward[i][j][d] = 5;
                    continue;
                }
                int x=i+dx[d],y=j+dy[d];
                if (x<0||x>=Size||y<0||y>=Size){
                    toP[i][j][d] = mp(i,j);
                    reward[i][j][d]=-1;
                }
                else {
                    toP[i][j][d] = mp(x,y);
                    reward[i][j][d] = 0;
                }
            }

    for (int turn = 0; turn < 1e6; turn++){
        for (int i=0;i<Size;i++)
            for (int j=0;j<Size;j++){
                double nowValue = 0;
                for (int d=0;d<4;d++){
                    auto toPoint = toP[i][j][d];
                    int rew = reward[i][j][d];
                    nowValue += gamma*value[toPoint.first][toPoint.second] + rew;
                }
                newValue[i][j] = nowValue/4.0;  //随即策略，每个方向都有1/4的概率
            }

        double maxdelta = 0;
        for (int i=0;i<Size;i++)
            for (int j=0;j<Size;j++){
                maxdelta = max(maxdelta,abs(newValue[i][j] - value[i][j]));
            }
        if (maxdelta < eps) break;

        for (int i=0;i<Size;i++)
            for (int j=0;j<Size;j++)
                value[i][j] = newValue[i][j];
    }   
    for (int i=0;i<Size;i++){
        for (int j=0;j<Size;j++) printf("%.2f ",value[i][j]);
        puts("");
    }
    return 0;
}