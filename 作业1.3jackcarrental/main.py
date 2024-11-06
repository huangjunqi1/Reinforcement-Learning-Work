import numpy as np

car_income = 10
move_spend = 2

lambda_rent = [3,4]
lambda_ret = [3,2]
gamma = 0.9
R = np.zeros((2,21))  //第二天的期望
trans_prob = np.zeros((2,21,21))


def possion_prob(lam,n):
    return np.exp(-lam) * np.float_power(lam,n) / np.math.factorial(n)

def day_prob(s,garage):
    for r in range(21):
        rent_prob = possion_prob(lambda_rent[garage])
        out_number = min(r,s)
        R[garage,s] += out_number*car_income*rent_prob
        for ret in range(21):
            ret_prob = possion_prob(lambda_ret[garage])
            s_next = min(s-out_number+ret,20)
            trans_prob[garage,s,s_next] += rent_prob * ret_prob

def init_prob():
    for i in range(21):
        day_prob(i,0)
        day_prob(i,1)

V = np.zeros((21,21))
action = np.zeros((21,21))


def policy_evaluate():
    delta = 0
    for i in range(21):
        for j in range(21):
            v = V[i,j]
            a = action[i,j] //从0挪到1的数量[-5,5]
            if (a>i): a = i
            if (a<-j) a = -j
            n = min(20,int(i-a))
            m = min(20,int(j+a))
            tmp_v = -np.abs(a)*move_spend //移车代价
            for i1 in range(21):
                for j1 in range(21):
                    tmp_v += trans_prob[0,n,i1]*trans_prob[1,m,j1]*(R[0,n]+R[1,m]+d)
            

