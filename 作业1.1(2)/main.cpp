#include <ctime>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include "tictactoe.hpp"

using namespace std;

class TicTacToePolicyBase{
    public:
        virtual TicTacToe::Action operator()(const TicTacToe::State& state) const = 0;
};

// randomly select a valid action for the step.
class TicTacToePolicyRandom : public TicTacToePolicyBase{
    public:
        TicTacToe::Action operator()(const TicTacToe::State& state) const {
            vector<TicTacToe::Action> actions = state.action_space();
            int n_action = actions.size();
            int action_id = rand() % n_action;
            if (state.turn == TicTacToe::PLAYER_X){
                return actions[action_id];
            } else {
                return actions[action_id];
            }
        }
        TicTacToePolicyRandom(){
            srand(time(nullptr));
        }
};

// select the first valid action.
class TicTacToePolicyDefault : public TicTacToePolicyBase{
    private:
        unordered_map<int,pair<int,int> >assume_value;   //unordered_map，记录每种状态达到的次数以及其胜利的次数，以此来计算状态值估计

    public:
        TicTacToe::Action operator()(const TicTacToe::State& state) const {
            vector<TicTacToe::Action> actions = state.action_space();
            if (state.turn == TicTacToe::PLAYER_X){
                double maxnow = 0;
                int n_action = actions.size();
                auto take_action = actions[0];
                for (int i=0;i<n_action;i++){
                    auto action = actions[i];
                    TicTacToe::State try_state(state);
                    try_state.put(action);
                    auto pa_ = (assume_value.find(state.board));
                    if (pa_ == assume_value.end()) continue;
                    auto pa = (*pa_).second;
                    int win_number = pa.first;
                    int get_number = pa.second;
                    if (get_number == 0) continue;
                    if ((double)win_number/get_number > maxnow)
                        maxnow = (double)win_number/get_number;
                        take_action = action;
                }
                return take_action;
            } else {
                return actions[0];
            }
        }
        TicTacToePolicyDefault(){
        }

        void Train(){     //训练
            srand(19260817);
            int turns = 10000000;
            puts("Now I'm training 0/10000000");
            while (turns--){
                TicTacToe env(false);
                if (turns % 1000000 == 0)
                {
                    printf("Now I'm training %d/10000000\n",10000000-turns);
                }
                while (true){
                    TicTacToe::State state = env.get_state();
                    TicTacToe::Action action;
                    vector<TicTacToe::Action> actions = state.action_space();
                    if (state.turn == TicTacToe::PLAYER_X){
                        int n_action = actions.size();
                        int action_id = rand() % n_action;
                        action = actions[action_id];
                    } else
                        action = actions[0];
                    env.step(action);
                    if (env.done()) break;
                } 
                int winner = env.winner(),win = 0;
                if (winner == TicTacToe::PLAYER_X) win = 1;
                while (true){
                    TicTacToe::State state = env.get_state();
                    assume_value[state.board].second += 1;
                    if (win) assume_value[state.board].first += win;
                    if (!env.step_back()) break;
                }
            }
        }
};


#include <chrono>
#include <thread>

// randomly select action
int main(){
    bool done = false;
    // set verbose true
    
    TicTacToePolicyDefault policy;
    policy.Train();
    //TicTacToePolicyRandom policy;
    TicTacToe env(true);
    while (not done){
        TicTacToe::State state = env.get_state();
        TicTacToe::Action action = policy(state);
        env.step(action);
        done = env.done();
        // env.step_back();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    int winner = env.winner();
    return 0;
};