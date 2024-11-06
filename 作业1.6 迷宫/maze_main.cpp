#include <ctime>
#include <cmath>
#include "maze.hpp"

inline int Rand(){
    long long x = rand();
    x=(x<<16)+rand();
    return x%100000000; 
}

class MazePolicyBase{
    public:
        virtual int operator()(const MazeEnv::State& state) const = 0;
};

class MazePolicyQLearning : public MazePolicyBase{
    public:
        int operator()(const MazeEnv::State& state) const {
            int best_action = 0;
            double best_value = q[locate(state, 0)];
            double q_s_a;
            for (int action = 1; action < 4; ++ action){
                q_s_a = q[locate(state, action)];
                if (q_s_a > best_value){
                    best_value = q_s_a;
                    best_action = action;
                }
            }
            return best_action;
        }

        MazePolicyQLearning(const MazeEnv& e) : env(e) {
            epsilon = 0.1;
            alpha = 0.1;
            gamma = 0.95;
            q = new double[e.max_x * e.max_y * 4];
            srand(2022);
            for (int i = 0; i < e.max_x * e.max_y * 4; ++ i){
                q[i] = 1.0 / (rand() % (e.max_x * e.max_y) + 1);
            }
        }

        ~MazePolicyQLearning(){
            delete []q;
        }

        void learn(int iter=3000, int verbose_freq=10){
            bool done;
            int action, next_action;
            double reward;
            int episode_step;
            MazeEnv::State state, next_state;
            MazeEnv::StepResult step_result;

            for (int i = 0; i < iter; ++ i){
                state = env.reset();
                done = false;
                episode_step = 0;
                while (not done){
                    action = epsilon_greedy(state);
                    step_result = env.step(action);
                    next_state = step_result.next_state;
                    reward = step_result.reward;
                    done = step_result.done;
                    ++ episode_step;
                    next_action = (*this)(next_state);
                    q[locate(state, action)] += alpha * (gamma * q[locate(next_state, next_action)] + reward - q[locate(state, action)]);
                    state = next_state;
                }
                if (i % verbose_freq == 0){
                    cout << i <<" episode_step: " << episode_step << endl;
                }
            }
        }

        int epsilon_greedy(MazeEnv::State state) const {
            if (Rand() % 100000 < epsilon * 100000) {
                return rand() % 4;
            }
            return (*this)(state);
        }

        inline int locate(MazeEnv::State state, int action) const {
            return state.second * env.max_x * 4 + state.first * 4 + action;
        }

        void print_policy() const {
            static const char action_vis[] = "<>v^";
            int action;
            MazeEnv::State state;
            for (int i = 0; i < env.max_y; ++ i){
                for (int j = 0; j < env.max_x; ++ j){
                    state = MazeEnv::State(j, i);
                    if (not env.is_valid_state(state)){
                        cout << "#";
                    } else if (env.is_goal_state(state)){
                        cout << "G";
                    } else {
                        action = (*this)(MazeEnv::State(j, i));
                        cout << action_vis[action];
                    }
                }
                cout << endl;
            }
            cout << endl;
        }

    private:
        MazeEnv env;
        double *q;
        double epsilon, alpha, gamma;
};

class MazePolicyDynaQ : public MazePolicyBase{
    public:
        int operator()(const MazeEnv::State& state) const {
            int best_action = 0;
            double best_value = q[locate(state, 0)];
            double q_s_a;
            for (int action = 1; action < 4; ++ action){
                q_s_a = q[locate(state, action)];
                if (q_s_a > best_value){
                    best_value = q_s_a;
                    best_action = action;
                }
            }
            return best_action;
        }

        MazePolicyDynaQ(const MazeEnv& e) : env(e) {
            epsilon = 0.1;
            alpha = 0.1;
            gamma = 0.95;
            q = new double[e.max_x * e.max_y * 4];
            model = new pair<double,MazeEnv::State> [e.max_x*e.max_y*4];
            visit = new bool [e.max_x*e.max_y*4];
            srand(2022);
            for (int i = 0; i < e.max_x * e.max_y * 4; ++ i){
                q[i] = 1.0 / (rand() % (e.max_x * e.max_y) + 1);
            }
        }

        ~MazePolicyDynaQ(){
            delete []q;
            delete [] model;
            delete [] visit;
        }

        void learn(int iter=3000, int verbose_freq=10, int n=5){
            bool done;
            int action, next_action;
            double reward;
            int episode_step;
            MazeEnv::State state, next_state;
            MazeEnv::StepResult step_result;
            vector<pair<MazeEnv::State,int> >Visited;
            double cumulative_reward = 0;

            for (int i = 0; i < iter; ++ i){
                if (i==1000)
                    env.maze_shortcut(8,3);  
                state = env.reset();
                done = false;
                episode_step = 0;
                while (not done){
                    action = epsilon_greedy(state);
                    step_result = env.step(action);
                    next_state = step_result.next_state;
                    reward = step_result.reward;
                    done = step_result.done;
                    ++ episode_step;
                    next_action = (*this)(next_state);
                    q[locate(state, action)] += alpha * (gamma * q[locate(next_state, next_action)] + reward - q[locate(state, action)]);
                    
                    model[locate(state,action)] = make_pair(reward,next_state);

                    if (!visit[locate(state,action)]){
                        Visited.push_back(make_pair(state,action));
                        visit[locate(state,action)] = true;
                    }
                    state = next_state;
                    for (int j=0;j<n && j<=Visited.size();j++){
                        int n_visited = Visited.size();
                        int chosen = rand()%n_visited;
                        auto model_content = model[locate(Visited[chosen].first,Visited[chosen].second)];
                        auto reward = model_content.first;
                        auto new_state = model_content.second;
                        next_action = (*this)(new_state);
                        q[locate(Visited[chosen].first,Visited[chosen].second)]+= \
                            alpha*(reward+gamma*q[locate(new_state,next_action)]-q[locate(Visited[chosen].first,Visited[chosen].second)]);
                    }
                }
                state = env.reset();
                auto action = (*this)(state);
                cumulative_reward += q[locate(state,action)]/episode_step;
                if (i % verbose_freq == 0){
                    cout << i <<" episode_step: " << episode_step << endl;
                }
            }
        }

        int epsilon_greedy(MazeEnv::State state) const {
            if (Rand() % 100000 < epsilon * 100000) {
                return rand() % 4;
            }
            return (*this)(state);
        }

        inline int locate(MazeEnv::State state, int action) const {
            return state.second * env.max_x * 4 + state.first * 4 + action;
        }

        void print_policy() const {
            static const char action_vis[] = "<>v^";
            int action;
            MazeEnv::State state;
            for (int i = 0; i < env.max_y; ++ i){
                for (int j = 0; j < env.max_x; ++ j){
                    state = MazeEnv::State(j, i);
                    if (not env.is_valid_state(state)){
                        cout << "#";
                    } else if (env.is_goal_state(state)){
                        cout << "G";
                    } else {
                        action = (*this)(MazeEnv::State(j, i));
                        cout << action_vis[action];
                    }
                }
                cout << endl;
            }
            cout << endl;
        }

    private:
        MazeEnv env;
        double *q;
        pair<double,MazeEnv::State> *model;
        bool *visit;
        double epsilon, alpha, gamma;
};

class MazePolicyDynaQplus : public MazePolicyBase{
    public:
        int operator()(const MazeEnv::State& state) const {
            int best_action = 0;
            double best_value = q[locate(state, 0)];
            double q_s_a;
            for (int action = 1; action < 4; ++ action){
                q_s_a = q[locate(state, action)];
                if (q_s_a > best_value){
                    best_value = q_s_a;
                    best_action = action;
                }
            }
            return best_action;
        }

        MazePolicyDynaQplus(const MazeEnv& e) : env(e) {
            epsilon = 0.1;
            alpha = 0.1;
            gamma = 0.95;
            q = new double[e.max_x * e.max_y * 4];
            model = new pair<double,MazeEnv::State> [e.max_x*e.max_y*4];
            visit = new bool [e.max_x*e.max_y*4];
            tm = new int [e.max_x*e.max_y*4];
            srand(2022);
            for (int i = 0; i < e.max_x * e.max_y * 4; ++ i){
                q[i] = 1.0 / (rand() % (e.max_x * e.max_y) + 1);
            }
        }

        ~MazePolicyDynaQplus(){

            delete [] q;
            delete [] model;
            delete [] visit;
            delete [] tm;
        }

        void learn(int iter=3000, int verbose_freq=10, int n=5,double k=0.005){
            bool done;
            int action, next_action;
            double reward;
            int episode_step;
            MazeEnv::State state, next_state;
            MazeEnv::StepResult step_result;
            vector<pair<MazeEnv::State,int> >Visited;
            double cumulative_reward = 0;

            for (int i = 0; i < iter; ++ i){
                if (i==1000)
                    env.maze_shortcut(8,3);  
                state = env.reset();
                done = false;
                episode_step = 0;
                while (not done){
                    action = epsilon_greedy(state);
                    step_result = env.step(action);
                    next_state = step_result.next_state;
                    reward = step_result.reward;
                    done = step_result.done;
                    ++ episode_step;
                    next_action = (*this)(next_state);
                    q[locate(state, action)] += alpha * (gamma * q[locate(next_state, next_action)] + reward - q[locate(state, action)]);
                    tm[locate(state,action)] = i;
                    model[locate(state,action)] = make_pair(reward,next_state);

                    if (!visit[locate(state,action)]){
                        Visited.push_back(make_pair(state,action));
                        visit[locate(state,action)] = true;
                    }
                    state = next_state;
                    for (int j=0;j<n && j<=Visited.size();j++){
                        int n_visited = Visited.size();
                        int chosen = rand()%n_visited;
                        auto model_content = model[locate(Visited[chosen].first,Visited[chosen].second)];
                        auto reward = model_content.first;
                        auto new_state = model_content.second;
                        next_action = (*this)(new_state);
                        int tau = i-tm[locate(Visited[chosen].first,Visited[chosen].second)];
                        q[locate(Visited[chosen].first,Visited[chosen].second)]+= \
                            alpha*(reward+gamma*q[locate(new_state,next_action)]-q[locate(Visited[chosen].first,Visited[chosen].second)] + k*sqrt(tau));
                    }
                    
                }
                state = env.reset();
                auto action = (*this)(state);
                cumulative_reward += q[locate(state,action)]/episode_step;
                if (i % verbose_freq == 0){
                    cout << i <<" episode_step: " << episode_step << endl;
                }
            }
        }

        int epsilon_greedy(MazeEnv::State state) const {
            if (Rand() % 100000 < epsilon * 100000) {
                return rand() % 4;
            }
            return (*this)(state);
        }

        inline int locate(MazeEnv::State state, int action) const {
            return state.second * env.max_x * 4 + state.first * 4 + action;
        }

        void print_policy() const {
            static const char action_vis[] = "<>v^";
            int action;
            MazeEnv::State state;
            for (int i = 0; i < env.max_y; ++ i){
                for (int j = 0; j < env.max_x; ++ j){
                    state = MazeEnv::State(j, i);
                    if (not env.is_valid_state(state)){
                        cout << "#";
                    } else if (env.is_goal_state(state)){
                        cout << "G";
                    } else {
                        action = (*this)(MazeEnv::State(j, i));
                        cout << action_vis[action];
                    }
                }
                cout << endl;
            }
            cout << endl;
        }

    private:
        MazeEnv env;
        double *q;
        pair<double,MazeEnv::State> *model;
        bool *visit;
        int *tm;
        double epsilon, alpha, gamma;
};


int main(){
    const int max_x = 9, max_y = 6;
    const int start_x = 3, start_y = 5;
    const int target_x = 8, target_y = 0;
    int maze[max_y][max_x] = {
        {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0},
        {0,1,1,1,1,1,1,1,1},
        {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0}
    };
    MazeEnv env(maze, max_x, max_y, start_x, start_y, target_x, target_y);
    env.reset();
    MazePolicyDynaQ policy(env);
    policy.learn();
    policy.print_policy();
    return 0;
}