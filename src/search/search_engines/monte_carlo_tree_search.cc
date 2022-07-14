#include "monte_carlo_tree_search.h"

#include "../open_list_factory.h"
#include "../option_parser.h"

#include "../algorithms/ordered_set.h"
#include "../task_utils/successor_generator.h"
#include "../task_utils/task_properties.h"
#include "../utils/logging.h"
#include "../utils/rng.h"
#include "../utils/rng_options.h"
#include "../plugin.h"

#include <algorithm>
#include <limits.h>
#include <vector>
#include <cmath>

using namespace std;


namespace monte_carlo_tree_search {
MonteCarloTreeSearch::MonteCarloTreeSearch(const Options &opts)
    : SearchEngine(opts),
    p(opts.get<double>("p")),
    reopen_closed_nodes(opts.get<bool>("reopen_closed_nodes")),
    eps(opts.get<double>("epsilon")),
    delt(opts.get<double>("delta")),
    heuristic(opts.get<shared_ptr<Evaluator>>("h")),
    tree_search_space(state_registry, log)
    {
    /*
      We initialize current_eval_context in such a way that the initial node
      counts as "preferred".
    */
}
void MonteCarloTreeSearch::initialize() {
    State initial_state = state_registry.get_initial_state();
    TreeSearchNode init = tree_search_space.get_node(initial_state);
    EvaluationContext init_eval(initial_state,0,true,&statistics);
    int h = (init_eval.get_result(heuristic.get())).get_evaluator_value();
    init.open_initial(h);
    statistics.inc_evaluated_states();
}

bool MonteCarloTreeSearch::check_goal_and_set_plan(const State &state) {
    if (task_properties::is_goal_state(task_proxy, state)) {
        Plan plan;
        tree_search_space.trace_path(state, plan);
        set_plan(plan);
        return true;
    }
    return false;
}

State MonteCarloTreeSearch::select_next_leaf_node(const State state){
    TreeSearchNode node = tree_search_space.get_node(state);
    StateID par = node.get_parent();
    node.inc_visited();
    assert(!node.is_new() && !node.is_dead_end());
    if(node.is_open()){
        return state;
    }
    vector<StateID> children = node.get_children();
    assert(!children.empty());
    if(children.size() == 1){
        return select_next_leaf_node(state_registry.lookup_state(children.at(0)));
    }
    int min_visits = INT_MAX;
    for(StateID sid : children){
        State succ_state = state_registry.lookup_state(sid);
        TreeSearchNode succ_node = tree_search_space.get_node(succ_state);
        int v = succ_node.get_visited();
        if(min_visits > v){
            min_visits = v;
        }
    }
    //PAC-bound with Median elimination
    int l = node.get_l();
    double eps_l = (l==0)?eps/4:eps*pow(0.75,l)/4;
    double delt_l = (l==0)?delt/2:delt*pow(0.5,l)/2;
    double visit_bound = 1/(pow(eps_l/2,2)*logl(3/delt_l)) + 1.0;
    if(min_visits > visit_bound){ 
        double mean = 0;
        double median = 0;
        vector<double> rewards = {};
        for(StateID sid : node.get_children()){
            State succ_state = state_registry.lookup_state(sid);
            TreeSearchNode succ_node = tree_search_space.get_node(succ_state);
            int v = succ_node.get_visited();
            double r = succ_node.get_reward();
            mean += (r/v);
            rewards.push_back((r/v));
        }
        mean /= node.get_children().size();
        if(node.get_children().size() % 2 == 0){
            double med1 = INT_MAX;
            double med2 = INT_MAX;
            for(double r : rewards){
                double diff1 = abs(med1 - mean);
                double diff2 = abs(med2 - mean);
                if(diff1 > abs(r - mean)){
                    double swap = med1;
                    med1 = r;
                    med2 = swap;
                }else if(diff2 > abs(r - mean)){
                    med2 = r;
                }
            }
            median = (med1 + med2) / 2;
        }else{
            double med = INT_MAX;
            for(double r : rewards){
                double diff = abs(med - mean);
                if(diff > abs(r - mean)){
                    med = r;
                }
            }
            median = med;
        }
        for(StateID sid : node.get_children()){
            State succ_state = state_registry.lookup_state(sid);
            TreeSearchNode succ_node = tree_search_space.get_node(succ_state);
            if(succ_node.get_reward()/succ_node.get_visited() > median){
                //cout << "median removal: "<< state.get_id() << endl;
                if(succ_node.get_best_h() != node.get_best_h()){
                    node.remove_child(sid);
                    node.add_child_to_forgotten(sid);
                }
            }
            succ_node.reset_visited();
        }
        node.inc_l();
    }
    double prob = drand48();
    //cout << eps << endl;
    bool epsilon_greedy = p >= prob;
    vector<State> min_state = vector<State>();
    int min_h = INT_MAX;
    for(StateID sid : children){
        State succ_state = state_registry.lookup_state(sid);
        TreeSearchNode succ_node = tree_search_space.get_node(succ_state);
        int h = succ_node.get_best_h();
        if(epsilon_greedy || h == min_h){
            min_state.push_back(succ_state);
        }else if(h < min_h){
            min_h = h;
            min_state = {succ_state};
        }
    }
    assert(!min_state.empty());
    State succ = min_state.at(rand() % min_state.size());
    return select_next_leaf_node(succ);
}

SearchStatus MonteCarloTreeSearch::expand_tree(const State state){
    TreeSearchNode node = tree_search_space.get_node(state);
    assert(node.is_open());
    node.close();
    statistics.inc_expanded();
    vector<OperatorID> successor_operators;
    successor_generator.generate_applicable_ops(
    state, successor_operators);
    if(successor_operators.empty()){
        //cout << "mark:" << state.get_id() << endl;
        node.mark_as_dead_end();
        node.set_best_h(INT_MAX);
        statistics.inc_dead_ends();
        return IN_PROGRESS;
    }
    for (OperatorID op_id : successor_operators) {
        statistics.inc_generated();
        OperatorProxy op = task_proxy.get_operators()[op_id];
        State succ_state = state_registry.get_successor_state(state, op);
        TreeSearchNode succ_node = tree_search_space.get_node(succ_state);
        StateID succ_id = succ_state.get_id();
        int succ_g = succ_node.get_real_g();
        if(succ_node.is_new()){
            node.add_child(succ_id);
            EvaluationContext succ_eval_context(
            succ_state, succ_g, true, &statistics);
            statistics.inc_evaluated_states();
            int h  = succ_eval_context.get_result(heuristic.get()).get_evaluator_value();
            succ_node.open(node, op, get_adjusted_cost(op), h);
            succ_node.add_reward(h);
            if(h >= bound){
                succ_node.mark_as_dead_end();
                succ_node.set_best_h(INT_MAX);//TODO : remove child from parents children list
                node.remove_child(succ_id);
            }
        }else if(succ_node.is_closed() && reopen_closed_nodes){
            int new_succ_g = node.get_real_g() + op.get_cost();
            if(new_succ_g < succ_g){
                State previous_parent = state_registry.lookup_state(succ_node.get_parent());
                TreeSearchNode pred_node = tree_search_space.get_node(previous_parent);//previous parent node
                StateID succ_id = succ_node.get_state().get_id();
                
                succ_node.update_g(succ_g - new_succ_g);
                reopen_g(succ_state,succ_g - new_succ_g); // recursive g_update

                if(succ_node.get_parent() == state.get_id()){
                    continue;
                }
                //cout<< "old parent:"<<previous_parent.get_id() << " new parent:" << state.get_id() << endl;
                pred_node.remove_child(succ_id);//remove child from old parent
                node.add_child(succ_id);//new parent node is state
                    
                succ_node.reopen(node,op,get_adjusted_cost(op));
                statistics.inc_reopened();

                back_propagate(previous_parent);//We bp this because it might now contain a dead-end/higher best-h
            }
        }
        if(check_goal_and_set_plan(succ_state)){
            return SOLVED;
        }
    }

    return IN_PROGRESS;
}

void MonteCarloTreeSearch::reopen_g(State state,int g_diff){
    TreeSearchNode node = tree_search_space.get_node(state);
    if(node.is_dead_end() || node.is_open())
        return;
    for(StateID s : node.get_children()){
        State c_state = state_registry.lookup_state(s);
        TreeSearchNode c_node = tree_search_space.get_node(c_state);
        c_node.update_g(g_diff);
        reopen_g(c_state,g_diff);
    }
}

void MonteCarloTreeSearch::back_propagate(State state){
    TreeSearchNode node = tree_search_space.get_node(state);
    bool dead_end = true;
    int min_h = INT_MAX;
    for (StateID child : node.get_children()) {
        State child_state = state_registry.lookup_state(child);
        TreeSearchNode child_node = tree_search_space.get_node(child_state);
        int h_child = child_node.get_best_h();
        if(child_node.is_dead_end() || (h_child == INT_MAX)){
            continue;
        }
        if(h_child < min_h)
            min_h = h_child;
        dead_end = false;
    }
    StateID pred_id = node.get_parent();
    if(dead_end && !node.is_dead_end()) {
        if(node.is_forgotten_empty()){//If we havent eliminated any successors of this node.
            assert(min_h == INT_MAX);
            //cout << "children:" << node.get_children() << endl;
            //cout << "mark bp:" << state.get_id() << endl;
            node.mark_as_dead_end();
            if(pred_id != StateID::no_state){
                TreeSearchNode pred_node = tree_search_space.get_node(state_registry.lookup_state(pred_id));
                pred_node.remove_child(state.get_id());
            }
            node.set_best_h(min_h);
            statistics.inc_dead_ends();
        }else {
            node.add_forgotten_to_child(); //Add forgotten children back to the children
            back_propagate(state); //back propagating with new successors
            return; // We dont want to bp a second time.
        }
    }else if(!dead_end){
        //int curr_h = node.get_best_h();
        //if(curr_h == min_h){
        //    return;
        //}
        assert(min_h != INT_MAX);
        node.add_reward(min_h);
        node.set_best_h(min_h);
    }
    OperatorID pred_op = node.get_operator();
    if(pred_id != StateID::no_state && pred_op != OperatorID::no_operator){
        State pred = state_registry.lookup_state(pred_id);
        back_propagate(pred);
    }   
}

SearchStatus MonteCarloTreeSearch::step() {
    State init = state_registry.get_initial_state();
    TreeSearchNode init_node = tree_search_space.get_node(init);
    if(init_node.is_dead_end())
        return FAILED;
    State leaf = select_next_leaf_node(init);
    //cout << "leaf:" << leaf.get_id() << endl;
    SearchStatus status = expand_tree(leaf);
    back_propagate(leaf);
    return status;
}

void MonteCarloTreeSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    tree_search_space.print_statistics();
}
static shared_ptr<SearchEngine> _parse(OptionParser &parser) {
    parser.document_synopsis("Monte carlo tree search", "");

    parser.add_option<shared_ptr<Evaluator>>(
        "h",
        "set heuristic.");

    parser.add_option<double>("p",
                         "probability", "0.001");
    parser.add_option<bool>("reopen_closed_nodes",
                         "Reopen", "false");
    parser.add_option<double>("epsilon",
                         "ME coefficient", "3");
    parser.add_option<double>("delta",
                         "confidence", "0.05");
    SearchEngine::add_options_to_parser(parser);
    Options opts = parser.parse();

    shared_ptr<monte_carlo_tree_search::MonteCarloTreeSearch> engine;
    if (!parser.dry_run()) {
        engine = make_shared<monte_carlo_tree_search::MonteCarloTreeSearch>(opts);
    }

    return engine;
}

static Plugin<SearchEngine> _plugin("mcts", _parse);
}