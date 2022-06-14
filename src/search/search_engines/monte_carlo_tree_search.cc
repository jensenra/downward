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
    heuristic(opts.get<shared_ptr<Evaluator>>("h")),
    tree_search_space(state_registry, log)
    {
    /*
      We initialize current_eval_context in such a way that the initial node
      counts as "preferred".
    */
}
void MonteCarloTreeSearch::initialize() {
    //log << "Conducting monte carlo tree search, (real) bound = " << bound << endl;
    
    State init = state_registry.get_initial_state();
    //cout << "init: " << init.get_id() << endl;
    vector<OperatorID> successor_operators;
    successor_generator.generate_applicable_ops(init, successor_operators);
    heuristic->notify_initial_state(init);
    TreeSearchNode init_node = tree_search_space.get_node(init);
    EvaluationContext init_eval(init,0,true,&statistics);
    int init_h =init_eval.get_result(heuristic.get()).get_evaluator_value();
    init_node.open_initial();
    init_node.set_best_h(init_h);
}

bool MonteCarloTreeSearch::check_goal_and_set_plan(const State &state) {
    if (task_properties::is_goal_state(task_proxy, state)) {
        //log << "Solution found!" << endl;
        Plan plan;
        tree_search_space.trace_path(state, plan);
        set_plan(plan);
        return true;
    }
    return false;
}

State MonteCarloTreeSearch::select_next_leaf_node(const State state){
    TreeSearchNode node = tree_search_space.get_node(state);
    vector<StateID> children = node.get_children();
    if(node.is_dead_end()){
        if(children.empty()){
            //cout << "Node Selection warning: Dead-end node with children." << endl;
            //exit(222);
        }
        node.set_best_h(INT_MAX);
        back_propagate_best_h(state);
        State parent = state_registry.lookup_state(node.get_parent());
        return select_next_leaf_node(parent);//Climb back up
    }
    else if(node.is_new()){
        //cout << "Node selection warning: New node selection." << endl;
        State parent = state_registry.lookup_state(node.get_parent());
        return select_next_leaf_node(parent);//Climb back up
    }
    else if(node.is_open()){
        return state;
    }
    else if(children.empty()){
        //cout << "Node Selection warning: Closed node without children (" << state.get_id() << ")" << endl;
        return state;
    }
    double eps = 0.001;
    double prob = drand48();
    if(prob > eps){ //Exploitation
        vector<State> min_state = vector<State>();
        int min_h = INT_MAX;
        for(StateID s : children){
            if(s.operator==(StateID::no_state)){
                continue;
            }
            State s_state = state_registry.lookup_state(s);
            TreeSearchNode s_node = tree_search_space.get_node(s_state);
            int h = s_node.get_best_h();
            if(min_h > h){
                min_h = h;
                min_state = {};
                min_state.push_back(s_state);
            }
            else if (min_h == h){
                min_state.push_back(s_state);
            }
        }
        State successor = min_state[(rand() % min_state.size())];
        return select_next_leaf_node(successor);
    } else { //Exploration
        State successor = state_registry.lookup_state(children[(rand() % children.size())]);
        return select_next_leaf_node(successor);
    }
}

SearchStatus MonteCarloTreeSearch::expand_tree(const State state){
   // Invariants:
    // - current_real_g is the g value of the current state (using real costs)
    TreeSearchNode node = tree_search_space.get_node(state);
    if(node.is_dead_end()){
        //cout << "Expansion warning: Dead-end node picked for expansion." << endl;
        return IN_PROGRESS;
    }
    else if(node.is_new()){
        //cout << "Expansion warning: New node picked for expansion." << endl;
        return IN_PROGRESS;
    }
    vector<OperatorID> successor_operators;
    successor_generator.generate_applicable_ops(state, successor_operators);
    statistics.inc_generated(successor_operators.size());
    if(node.is_open()){
        node.close();
    }
    else if(node.is_closed()){
        //cout << "Expansion warning: Closed node picked for expansion." << endl;
    }
    if(successor_operators.empty()){
        node.mark_as_dead_end();
        back_propagate_dead_end(state);
        return IN_PROGRESS;
    }
    bool no_addition = true;
    vector<StateID> all_succs = {};
    for (OperatorID op_id : successor_operators) { 
        OperatorProxy op = task_proxy.get_operators()[op_id];
        State succ_state = state_registry.get_successor_state(state, op);
        StateID succ_id = succ_state.get_id();
        all_succs.push_back(succ_id);
    }
    //cout << "successors:" <<all_succs << endl;
    for (OperatorID op_id : successor_operators) {
        OperatorProxy op = task_proxy.get_operators()[op_id];
        State succ_state = state_registry.get_successor_state(state, op);
        TreeSearchNode succ_node = tree_search_space.get_node(succ_state);
        if(succ_node.is_dead_end()){
            continue;
        }
        StateID succ_id = succ_state.get_id();
        int succ_g = succ_node.get_real_g();
        int succ_h;
        if(succ_node.is_new()){
            //cout << "new_succ_id: " << succ_id << endl;
            no_addition = false;
            node.add_child(succ_id);
            succ_node.open(node, op, get_adjusted_cost(op));
            EvaluationContext succ_eval_context(succ_state, succ_g, true, &statistics);
            succ_h = succ_eval_context.get_result(heuristic.get()).get_evaluator_value();
            succ_node.set_best_h(succ_h);
            if(node.get_best_h() > succ_h){
                back_propagate_best_h(state);
            }
        }
        else{
            int new_succ_g = node.get_real_g() + op.get_cost();
            if(succ_g > new_succ_g) {
            //cout << "re_succ_id: " << succ_id << endl;
                no_addition = false;
                State previous_parent = state_registry.lookup_state(succ_node.get_parent());
                TreeSearchNode previous_parent_node = tree_search_space.get_node(previous_parent);//previous parent node
                previous_parent_node.remove_child(succ_id);//remove child from old parent
                node.add_child(succ_id);
                succ_node.reopen(node,op,get_adjusted_cost(op));
                forward_propagate_g(succ_state,succ_g - new_succ_g); // recursive g_update
                EvaluationContext succ_eval_context(succ_state, new_succ_g, true, &statistics);
                succ_h = succ_eval_context.get_result(heuristic.get()).get_evaluator_value();
                if(node.get_best_h() > succ_h){
                    back_propagate_best_h(state);
                }
                if(previous_parent_node.get_best_h() == succ_h){
                    back_propagate_best_h(previous_parent);
                }
            }
        }
        if(check_goal_and_set_plan(succ_state)){
            cout << "goal" << succ_state.get_id() << endl;
            return SOLVED;
        }
    }
    //cout << "children: "<< node.get_children() << endl;
    if(no_addition){
        node.mark_as_dead_end();
    }
    return IN_PROGRESS;
}

void MonteCarloTreeSearch::forward_propagate_g(State state,int g_diff){
    TreeSearchNode node = tree_search_space.get_node(state);
    for(StateID s : node.get_children()){
        if(s.operator==(StateID::no_state)){
            continue;
        }
        State c_state = state_registry.lookup_state(s);
        TreeSearchNode c_node = tree_search_space.get_node(c_state);
        c_node.update_g(g_diff);
        forward_propagate_g(c_state,g_diff);
    }
}

void MonteCarloTreeSearch::back_propagate_dead_end(State state){
    TreeSearchNode node = tree_search_space.get_node(state);
    State pred = state_registry.lookup_state(node.get_parent());
    bool dead_end = true;
    vector<StateID> children = tree_search_space.get_node(pred).get_children();
    if(children.size() == 0){
        //cout << "Child is not part of parent's children vector." << endl;
    }
    for(StateID s : children){
        TreeSearchNode s_node = tree_search_space.get_node(state_registry.lookup_state(s));
        dead_end &= s_node.is_dead_end();
    }
    if(dead_end)
        back_propagate_dead_end(pred);
    return;
}

void MonteCarloTreeSearch::back_propagate_best_h(State state){
    TreeSearchNode node = tree_search_space.get_node(state);
    vector<StateID> children = node.get_children();
    EvaluationContext curr_eval_context(state, node.get_real_g(), true, &statistics);
    int min_h  = curr_eval_context.get_result(heuristic.get()).get_evaluator_value();
    for(StateID s : children){
        TreeSearchNode s_node = tree_search_space.get_node(state_registry.lookup_state(s));
        int s_h = s_node.get_best_h();
        if(min_h > s_h){
            min_h = s_h;
        }
    }
    if(node.get_best_h() == min_h){
        return;
    }
    node.set_best_h(min_h);
    OperatorID op_id = node.get_operator();
    StateID pred_id = node.get_parent();
    if(op_id.operator==(OperatorID::no_operator) && pred_id.operator==(StateID::no_state)){
        back_propagate_best_h(state_registry.lookup_state(pred_id));
    }

}


SearchStatus MonteCarloTreeSearch::step() {
    // Invariants:
    // - current_real_g is the g value of the current state (using real costs)
    State init = state_registry.get_initial_state();
    TreeSearchNode init_node = tree_search_space.get_node(init);
    //cout << "id_init: " << init.get_id() << endl;
    if(init_node.is_dead_end()){
        return FAILED;
    }
    State leaf = select_next_leaf_node(init);
    //cout << "id: " << leaf.get_id() << endl;
    SearchStatus status = expand_tree(leaf);
    return status;
}

void MonteCarloTreeSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    tree_search_space.print_statistics();
}
static shared_ptr<SearchEngine> _parse(OptionParser &parser) {
    parser.document_synopsis("Monte carlo tree search", "");

    //parser.add_option<bool>("reopen_closed",
      //                      "reopen closed nodes", "false");
    parser.add_option<shared_ptr<Evaluator>>(
        "h",
        "set heuristic.");

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
