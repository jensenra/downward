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
#include <limits>
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
    log << "Conducting monte carlo tree search, (real) bound = " << bound << endl;
    
    State initial_state = state_registry.get_initial_state();
    heuristic->notify_initial_state(initial_state);
    TreeSearchNode init = tree_search_space.get_node(initial_state);
    init.open_initial();
}

bool MonteCarloTreeSearch::check_goal_and_set_plan(const State &state) {
    if (task_properties::is_goal_state(task_proxy, state)) {
        log << "Solution found!" << endl;
        Plan plan;
        tree_search_space.trace_path(state, plan);
        set_plan(plan);
        return true;
    }
    return false;
}

void MonteCarloTreeSearch::generate_successors(State state, EvaluationContext eval_context) {
    vector<OperatorID> successor_operators ;
    successor_generator.generate_applicable_ops(
        state, successor_operators);

    statistics.inc_generated(successor_operators.size());
   
    TreeSearchNode node = tree_search_space.get_node(state);
     int current_g = node.get_real_g();
      int current_real_g = node.get_real_g();
    for (OperatorID op_id : successor_operators) {
        OperatorProxy op = task_proxy.get_operators()[op_id];
        int new_g = current_g + get_adjusted_cost(op);
        int new_real_g = current_real_g + op.get_cost();
        if (new_real_g < bound) {
            EvaluationContext new_eval_context(
                eval_context, new_g, false, nullptr);
        }
    }
}

State MonteCarloTreeSearch::select_next_leaf_node(const State state){
    TreeSearchNode node = tree_search_space.get_node(state);
    if(node.is_new()){
        StateID pred = node.get_parent();
        State parent = state_registry.lookup_state(pred);
        return select_next_leaf_node(parent);
    }
    if(node.is_open()){
        //cout << "open:" << state.get_id() << endl;
        return state;
    }
    vector<StateID> children = node.get_children();
    if(children.empty()){
        node.mark_as_dead_end();
        back_propagate(state);
        return state;
    }
    double eps = 1e-4;
    if((double) (rand()/RAND_MAX) > eps){
        vector<State> min_state = vector<State>();
        int min_h = numeric_limits<int>::max();
        for(StateID sid : children){
            if(sid == StateID::no_state)
                continue;
            State succ_state = state_registry.lookup_state(sid);
            TreeSearchNode succ_node = tree_search_space.get_node(succ_state);
            if(succ_node.is_dead_end())
                continue;
            int h = succ_node.get_best_h();
            if(h < min_h){
                min_h = h;
                min_state = {succ_state};
            }else if (h == min_h){
                min_state.push_back(succ_state);
            }
        }
        State succ = min_state.at(rand() % min_state.size());
        return select_next_leaf_node(succ);
    } else {
        StateID succ_id = children.at(rand() % children.size());
        State succ = state_registry.lookup_state(succ_id);
        return select_next_leaf_node(succ);
    }
}

SearchStatus MonteCarloTreeSearch::expand_tree(const State state){
   // Invariants:
    // - current_state is the next state for which we want to compute the heuristic.
    // - current_predecessor is a permanent pointer to the predecessor of that state.
    // - current_operator is the operator which leads to current_state from predecessor.
    // - current_g is the g value of the current state according to the cost_type
    // - current_real_g is the g value of the current state (using real costs)
    TreeSearchNode node = tree_search_space.get_node(state);
    if(node.is_dead_end())
        return IN_PROGRESS;
    //cout << node.is_open() << endl;
    node.close();
    vector<OperatorID> successor_operators;
    successor_generator.generate_applicable_ops(
    state, successor_operators);
    //TODO: deal with no appl. ops (aka dead end problems)
    if(successor_operators.empty()){
        if(!node.is_dead_end())
            node.mark_as_dead_end();
        return IN_PROGRESS;
    }
    for (OperatorID op_id : successor_operators) {
        OperatorProxy op = task_proxy.get_operators()[op_id];
        State succ_state = state_registry.get_successor_state(state, op);
        TreeSearchNode succ_node = tree_search_space.get_node(succ_state);
        if(succ_node.is_dead_end())
            continue;
        StateID succ_id = succ_state.get_id();
        int succ_g = succ_node.get_real_g();
        if(succ_node.is_new()){
            node.add_child(succ_id);
            EvaluationContext succ_eval_context(
            succ_state, succ_g, true, &statistics);
            int h  = succ_eval_context.get_result(heuristic.get()).get_evaluator_value();
            succ_node.open(node, op, get_adjusted_cost(op), h);
            back_propagate(state);
            //cout << "id: " << succ_node.get_operator().get_index() << endl;
        }else{
            //cout << "reop" << endl;
            int new_succ_g = node.get_real_g() + op.get_cost();
            if(new_succ_g < succ_g){
                //cout << "if reop" << endl;
                node.add_child(succ_id);
                //cout << "reop new child" << endl;
                succ_node.update_g(succ_g - new_succ_g);
                State previous_parent = state_registry.lookup_state(succ_node.get_parent());
                TreeSearchNode pred_node = tree_search_space.get_node(previous_parent);//previous parent node
                State current_parent = node.get_state();//new parent node
                StateID curr_id = succ_node.get_state().get_id();
                pred_node.remove_child(curr_id);//remove child from old parent
                //cout << "reopchild" << endl;
                back_propagate(previous_parent);//We bp this because it might now contain a dead-end/higher best-h
                //cout << "reopbp1" << endl;
                back_propagate(state);//We bp this because it might now contain a lower best-h
                //cout << "rec updt" << endl;
                succ_node.reopen(node,op,get_adjusted_cost(op));
                reopen_g(succ_state,succ_g - new_succ_g); // recursive g_update
                //cout << "reopg" << endl;
            }
        }
        if(check_goal_and_set_plan(succ_state)){
            cout << "goal" << succ_state.get_id() << endl;
            return SOLVED;
        }
    }

    return IN_PROGRESS;
}

void MonteCarloTreeSearch::reopen_g(State state,int g_diff){
    TreeSearchNode node = tree_search_space.get_node(state);
    //cout << state.get_id() << endl;
    if(node.is_dead_end() || node.is_open())
        return;
    for(StateID s : node.get_children()){
        //cout << "for" << endl;
        if(s == StateID::no_state)
            continue;
        //cout << "forstate" << endl;
        State c_state = state_registry.lookup_state(s);
        TreeSearchNode c_node = tree_search_space.get_node(c_state);
        c_node.update_g(g_diff);
        //cout << "forupdt" << endl;
        reopen_g(c_state,g_diff);
        //cout << "forreopg" << endl;
    }
}

void MonteCarloTreeSearch::back_propagate(State state){
    TreeSearchNode node = tree_search_space.get_node(state);
    //back propagate dead-ends
    bool dead_end = true;
    EvaluationContext curr_eval_context(
    state, node.get_real_g(), true, &statistics);
    int h  = curr_eval_context.get_result(heuristic.get()).get_evaluator_value();
    int min_h(h);
    node.set_best_h(h);
    for (StateID child : node.get_children()) {
        if(child == StateID::no_state)
            continue;
        State child_state = state_registry.lookup_state(child);
        TreeSearchNode child_node = tree_search_space.get_node(child_state);
        int h_child = child_node.get_best_h();
        if(h_child < min_h)
            min_h = h_child;
        dead_end &= child_node.is_dead_end();
    }
    int h_curr = node.get_best_h();
    if(h_curr > min_h)
        node.set_best_h(min_h); 
    if(dead_end) 
        node.mark_as_dead_end();
    StateID pred_id = node.get_parent();
    OperatorID pred_op = node.get_operator();
    if(pred_id != StateID::no_state && pred_op != OperatorID::no_operator){
        State pred = state_registry.lookup_state(pred_id);
        back_propagate(pred);
    }      
}

SearchStatus MonteCarloTreeSearch::step() {
    // Invariants:
    // - current_state is the next state for which we want to compute the heuristic.
    // - current_predecessor is a permanent pointer to the predecessor of that state.
    // - current_operator is the operator which leads to current_state from predecessor.
    // - current_g is the g value of the current state according to the cost_type
    // - current_real_g is the g value of the current state (using real costs)
    State init = state_registry.get_initial_state();
    TreeSearchNode init_node = tree_search_space.get_node(init);
    if(init_node.is_dead_end())
        return FAILED;
    //cout << "Hi" <<endl;
    State leaf = select_next_leaf_node(init);
    //cout << "a" << endl;
    //cout << leaf.get_id() << endl;
    SearchStatus status = expand_tree(leaf);
    //cout << "b" << endl;
    back_propagate(leaf);
    //cout << "c" << endl;
    return status;
}

void MonteCarloTreeSearch::reward_progress() {
    //open_list->boost_preferred();
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
