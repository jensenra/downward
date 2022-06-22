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
    epsilon(opts.get<double>("epsilon")),
    reopen_closed_nodes(opts.get<bool>("reopen_closed_nodes")),
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
    EvaluationContext init_eval(initial_state,0,true, &statistics);
    statistics.inc_evaluated_states();
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
                eval_context, new_g, false, &statistics);
                statistics.inc_evaluated_states();

        }
    }
}

State MonteCarloTreeSearch::select_next_leaf_node(const State state){
    TreeSearchNode node = tree_search_space.get_node(state);
    assert(!node.is_new() && !node.is_dead_end());
    if(node.is_open()){
        return state;
    }
    vector<StateID> children = node.get_children();
    assert(!children.empty());
    double prob = drand48();
    bool epsilon_greedy = (epsilon >= prob);
    vector<State> min_state = vector<State>();
    int min_h = INT_MAX;
    for(StateID sid : children){
        State succ_state = state_registry.lookup_state(sid);
        TreeSearchNode succ_node = tree_search_space.get_node(succ_state);
        int h = succ_node.get_best_h();
        if(succ_node.is_dead_end() || h == INT_MAX){
            continue;
        }
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
    statistics.inc_generated(successor_operators.size());
    for (OperatorID op_id : successor_operators) {
        OperatorProxy op = task_proxy.get_operators()[op_id];
        State succ_state = state_registry.get_successor_state(state, op);
        TreeSearchNode succ_node = tree_search_space.get_node(succ_state);
        StateID succ_id = succ_state.get_id();
        int succ_g = succ_node.get_real_g();
        if(succ_node.is_new()){
            node.add_child(succ_id);
            EvaluationContext succ_eval_context(
            succ_state, node.get_g() + get_adjusted_cost(op), true, &statistics);
            statistics.inc_evaluated_states();
            int h  = succ_eval_context.get_result(heuristic.get()).get_evaluator_value();
            succ_node.open(node, op, get_adjusted_cost(op), h);
        }else if(succ_node.is_closed() && reopen_closed_nodes){
            int new_succ_g = node.get_real_g() + op.get_cost();
            if(new_succ_g < succ_g){
                statistics.inc_reopened();
                node.add_child(succ_id);
                succ_node.update_g(succ_g - new_succ_g);
                reopen_g(succ_state,succ_g - new_succ_g); // recursive g_update
                State previous_parent = state_registry.lookup_state(succ_node.get_parent());
                TreeSearchNode previous_parent_node = tree_search_space.get_node(previous_parent);//previous parent node
                StateID state_id = state.get_id();
                previous_parent_node.remove_child(state_id);//remove child from old parent
                succ_node.reopen(node,op,get_adjusted_cost(op));
                back_propagate(previous_parent);//We backprop because it might now contain higher best-h or be a dead-end
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
    assert(!node.is_new());
    if(node.is_dead_end() || node.is_open())
        return;
    vector<StateID> children = node.get_children();
    assert(!children.empty());
    for(StateID s : children){
        State succ_state = state_registry.lookup_state(s);
        TreeSearchNode succ_node = tree_search_space.get_node(succ_state);
        succ_node.update_g(g_diff);
        reopen_g(succ_state,g_diff);
    }
}

void MonteCarloTreeSearch::back_propagate(State state){
    TreeSearchNode node = tree_search_space.get_node(state);
    bool dead_end = true;
    int min_h = INT_MAX;
    for (StateID child : node.get_children()) {
        if(child == StateID::no_state)
            continue;
        State child_state = state_registry.lookup_state(child);
        TreeSearchNode child_node = tree_search_space.get_node(child_state);
        int h_child = child_node.get_best_h();
        if(h_child < min_h)
            min_h = h_child;
        dead_end &= (child_node.is_dead_end() || (child_node.get_best_h() == INT_MAX));
    }
    if(dead_end) {
        node.mark_as_dead_end();
        node.set_best_h(min_h);
        statistics.inc_dead_ends();
        assert(min_h == INT_MAX);
    }else{
        //Still feel there is a case missing here
        int curr_h = node.get_best_h();
        if(curr_h == min_h){
            return;
        }
        assert(min_h != INT_MAX);
        node.set_best_h(min_h);
    }
    StateID pred_id = node.get_parent();
    OperatorID pred_op = node.get_operator();
    if(pred_id != StateID::no_state && pred_op != OperatorID::no_operator){
        State pred = state_registry.lookup_state(pred_id);
        back_propagate(pred);
    }   
}

SearchStatus MonteCarloTreeSearch::step() {
    // Invariants:
    // - 
    State init = state_registry.get_initial_state();
    TreeSearchNode init_node = tree_search_space.get_node(init);
    if(init_node.is_dead_end())
        return FAILED;
    State leaf = select_next_leaf_node(init);
    SearchStatus status = expand_tree(leaf);
    back_propagate(leaf);
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

    parser.add_option<shared_ptr<Evaluator>>(
        "h",
        "set heuristic.");

    parser.add_option<double>("epsilon",
                         "Epsilon", "0.001");
    parser.add_option<bool>("reopen_closed_nodes",
                         "Reopen", "false");
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
