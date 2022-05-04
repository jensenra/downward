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

using namespace std;

namespace monte_carlo_tree_search {
MonteCarloTreeSearch::MonteCarloTreeSearch(const Options &opts)
    : SearchEngine(opts),
    heuristic(opts.get<shared_ptr<Evaluator>>("h")),
    //current_state(state_registry.get_initial_state()),
    current_predecessor_id(StateID::no_state),
    current_operator_id(OperatorID::no_operator),
    current_g(0),
    current_real_g(0),
    //current_eval_context(current_state, 0, true, &statistics),
    //current_h(current_eval_context.get_result(heuristic.get()).get_evaluator_value()),
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
     int current_g = node.get_g();
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
    assert(!node.is_new());
    if(node.is_open()){
        //Move to state with state
        return state;
    }
    vector<StateID> children = node.get_children();
    //TODO: If children empty, mark a dead end
    if(children.empty()){
        node.mark_as_dead_end();
        State* ptr = nullptr;
        return *ptr;
    }
    //TODO: Add epsilon exploration
    vector<State> min_state = vector<State>();
    int min_h = numeric_limits<int>::max();
    for(StateID sid : children){
        State succ_state = state_registry.lookup_state(sid);
        TreeSearchNode succ_node = tree_search_space.get_node(succ_state);
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
}

SearchStatus MonteCarloTreeSearch::expand_tree(const State state){
   // Invariants:
    // - current_state is the next state for which we want to compute the heuristic.
    // - current_predecessor is a permanent pointer to the predecessor of that state.
    // - current_operator is the operator which leads to current_state from predecessor.
    // - current_g is the g value of the current state according to the cost_type
    // - current_real_g is the g value of the current state (using real costs)
    TreeSearchNode node = tree_search_space.get_node(state);
    assert(node.is_open());
    vector<OperatorID> successor_operators;
    successor_generator.generate_applicable_ops(
    state, successor_operators);
    //TODO: deal with no appl. ops (aka dead end problems)
    assert(!successor_operators.empty());
    for (OperatorID op_id : successor_operators) {
        OperatorProxy op = task_proxy.get_operators()[op_id];
        State succ_state = state_registry.get_successor_state(state, op);
        TreeSearchNode succ_node = tree_search_space.get_node(succ_state);
        if(check_goal_and_set_plan(succ_state)){
            return SOLVED;
            }
        if(succ_node.is_new()){
            StateID succ_id = succ_state.get_id();
            int succ_g = succ_node.get_g();
            node.add_child(succ_id);
            EvaluationContext succ_eval_context(
            succ_state, succ_g, true, &statistics);
            int h  = succ_eval_context.get_result(heuristic.get()).get_evaluator_value();
            succ_node.open(node, op, op.get_cost(), h);
        }else{
            //Handle reopened nodes
            assert(false);
        } 
    }
    /*
    TreeSearchNode node = tree_search_space.get_node(current_state);
    TreeSearchNode pred = tree_search_space.get_node(state_registry.lookup_state(current_predecessor_id));
    bool reopen = reopen_closed_nodes && !node.is_new() &&
        !node.is_dead_end() && (current_g < node.get_g());
    if (node.is_new() || reopen) {
        statistics.inc_evaluated_states();
        if (true) {
            // TODO: Generalize code for using multiple evaluators.
            if (current_predecessor_id == StateID::no_state) {
                node.open_initial();
                if (search_progress.check_progress(current_eval_context))
                    statistics.print_checkpoint_line(current_g);
            } else {
                OperatorProxy current_operator = task_proxy.get_operators()[current_operator_id];
                if (reopen) {
                    node.reopen(pred, current_operator, get_adjusted_cost(current_operator));
                    statistics.inc_reopened();
                } else {
                    node.open(pred, current_operator, get_adjusted_cost(current_operator));
                }
            }
            node.close();
            if (check_goal_and_set_plan(current_state)){    
                return SOLVED;
                }
            if (search_progress.check_progress(current_eval_context)) {
                statistics.print_checkpoint_line(node.get_g());
                //reward_progress();
            }
            generate_successors(current_state, current_eval_context);
            statistics.inc_expanded();
        } else {//We need this for reopening
            node.mark_as_dead_end();
            statistics.inc_dead_ends();
        }
        if (current_predecessor_id == StateID::no_state) {
            print_initial_evaluator_values(current_eval_context, log);
        }
    }
        */
    return IN_PROGRESS;
}

void MonteCarloTreeSearch::backpropagation(){  
    while(current_predecessor_id != StateID::no_state && current_operator_id != OperatorID::no_operator){
        update_best_h();
        State pred = state_registry.lookup_state(tree_search_space.get_node(current_state).get_parent());
        TreeSearchNode pred_node = tree_search_space.get_node(pred);
        EvaluationContext pred_eval(pred,pred_node.get_g(),true,&statistics);
        move_to_state(pred, pred_eval, pred_node.get_parent(), pred_node.get_operator());
    }      
}

void MonteCarloTreeSearch::move_to_state(State state, EvaluationContext eval_context, StateID pred, OperatorID op_id){  
    current_state = state;
    current_eval_context = eval_context;
    current_predecessor_id = pred;
    current_operator_id = op_id;
    State current_predecessor = state_registry.lookup_state(current_predecessor_id);
    OperatorProxy current_operator = task_proxy.get_operators()[current_operator_id];
    assert(task_properties::is_applicable(current_operator, current_predecessor));
    TreeSearchNode curr_node = tree_search_space.get_node(current_state);
    current_g = curr_node.get_g();
    current_real_g = curr_node.get_real_g();
    current_h = curr_node.get_best_h();  
}

void MonteCarloTreeSearch::update_best_h(){  
    TreeSearchNode curr_node = tree_search_space.get_node(current_state); 
    vector<StateID> children = curr_node.get_children();
    int min_h(numeric_limits<int>::max());
    for (StateID child : children) {
        TreeSearchNode child_node = tree_search_space.get_node(state_registry.lookup_state(child));
        int h_child = child_node.get_best_h();
        min_h = (h_child < min_h) ? h_child : min_h;
    }
    int h_curr = curr_node.get_best_h();
    if(h_curr > min_h)
        curr_node.set_best_h(min_h);  
}

SearchStatus MonteCarloTreeSearch::step() {
    // Invariants:
    // - current_state is the next state for which we want to compute the heuristic.
    // - current_predecessor is a permanent pointer to the predecessor of that state.
    // - current_operator is the operator which leads to current_state from predecessor.
    // - current_g is the g value of the current state according to the cost_type
    // - current_real_g is the g value of the current state (using real costs)
    cout << "Hi" <<endl;
    State leaf = select_next_leaf_node(state_registry.get_initial_state());
    cout << "a" << endl;
    SearchStatus status = expand_tree(leaf);
    cout << "b" << endl;
    backpropagation();
    cout << "c" << endl;
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
