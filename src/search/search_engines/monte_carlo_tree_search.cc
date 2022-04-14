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
    current_h(numeric_limits<int>::max()),
    tree_search_space(state_registry, log) {
    /*
      We initialize current_eval_context in such a way that the initial node
      counts as "preferred".
    */
}
void MonteCarloTreeSearch::initialize() {
    log << "Conducting monte carlo tree search, (real) bound = " << bound << endl;
    
    State initial_state = state_registry.get_initial_state();
    heuristic->notify_initial_state(initial_state);
}

void MonteCarloTreeSearch::generate_successors(State state, EvaluationContext eval_context) {
    vector<OperatorID> successor_operators ;
    successor_generator.generate_applicable_ops(
        state, successor_operators);

    statistics.inc_generated(successor_operators.size());
   
    SearchNode node = search_space.get_node(state);
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

void MonteCarloTreeSearch::trial() {
    select_next_leaf_node();
}

void MonteCarloTreeSearch::select_next_leaf_node(){
    State init = state_registry.get_initial_state();
    TreeSearchNode* current_tree_node = &tree_search_space.get_node(init);
    EvaluationContext eval_context(init, 0, false, &statistics);
    if(current_tree_node->is_new())
        expand_tree(OperatorID::no_operator, init,eval_context);
    vector<StateID> children = current_tree_node->get_children();
    while(!children.empty()){
        children = current_tree_node->get_children();
        //TODO: Add epsilon exploration
        vector<State> min_state = vector<State>();
        int min_h = numeric_limits<int>::max();
        for(StateID sid : children){
            State succ_state = state_registry.lookup_state(sid);
            SearchNode succ_node = search_space.get_node(succ_state);
            int succ_g = succ_node.get_g();
            EvaluationContext succ_eval_context(
            succ_state, succ_g, false, &statistics);
            int h = succ_eval_context.get_result(heuristic.get()).get_evaluator_value();
            if(h < min_h){
                min_h = h;
                min_state = vector<State>();
                min_state.push_back(succ_state);
            }else if (h == min_h){
                min_state.push_back(succ_state);
            }
        }
        current_tree_node = &tree_search_space.get_node(min_state.at(rand() % min_state.size()));
    }         
        vector<OperatorID> successor_operators;
        successor_generator.generate_applicable_ops(
        current_tree_node->get_state(), successor_operators);
        for (OperatorID op_id : successor_operators) {
            OperatorProxy op = task_proxy.get_operators()[op_id];
            State succ_state = state_registry.get_successor_state(current_tree_node->get_state(), op);
            TreeSearchNode succ_node = tree_search_space.get_node(succ_state);
            StateID succ_id = succ_state.get_id();
            int succ_g = succ_node.get_g();
            succ_node.add_child(succ_id);
            EvaluationContext succ_eval_context(
            succ_state, succ_g, false, &statistics);//set to false or not?
            expand_tree(op_id ,succ_state, succ_eval_context);
            //Maybe add some new statistics here then
        }
}

void MonteCarloTreeSearch::expand_tree(OperatorID opid, State state, EvaluationContext eval_context){
   // Invariants:
    // - current_state is the next state for which we want to compute the heuristic.
    // - current_predecessor is a permanent pointer to the predecessor of that state.
    // - current_operator is the operator which leads to current_state from predecessor.
    // - current_g is the g value of the current state according to the cost_type
    // - current_real_g is the g value of the current state (using real costs)
    SearchNode node = search_space.get_node(state);
    TreeSearchNode tree_node = tree_search_space.get_node(state);
    StateID pred_id = tree_node.get_parent();
    SearchNode pred = search_space.get_node(state_registry.lookup_state(pred_id));
    int current_g =pred.get_g() + get_adjusted_cost(task_proxy.get_operators()[opid]);
    bool reopen = reopen_closed_nodes && !node.is_new() &&
        !node.is_dead_end() && (current_g < node.get_g()); // 0 was current_g (is now deleted)
    if (node.is_new() || reopen) {
        statistics.inc_evaluated_states();
        if (true) {
            // TODO: Generalize code for using multiple evaluators.
            if (pred_id == StateID::no_state) {
                node.open_initial();
                if (search_progress.check_progress(eval_context))
                    statistics.print_checkpoint_line(current_g);
            } else {
                State parent_state = state_registry.lookup_state(pred_id);
                SearchNode parent_node = search_space.get_node(parent_state);
                OperatorProxy current_operator = task_proxy.get_operators()[opid];
                if (reopen) {
                    node.reopen(parent_node, current_operator, get_adjusted_cost(current_operator));
                    statistics.inc_reopened();
                } else {
                    node.open(parent_node, current_operator, get_adjusted_cost(current_operator));
                }
            }
            node.close();
            if (check_goal_and_set_plan(state));

            if (search_progress.check_progress(eval_context)) {
                statistics.print_checkpoint_line(node.get_g());
                //reward_progress();
            }
            generate_successors(state, eval_context);
            statistics.inc_expanded();
        } else {//We need this for reopening
            node.mark_as_dead_end();
            statistics.inc_dead_ends();
        }
        if (pred_id == StateID::no_state) {
            print_initial_evaluator_values(eval_context, log);
        }
    }
}

void MonteCarloTreeSearch::backpropagation(){
    //update heuristic values backtracking to root
    //update best_h
}

bool SearchEngine::check_goal_and_set_plan(const State &state) {
    if (task_properties::is_goal_state(task_proxy, state)) {
        log << "Solution found!" << endl;
        Plan plan;
        search_space.trace_path(state, plan);
        set_plan(plan);
        return true;
    }
    return false;
}


void MonteCarloTreeSearch::reward_progress() {
    //open_list->boost_preferred();
}

void MonteCarloTreeSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
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
