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
    heuristic(opts.get<shared_ptr<Evaluator>>("heuristic")),
      current_state(state_registry.get_initial_state()),
      current_predecessor_id(StateID::no_state),
      current_operator_id(OperatorID::no_operator),
      current_g(0),
      current_real_g(0),
      current_eval_context(current_state, 0, true, &statistics),
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

void MonteCarloTreeSearch::generate_successors() {
    vector<OperatorID> successor_operators ;
    successor_generator.generate_applicable_ops(
        current_state, successor_operators);

    statistics.inc_generated(successor_operators.size());

    for (OperatorID op_id : successor_operators) {
        OperatorProxy op = task_proxy.get_operators()[op_id];
        int new_g = current_g + get_adjusted_cost(op);
        int new_real_g = current_real_g + op.get_cost();
        if (new_real_g < bound) {
            EvaluationContext new_eval_context(
                current_eval_context, new_g, false, nullptr);
        }
    }
}

void MonteCarloTreeSearch::trial() {

}

void MonteCarloTreeSearch::select_next_leaf_node(){
    TreeSearchNode current_tree_node = tree_search_space.get_node(state_registry.get_initial_state());
    while(true){
        vector<StateID> children = current_tree_node.get_children();
        vector<OperatorID> successor_operators;
        successor_generator.generate_applicable_ops(
        current_state, successor_operators);
        //Are all successors in the search tree?
        if(children.size() == successor_operators.size()){
            //TODO: Find the node for which the heuristic value ranks the highest
            //TODO: Add epsilon exploration
            int min_h = INT_MAX;
            for(auto iterator = children.cbegin(); iterator != children.cend(); iterator++){
                State child_state = state_registry.lookup_state(*iterator);
                //get heuristic value
            }
        } else break;
    }
}
void MonteCarloTreeSearch::expand_tree(){
   // Invariants:
    // - current_state is the next state for which we want to compute the heuristic.
    // - current_predecessor is a permanent pointer to the predecessor of that state.
    // - current_operator is the operator which leads to current_state from predecessor.
    // - current_g is the g value of the current state according to the cost_type
    // - current_real_g is the g value of the current state (using real costs)


    TreeSearchNode node = tree_search_space.get_node(current_state);
    bool reopen = reopen_closed_nodes && !node.is_new() &&
        !node.is_dead_end() && (current_g < node.get_g());

    if (node.is_new() || reopen) {
        if (current_operator_id != OperatorID::no_operator) {
            assert(current_predecessor_id != StateID::no_state);
            if (!path_dependent_evaluators.empty()) {
                State parent_state = state_registry.lookup_state(current_predecessor_id);
                for (Evaluator *evaluator : path_dependent_evaluators)
                    evaluator->notify_state_transition(
                        parent_state, current_operator_id, current_state);
            }
        }
        statistics.inc_evaluated_states();
        //!open_list->is_dead_end(current_eval_context)
        //Check if dead end
        if (true) {
            // TODO: Generalize code for using multiple evaluators.
            if (current_predecessor_id == StateID::no_state) {
                node.open_initial();
                if (search_progress.check_progress(current_eval_context))
                    statistics.print_checkpoint_line(current_g);
            } else {
                State parent_state = state_registry.lookup_state(current_predecessor_id);
                TreeSearchNode parent_node = tree_search_space.get_node(parent_state);
                OperatorProxy current_operator = task_proxy.get_operators()[current_operator_id];
                if (reopen) {
                    node.reopen(parent_node, current_operator, get_adjusted_cost(current_operator));
                    statistics.inc_reopened();
                } else {
                    node.open(parent_node, current_operator, get_adjusted_cost(current_operator));
                }
                StateID curr = current_state.get_id();
                parent_node.add_child(curr);
            }
            node.close();
            if (check_goal_and_set_plan(current_state))
                return SOLVED;
            if (search_progress.check_progress(current_eval_context)) {
                statistics.print_checkpoint_line(current_g);
                reward_progress();
            }
            generate_successors();
            statistics.inc_expanded();
        } else {
            node.mark_as_dead_end();
            statistics.inc_dead_ends();
        }
        if (current_predecessor_id == StateID::no_state) {
            print_initial_evaluator_values(current_eval_context, log);
        }
    }
}
void MonteCarloTreeSearch::simulate(){
    //take random actions until a terminal node is reached
}
void MonteCarloTreeSearch::backpropagation(){
    //update heuristic values backtracking to root
}

bool SearchEngine::check_goal_and_set_plan(const State &state) {
    if (task_properties::is_goal_state(task_proxy, state)) {
        log << "Solution found!" << endl;
        Plan plan;
        tree_search_space.trace_path(state, plan);
        set_plan(plan);
        return true;
    }
    return false;
}




SearchStatus MonteCarloTreeSearch::step() {
    // Invariants:
    // - current_state is the next state for which we want to compute the heuristic.
    // - current_predecessor is a permanent pointer to the predecessor of that state.
    // - current_operator is the operator which leads to current_state from predecessor.
    // - current_g is the g value of the current state according to the cost_type
    // - current_real_g is the g value of the current state (using real costs)


    SearchNode node = search_space.get_node(current_state);
    bool reopen = reopen_closed_nodes && !node.is_new() &&
        !node.is_dead_end() && (current_g < node.get_g());

    if (node.is_new() || reopen) {
        if (current_operator_id != OperatorID::no_operator) {
            assert(current_predecessor_id != StateID::no_state);
            if (!path_dependent_evaluators.empty()) {
                State parent_state = state_registry.lookup_state(current_predecessor_id);
                for (Evaluator *evaluator : path_dependent_evaluators)
                    evaluator->notify_state_transition(
                        parent_state, current_operator_id, current_state);
            }
        }
        statistics.inc_evaluated_states();
        //!open_list->is_dead_end(current_eval_context)
        if (true) {
            // TODO: Generalize code for using multiple evaluators.
            if (current_predecessor_id == StateID::no_state) {
                node.open_initial();
                if (search_progress.check_progress(current_eval_context))
                    statistics.print_checkpoint_line(current_g);
            } else {
                State parent_state = state_registry.lookup_state(current_predecessor_id);
                SearchNode parent_node = search_space.get_node(parent_state);
                OperatorProxy current_operator = task_proxy.get_operators()[current_operator_id];
                if (reopen) {
                    node.reopen(parent_node, current_operator, get_adjusted_cost(current_operator));
                    statistics.inc_reopened();
                } else {
                    node.open(parent_node, current_operator, get_adjusted_cost(current_operator));
                }
            }
            node.close();
            if (check_goal_and_set_plan(current_state))
                return SOLVED;
            if (search_progress.check_progress(current_eval_context)) {
                statistics.print_checkpoint_line(current_g);
                reward_progress();
            }
            generate_successors();
            statistics.inc_expanded();
        } else {
            node.mark_as_dead_end();
            statistics.inc_dead_ends();
        }
        if (current_predecessor_id == StateID::no_state) {
            print_initial_evaluator_values(current_eval_context, log);
        }
    }
    return fetch_next_state();
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
