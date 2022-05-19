#ifndef SEARCH_ENGINES_MONTE_CARLO_TREE_SEARCH_H
#define SEARCH_ENGINES_MONTE_CARLO_TREE_SEARCH_H

#include "../evaluation_context.h"
#include "../evaluator.h"
#include "../open_list.h"
#include "../operator_id.h"
#include "../search_engine.h"
#include "../search_progress.h"
#include "../search_space.h"
#include "../treesearch_space.h"

#include "../utils/rng.h"
#include <priority_queues.h>

#include <memory>
#include <vector>

namespace options {
class Options;
}

namespace monte_carlo_tree_search {
class MonteCarloTreeSearch : public SearchEngine {
protected:
    // Search behavior parameters
    bool reopen_closed_nodes; // whether to reopen closed nodes upon finding lower g paths
    bool randomize_successors;
    bool preferred_successors_first;
    std::shared_ptr<utils::RandomNumberGenerator> rng;

    std::shared_ptr<Evaluator> heuristic;
    
    std::priority_queue<TreeSearchNode, vector<TreeSearchNode>, std::greater<TreeSearchNode>> q;
    TreeSearchSpace tree_search_space;
    bool check_goal_and_set_plan(const State &state);
public:

    virtual void initialize() override;
    virtual SearchStatus step() override;
    State select_next_leaf_node(const State state);
    SearchStatus expand_tree(const State state);
    //void simulate();
    void back_propagate(State state);
    void reopen_h(TreeSearchNode node, TreeSearchNode succ_node);
    void update_best_h(State state);
    void recursive_prio_queue_add(&priority_queue<TreeSearchNode> q, TreeSearchNode node);
    void reopen_g(State state, int g_diff, boolean first);

    void generate_successors(State state, EvaluationContext eval_context);
    SearchStatus fetch_next_state();

    void reward_progress();
    void move_to_state(State state, EvaluationContext eval_context, StateID pred, OperatorID op_id);

    std::vector<OperatorID> get_successor_operators(
        const ordered_set::OrderedSet<OperatorID> &preferred_operators) const;

    explicit MonteCarloTreeSearch(const options::Options &opts);
    virtual ~MonteCarloTreeSearch() = default;

    void set_preferred_operator_evaluators(std::vector<std::shared_ptr<Evaluator>> &evaluators);

    virtual void print_statistics() const override;
};
}

#endif
