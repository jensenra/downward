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
#include <queue>

#include <memory>
#include <vector>

namespace options {
class Options;
}

namespace monte_carlo_tree_search {
class MonteCarloTreeSearch : public SearchEngine {
protected:
    // Search behavior parameters
    double epsilon;
    bool reopen_closed_nodes; // whether to reopen closed nodes upon finding lower g paths
    bool randomize_successors;
    bool preferred_successors_first;
    std::shared_ptr<utils::RandomNumberGenerator> rng;

    std::shared_ptr<Evaluator> heuristic;

    TreeSearchSpace tree_search_space;
    bool check_goal_and_set_plan(const State &state);
public:

    virtual void initialize() override;
    virtual SearchStatus step() override;
    State select_next_leaf_node(const State state);
    SearchStatus expand_tree(const State state);
    void back_propagate(State state);
    void back_propagate_dead_end(State state);
    void re_back_propagate_best_h(State state);
    void back_propagate_best_h(State state);
    void forward_propagate_g(State state, int g_diff);

    std::vector<OperatorID> get_successor_operators(
        const ordered_set::OrderedSet<OperatorID> &preferred_operators) const;

    explicit MonteCarloTreeSearch(const options::Options &opts);
    virtual ~MonteCarloTreeSearch() = default;

    virtual void print_statistics() const override;
};
}

#endif
