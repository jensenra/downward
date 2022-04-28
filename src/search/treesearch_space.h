#ifndef TREESEARCH_SPACE_H
#define TREESEARCH_SPACE_H

#include "operator_cost.h"
#include "per_state_information.h"
#include "treesearch_node_info.h"
#include "search_node_info.h"
#include "search_space.h"

#include <vector>

class OperatorProxy;
class State;
class TaskProxy;

namespace utils {
class LogProxy;
}

class TreeSearchNode{
    State state;
    TreeSearchNodeInfo &info;
public:
    TreeSearchNode(const State &state, TreeSearchNodeInfo &info);
    const State &get_state() const;

    bool is_new() const;
    bool is_open() const;
    bool is_closed() const;
    bool is_dead_end() const;

    int get_g() const;
    int get_real_g() const;
    vector<StateID> get_children();
    void add_child(StateID &child);
    StateID get_parent();
    OperatorID get_operator();

    void open_initial();
    int get_best_h();
    void set_best_h(int new_best_h);
    void open(const TreeSearchNode &parent_node,
              const OperatorProxy &parent_op,
              int adjusted_cost);
    void reopen(const TreeSearchNode &parent_node,
                const OperatorProxy &parent_op,
                int adjusted_cost);
    void update_parent(const TreeSearchNode &parent_node,
                       const OperatorProxy &parent_op,
                       int adjusted_cost);
    void close();
    void mark_as_dead_end();
    
    void dump(const TaskProxy &task_proxy, utils::LogProxy &log) const;
};


class TreeSearchSpace{
    PerStateInformation<TreeSearchNodeInfo> search_node_infos;
    StateRegistry &state_registry;
    utils::LogProxy &log;
public:
    TreeSearchSpace(StateRegistry &state_registry, utils::LogProxy &log);

    TreeSearchNode get_node(const State &state);
    void trace_path(const State &goal_state,
                    std::vector<OperatorID> &path) const;

    void dump(const TaskProxy &task_proxy) const;
    void print_statistics() const;
};

#endif
