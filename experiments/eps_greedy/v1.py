#! /usr/bin/env python

import itertools
import os

from lab.environments import LocalEnvironment, BaselSlurmEnvironment

from downward.reports.compare import ComparativeReport

import common_setup
from common_setup import IssueConfig, IssueExperiment

DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
BENCHMARKS_DIR = os.environ["DOWNWARD_BENCHMARKS"]
REVISIONS = ["c9ae3c9645e77d95e6532e97a6226c5afea4e0e8"]
CONFIGS = [
    IssueConfig("eps-greedy-tree", ["--search", "mcts(ff())"]),
    #IssueConfig("eps-greedy-list", ["--search", "eager(epsilon_greedy(ff(),epsilon=0.0001),reopen_closed=true)"]),
]

SUITE = common_setup.DEFAULT_OPTIMAL_SUITE
ENVIRONMENT = BaselSlurmEnvironment(
    partition="infai_2",
    email="r.jensen@stud.unibas.ch",
    export=["PATH", "DOWNWARD_BENCHMARKS"])

#if common_setup.is_test_run():
SUITE = IssueExperiment.DEFAULT_TEST_SUITE
ENVIRONMENT = LocalEnvironment(processes=2)

exp = IssueExperiment(
    revisions=REVISIONS,
    configs=CONFIGS,
    environment=ENVIRONMENT,
)
exp.add_suite(BENCHMARKS_DIR, SUITE)

exp.add_parser(exp.EXITCODE_PARSER)
exp.add_parser(exp.SINGLE_SEARCH_PARSER)
exp.add_parser(exp.PLANNER_PARSER)

exp.add_step('build', exp.build)
exp.add_step('start', exp.start_runs)
exp.add_fetcher(name='fetch')

exp.add_absolute_report_step()
#exp.add_comparison_table_step()
#exp.add_scatter_plot_step(relative=True, attributes=["total_time", "memory"])

exp.run_steps()
