"""Task-aware actuator co-design on top of the mjlab Matlas stack.

actorob (https://mkakanov.github.io/actorob/) co-designs actuators by searching
motor/gearbox specs against a task. Here the inner loop is RL: a single
design-conditioned policy is trained over a distribution of actuator designs
(per parallel env), then a multi-objective search (NSGA-II) finds the
mass / performance / energy Pareto front against that frozen policy.
"""
