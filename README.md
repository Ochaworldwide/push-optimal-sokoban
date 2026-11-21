# Room-by-Room Push-Optimal Sokoban Solver

A Sokoban solver for make.fun's “Build the best Sokoban solver, 21 November 2026” bounty. It answers every level of a set in exactly the push-optimal count P* on the practice set (score 1,000,000 / 1,000,000 locally, against the house plain-astar baseline's 200,000).

How it works:

- **Rooms first.** Cutting the dead squares of the player's hallway row out of the floor splits the level into its rooms. A box can never leave its room in a solution, so the rooms are independent and their push optima add up to the level's.
- **A\* over pushes.** Each room is searched in pushes, not moves: a state is the boxes' squares plus the player's reachable region, and the heuristic is a minimum-cost box-to-goal matching over true push distances — admissible, so the first solution found is push-optimal.
- **Deadlock pruning.** Boxes pushed onto dead squares (no push can bring them to a goal) and frozen 2×2 blocks are cut before they waste the search.
- **Fallbacks.** A non-room level is solved whole with the same search, and a room that exhausts its budget is retried in breadth-first order.

Run it locally:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

MIT licensed.
