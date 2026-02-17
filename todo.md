Graph of hypervolume of all points collected up until that iteration - Yes

Plotting of queried points in the objective space with pareto points - Yes

use qScalarizedUpperConfidenceBound https://archive.botorch.org/v/0.6.2/tutorials/custom_acquisition
3 experiments) - Yes
    Grid of beta values qUCB runs - Set up 
    LLM selecting beta values along with allocation options - Not set up
    LLM doing allocation selection with a fixed beta = 2 - Set up
    chebyschev + a z somewhere for scalarizing - Not yet




pareto-front plot axes = "Objective 1 (-|CTE|)", "Objective 2(K)"
summing uncertainty and graphing like hypervolume 
Change labelling on CI plots to qEVHI instead of pure exploitation and qUCB [beta = x]
decrease total seeds to 20
swap to gpt 5.1
have the agent give *mathmatical or statistical* reasoning for its decision.

Do an additional 5-10 seeds of all of the data vs just past 3 iterations for the agent. 
With confidence intervals on the agent-decisions plot.

## Future generalization for benchmarking
Simplex constraints need to be moved out
Double check agent manager 
Figure out best trevor problem

Experiment Run: 204995

Check out the unusual values in the agent's "best point" reporting stuff:
[Agent:INFO]
Analysis:
## Key Decision Criteria

**1. Runway Assessment**:
   - Approx. iterations remaining: ~18
   - Approx. affordable points: ~90
   - Limiting factor: budget

**2. Progress Momentum**:
   - Total points: 10
   - Best score: 9.045
   - Trend (last 3 vs previous 3): Accelerating (+5525.2%)

**3. Best Observed Point**:
   - CTE: -0.1707
   - K: 1.544
   - Coordinates: x_0=0.300, x_1=0.400, x_2=0.100, x_3=0.100, x_4=0.100