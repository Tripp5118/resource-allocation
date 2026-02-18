### One Run to Test Out Agents Reasoning

Plot of budgets decreasing (probably with convergence stats?)

# Big run tonight
decrease total seeds to 20
Do an additional 5-10 seeds of all of the data vs just past 3 iterations for the agent. 
With confidence intervals on the agent-decisions plot.

do 95th percentile because it's probably not normally distributed

Mutual information using scikit-learn

## Future generalization for benchmarking
Simplex constraints need to be moved out to problem definition (this should be true now that we're passing design space)
Double check agent manager 
Figure out best trevor problem

### Eventuallly a run where the agent can control beta.


# Eventual Restructuring to make easier to use
   - The point of this work is for autonomously doing decision-making in a BO loop in order to balance:
      - Optimization
      - Exploration
      - Time
      - Budget
   BO in general has:
      - Gaussian Process
      - Acquisition Function
      - Truth Function
   
   We do "resource allocation" by using an LLM agent informed by:
      - Optimization Progress and Statistics
      - Batchwise "options" of points we could evaluate generated from a mix of acquisition functions
         We evaluate each "option" using metrics pertaining to optimization and exploration
            - Hypervolume
            - Mutual Information
   
   So what's the way this project should be structured to line up with all of these aspects?

   There's 4 things to set up for arbitrary problems:

      These are the "BO" parameters that our work has to account for and that we sometimes have to make compromises with
      - Problem Definition
         - Design Space - Up to 10D? Arbitrarily? I think it's limited somewhere.
         - Truth Function - Discrete or Continuous
      - Gaussian Process Setup
         - Kernel - No constraints on our end
         - Multi-Task vs Single-Task - Just Multi-Task for now. Single Task conflicts with visualization and acquisition functions. Also possibly metrics calculation.
         - etc...
         - Acquisition Functions - Batched and 

      These are the "Resource Allocation" parameters - AKA the things that we handle specially for our implementation / work
      - Optimization Parameters
         - Number of Iterations
         - Batch Size
         - Time Constraint
         - Budget Constraint
         - Events
         - (eventually) Output Constraints
      - LLM Agent Configuration
         - LLM
         - Optimization Statistics
         - Allocation Option Metrics
         - (eventually) Tools











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