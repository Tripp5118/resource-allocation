1 Version - No guidelines, just let the agent act on its own.
1 Version - General guidelines
1 Version - Old guidelines

Add in hypervolume and uncertainty information
Add in a version that explicitly is told to look at its prior decision and judge whether it was successful or not. 
   Maybe 3 Agents? Planner, Resource Allocator, Critic? Planner and Critic only act every 2 or 3 iterations?
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
