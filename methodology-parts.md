# This is a note for us to begin framing out the methodology for the paper, specifically referencing the steps we take in this project for batched runs

### Problem Selection
    #### Design Space

### Ground Truth Building
    

### Strategy Definitions
    We want to test out agents, so we're going to demonstrate 3 methods, pure exploration, pure exploitation, and agents.

    #### Acquisition Functions

    #### Metrics

    #### Agents Implementation

    #### Different Seeds for generalization

    ##### How do we monitor optimization progress? 

    #### Events - Budget and Time


    % How do we configure the BO loop? 
        The main goal of this paper is to prepare a framework that works in general across whatever sort of BO loop you want to put together. We just want to enable using different acquisition functions or combinations of acquisition functions controlled by a "decision-maker" component. To that end, the project was developed using standard BOTorch tools with minimal customization. This way, it should be reasonably easy to tailor the problem however one needs for their situation.
    % There's the classic "BO" stuff
        % Multi-Task GP
        % Design Space
            Limited to 2D to 10D design space.
        % Truth Function
            Random Forest Regressor with Hyperparameter tuning and 
        % - Discrete Optimization by default
        
    % There's the custom stuff we do
        % Dual Acquisition Functions
            % qEHVI
            % qUCB
                % StandardScalarizer
                % b = 2.0
        % Acquisition function "Metrics"
            % EHVI
            % Mutual Information
        % Allocation Options
        % Strategies
            % Optimization
            % Exploration
            % LLM Agent
                % Prompting

% Given the BO Loop, which problems did we do? Why, and How?
    % FeCoCrNiV - K vs CTE
        % What is this material? What is the design space? Where is it used?
        % Why did we use this problem / material?
        % How did we adjust around the problem? Did we make the design space more complex?
    % TiVNbMoHfTaW BCC - Density vs Melting Point
        % What is this material? What is the design space? Where is it used?
        % Why did we use this problem / material?
        % How did we adjust around the problem? Did we make the design space more complex?

% Everything else specific to this work goes in results / discussion.

Future Direction : Numbers for confidence -> Do like a 