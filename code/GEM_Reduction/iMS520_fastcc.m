% clc; clearvars;

% Load model
model = readCbModel('iMS520.mat');

% Run FASTCC with a flux threshold of 1e-4
consistentRxnIndices = fastcc(model, 1e-4);  % returns indices

% Get the reaction IDs of consistent reactions
consistentRxns = model.rxns(consistentRxnIndices);

% Build the consistent submodel
consistentModel = extractSubNetwork(model, consistentRxns);

% Number of reactions in the original model
numOriginalRxns = length(model.rxns);

% Number of reactions in the consistent model
numConsistentRxns = length(consistentModel.rxns);

% Number of removed (blocked) reactions
numRemoved = numOriginalRxns - numConsistentRxns;

% Find and display removed reactions
removedRxns = setdiff(model.rxns, consistentRxns);

% Display results
fprintf('Original model: %d reactions\n', numOriginalRxns);
fprintf('Consistent model: %d reactions\n', numConsistentRxns);
fprintf('Removed (blocked) reactions: %d\n\n', numRemoved);

% Optional: Save removed reactions to file
writetable(cell2table(removedRxns), 'iMS520_removed_reactions.txt', 'WriteVariableNames', false);

% Save reduced model
save('iMS520_fastcc_red.mat', 'consistentModel');

%--------------------------------------------------------------------------


