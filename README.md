

# Completions Layer
The first module that we need to build out is the completions layer. It is a primitive that should allow an agent (consider in this case the agent as the file with the dedalus agent, so a python file) to create multiple conversations with other dedalus agents. The arguments are probably the original agent as well as the prompt for the conversational agents . The output is a list of dictionaries each with a single conversation that the base agent had with the second agents. This will be the building block for being passed along to the judging layer so be mindful of making it scaleable. 

# Judging Layer 
Now we will implement a judging layer. The judging layer will take as its input a dictionary with a set of judging criteria. So imagine that it will ingest like, 'Offer payment agreement' and this would be like hjudging judging criteria number #1, etc. Then. the judging layer will take as inputs the set of conversations that the completions layer generates, as per what we defined before. It will also take as its input a prompt for a judging LLM. Then, what we need is to give the llm with the judging prompt the list of conversations and ask it if the different judging criteria have been met. What we want to get out of it is a simple bool scoring of each of the judging criteria provided at the beguinning. So to recap, this layer is going to take in the list of conversations provided by the completions layer as well as a list of goals, and we're asking an evaluator agent with an evaluation prompt to give a boolean output for each one of the evaluations criterias. Again keep it modular, keep it as a fundamental bulding block

# Simulation Layer
Now we're going to go ahead and create the simulation layer. This is the overview. It's going to ingest a list from the judging layer on the boolean value of each of the goals. If all of them are true it should return something simple like 'Everything works!', for the ones that are not true, for each, we will instantiate and instance of a 'fixer' agent for each. It's going to be given: 
1. The python file for the agent( meaning the Dedalus file)
2. The original goal that was meant to be accomplished but didn't pass through. 
3. The completion that that the judging layer determined to not be good (in this case that'd mean the conversation that was marked as false.)
4. [part of the base prompt] A list of all the MCP's that Dedalus has with instructions for each one so that if wants to add an MCP because it think it would be useful then it knows which are available. 
5. The instructions for how to add tools, based on the Dedalus documentation. So, how to add them in python and then include them as part of the dedalus agent. 
Each agent must work async (meaning we need N fixers to run in parallale for however many issues). Their output should be a diff of the original file with whatever they think is a worthwhile change that will be in either the new prompt or on new tools they defined and added or mcps they defined and added. What is really important is that they each have clear instructions on what to add and how to add it. 
So remember, the output is a new changed file after they saw what was wrong and changed something in the agent they believed would fix it. 

# Loop layer #1
Now build out the first part of the improvement loop. The first part should be simple enough. It is just that for each new file made in the previous simulation layer it should re-run the conversations layer, re-judge it using the judging layer, and then depending on the output of each one of the criteria either go back to the simulation layer for further improvement 

