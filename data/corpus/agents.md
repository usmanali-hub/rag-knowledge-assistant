# LLM agents and tool use

An agent wraps a language model in a loop. The model reads the goal, decides whether to call a tool, receives the tool result and repeats until it can answer. Tools are ordinary functions described to the model with a name, a description and a JSON schema for the arguments. Good agents keep the loop bounded with a maximum number of steps, validate every tool argument before running it and log each step so a failed run can be replayed. Giving an agent fewer, sharper tools usually works better than giving it dozens of overlapping ones.
