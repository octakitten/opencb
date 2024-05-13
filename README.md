# opencb
Open-Source Computer Behavior

What is this?

This is a hobby project aimed at creating a neural network (of sorts) which can learn and act in real time. I don't really have the methodology documented well, yet, but I'm working on it. Currently, the model sees a pixel greyscale image of arbitrary size and reacts to it with predefined actions. These actions allow it to play a series of games which have win and loss conditions. The model can be initialized with randomized parameters, or these parameters can be iterated on each time the model wins a game.

Currently there are only a handful of simplistic games and the most powerful model runs ridiculously slowly on a home pc. Things on the roadmap are: creating a minesweeper-like game and creating a more optimized model.

If you'd like to playtest this weird little project, I'm including some wheel releases of the library which can be downloaded from the Releases page and installed locally to your python environment with pip. Fair warning, the project has not yet achieved any measurable results. I haven't gotten a model to win a game more than once so far. But, if you'd like to run a game idly for some reason, maybe to see if it wins a game, feel free to look at the testing.py script in the Tests folder of the main repo. You can run the most recent game (as of 1/30/24) with the test009() function in that file.