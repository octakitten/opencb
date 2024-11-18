# silky
A framework for cognitive neural networks with persistent memory

What is this?

This is a hobby project aimed at creating a neural network (of sorts) which can learn and act in real time. I don't really have the methodology documented well, yet, but I'm working on it. Currently, the model sees a pixel greyscale image of arbitrary size and reacts to it with predefined actions. These actions allow it to play a series of games which have win and loss conditions. The model can be initialized with randomized parameters, or these parameters can be iterated on each time the model wins a game.

Currently there are only a handful of simplistic games and the most powerful model runs ridiculously slowly on a home pc. Things on the roadmap are: creating a minesweeper-like game, integrating pytensorplus into the project, and creating a more optimized model.

If you'd like to playtest this weird little project, I'm including some wheel releases of the library which can be downloaded from the Releases page and installed locally to your virtual python environment with pip. Fair warning, the project doesn't solve any meaningful problems yet. Thus far I've found that the latest model can "win" the Forest game in anywhere from a few hundred to a few thousand iterations. If you have a GPU with Cuda version 12 or higher (nvidia 10-series or later), then you'll be able to use it to test the model on your home pc in a reasonable amount of time. Just write a small python file that imports the package and runs the "run_general_dev()" function from the iteration module and off it will go!
