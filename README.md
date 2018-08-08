# CUDA-CellularAutomata

Generate game maps for top-down games using Cellular Automata (GPU) and Flood fill.

In practice, Cellular Automata using GPU to generate map is not really necessary IF the game worlds dimension is relatively small let's say 126x126 or even 512x512, the differences between CPU and GPU is not much, CPU still can handle it. Unless you want to process million(s) of cells, then GPU is coming to save the world.

How to:
1. Build kernel.cu as .dll -> need to install CUDA stuff
2. Just use dat .dll
