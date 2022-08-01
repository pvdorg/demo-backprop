# demo-backprop

Example of neural network backpropagation

## history

This code was written in 1993 for demonstrating in a simplified way how neural network learning worked to non-specialists.

The code was meant to run on DOS and show some primitive graphic visualization of the learning state at a specific cycle.

## configuration

The neural net has one hidden layer.

The number of inputs, outputs and size of the hidden layer are specified with the following constants.

```C
#define MAXINPUT                2
#define MAXHIDDEN               100
#define MAXOUTPUT               1
```

It should be straightforward to add multiple layers to the code. This was out-of-scope for the simple demo.

## experiments

The learning capacity (brain power) of the NN increases with extra nodes in the hidden layer.

As an experiment (or demo) it is interesting to see what happens when there are only 2 hidden nodes versus 5, 10, 50, 100. Too few and the NN has difficulty learning to discriminate good vs bad samples. Too many and the NN will not generalize very well and learn the examples by heart.

This shows itself in the visual representation when you make a video out of it.

See this video (https://www.youtube.com/watch?v=DZkrMgoc69o) for an example using openFrameworks video capture. Black = 0, white = 1. The NN quickly settles for a first division of the solution space but then takes a long time to generalize the idea of the lower-right quadrant as an approximation for the lower-right white sample.

## source code

The folder OLDCODE contains the code as-is as I found it in my archives. I didn't include the exe files as these are 32-bit 1993 format.

The main folder contains different CPP files that each have a variant of number of node specified. I suppose this could be merged in one CPP file by making this dynamic by using a config setting.

*.DAT are training data files

*.ENV are saved configurations. This allows you to continue a learning cycle without having to start from zero again. In those days CPU was 1 core and Gigahertz was the equivalent of Star Trak warp speed.

### folder WNN

I think this was an attempt to make it run as a Windows GUI program.

### folder MFCNN

Not sure about this folder.

### folder of_v0.11.2_vs2017_release

This folder contains an openframeworks project for Windows Visual Studio 2022. Put the project in the apps folder of openframeworks.

The training file is hard-coded in the file testApp.cpp at line 21

```c
  const char * training_file = "C:\\PVDDATA\\of_v0.11.2_vs2017_release\\apps\\BP\\src\\XOR.DAT";
```

Change this so it points to a correct folder at your machine.

The NN config is in the file bp.h at lines 14-17.

```c
#define MAXINPUT		2
#define MAXHIDDEN		2
#define MAXOUTPUT		1
#define MAXTRAININGSAMPLES	500
```

Change the number of nodes in the hidden layer as convenient. The default 2 is a small brain. But it shows what the impact is of having to generalize with minimal variables. Playing with this value for a given training set e.g 2,4,8... it instructive.

There are some screenshots in the 'screenshots' folder for the XOR training with hidden 2,4 and 8 layers.
