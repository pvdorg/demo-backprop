//
// BPMB2-NN.H
//

#ifndef BPMB2_NN_H
#define BPMB2_NN_H

// ********************
// * global constants *
// ********************

#define FLOAT float

#define MAXINPUT                2
#define MAXHIDDEN               2
#define MAXOUTPUT               1
#define MAXTRAININGSAMPLES      100

// global variables

extern FLOAT learning_rate;
extern FLOAT momentum;

extern int cycle_count;

extern int number_of_trainingsamples;
extern char current_trainingfile[];

// layers

extern FLOAT ilayer[MAXINPUT];
extern FLOAT hlayer[MAXHIDDEN];
extern FLOAT olayer[MAXOUTPUT];

// weights

extern FLOAT w1[MAXINPUT][MAXHIDDEN];
extern FLOAT w2[MAXHIDDEN][MAXOUTPUT];

extern FLOAT bias1[MAXHIDDEN]; 
extern FLOAT bias2[MAXOUTPUT];

// weight changes

extern FLOAT delta_w1[MAXINPUT][MAXHIDDEN];
extern FLOAT delta_w2[MAXHIDDEN][MAXOUTPUT];

extern FLOAT delta_bias1[MAXHIDDEN]; 
extern FLOAT delta_bias2[MAXOUTPUT];

// deltas

extern FLOAT hdelta[MAXHIDDEN];
extern FLOAT odelta[MAXOUTPUT];

// training set

extern FLOAT tinput[MAXTRAININGSAMPLES][MAXINPUT];
extern FLOAT toutput[MAXTRAININGSAMPLES][MAXOUTPUT];

// *******************************
// ***** Function Prototypes *****
// *******************************

// Initialization

void InitializeNetwork( void );
void RandomizeWeights( void );

// Training Data

void LoadTrainingSet( const char * filename );

// Cycle

void Cycle( int count, int printit );

#endif

