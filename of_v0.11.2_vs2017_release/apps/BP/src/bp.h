//
//  bp.h
//
//  Created by Peter Van Damme on 04-October-2012.
//  Updated by Peter Van Damme on 1-August-2022.
//  Copyright (c) 2012-2022 Peter Van Damme. All rights reserved.
//

#ifndef bp_h
#define bp_h

#define FLOAT float

#define MAXINPUT		2
#define MAXHIDDEN		2
#define MAXOUTPUT		1
#define MAXTRAININGSAMPLES	500

extern int number_of_trainingsamples;

// layers

extern FLOAT ilayer[MAXINPUT];
extern FLOAT hlayer[MAXHIDDEN];
extern FLOAT olayer[MAXOUTPUT];

// weights

extern FLOAT w1[MAXINPUT][MAXHIDDEN];
extern FLOAT w2[MAXHIDDEN][MAXOUTPUT];

extern FLOAT delta_w1[MAXINPUT][MAXHIDDEN];
extern FLOAT delta_w2[MAXHIDDEN][MAXOUTPUT];

// deltas

extern FLOAT hdelta[MAXHIDDEN];
extern FLOAT odelta[MAXOUTPUT];

// training data

extern FLOAT tinput[MAXTRAININGSAMPLES][MAXINPUT];
extern FLOAT toutput[MAXTRAININGSAMPLES][MAXOUTPUT];

#define DEFAULT_LEARNING_RATE 0.3
#define DEFAULT_MOMENTUM      0.9

extern FLOAT learning_rate;
extern FLOAT momentum;

// API

void initialize( void );
void RandomizeWeights( void );

FLOAT Sigmoid( FLOAT x );
void ShowWeights( void );

int LoadTrainingSet( FILE * fp );

FLOAT CalculateOutput( FLOAT x, FLOAT y );

void Cycle( int count );

FLOAT GetSampleInput( int sample_index, int input_index );
FLOAT GetSampleOutput( int sample_index, int input_index );

int GetNumberOfTrainingSamples( void );
void AddTrainingSample( FLOAT x, FLOAT y, FLOAT val );

#endif
