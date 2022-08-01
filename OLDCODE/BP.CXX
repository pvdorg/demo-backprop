// *****************************************************
//
// Backpropagation network with 1 hidden layer.
//
// *****************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FLOAT float

#define MAXINPUT		2
#define MAXHIDDEN		2
#define MAXOUTPUT		1
#define MAXTRAININGSAMPLES	100

int number_of_trainingsamples;

FLOAT ilayer[MAXINPUT];
FLOAT hlayer[MAXHIDDEN];
FLOAT olayer[MAXOUTPUT];

FLOAT w1[MAXINPUT][MAXHIDDEN];
FLOAT w2[MAXHIDDEN][MAXOUTPUT];

FLOAT delta_w1[MAXINPUT][MAXHIDDEN];
FLOAT delta_w2[MAXHIDDEN][MAXOUTPUT];

FLOAT hdelta[MAXHIDDEN];
FLOAT odelta[MAXOUTPUT];

FLOAT tinput[MAXTRAININGSAMPLES][MAXINPUT];
FLOAT toutput[MAXTRAININGSAMPLES][MAXOUTPUT];


FLOAT learning_rate = 0.3;
FLOAT momentum = 0.9;


// *****************************************************
// INITIALIZATION
// *****************************************************

void initialize( void )
{
  int i,j;

  for( i = 0; i < MAXINPUT; i++ )
    {
    ilayer[i] = 0.0;

    for( j = 0; j < MAXHIDDEN; j++ )
      {
      w1[i][j]       = 1.0;
      delta_w1[i][j] = 0.0;
      }
    }

  for( i = 0; i < MAXHIDDEN; i++ )
    {
    hlayer[i] = 0.0;

    for( j = 0; j < MAXOUTPUT; j++ )
      {
      w2[i][j]       = 1.0;
      delta_w2[i][j] = 0.0;
      }
    }

  for( i = 0; i < MAXOUTPUT; i++ )
    olayer[i] = 0.0;

}

// ***** Randomization of weights *****

void RandomizeWeights( void )
{
  int i,j;

  for( i = 0; i < MAXINPUT; i++ )
    for( j = 0; j < MAXHIDDEN; j++ )
      w1[i][j]       = ( (FLOAT) rand() ) / ( (FLOAT) RAND_MAX );

  for( i = 0; i < MAXHIDDEN; i++ )
    for( j = 0; j < MAXOUTPUT; j++ )
      w2[i][j]       = ( (FLOAT) rand() ) / ( (FLOAT) RAND_MAX );
}


// *****************************************************
// AUXILIARY FUNCTIONS
// *****************************************************

FLOAT Sigmoid( FLOAT x )
{
  return( 1 / (1+exp(-x)) );
}


void ShowWeights( void )
{
  int i,j;

  printf("(");

  for( j = 0; j < MAXHIDDEN; j++ )
    for( i = 0; i < MAXINPUT; i++ )
      printf(" %5.3f", w1[i][j]);

  printf(") - (");

  for( j = 0; j < MAXOUTPUT; j++ )
    for( i = 0; i < MAXHIDDEN; i++ )
      printf(" %5.3f", w2[i][j]);

  printf(")\n");
}

// *****************************************************
// TRAINING DATA
// *****************************************************


void LoadTrainingSet( FILE * fp )
{
  int i,j;

  if( fscanf(fp, " %d", &number_of_trainingsamples ) == EOF )
    {
    printf("Unexpexted EOF in scanf while reading data count.\n");
    exit(-1);
    }
  if( number_of_trainingsamples > MAXTRAININGSAMPLES )
    {
    number_of_trainingsamples = MAXTRAININGSAMPLES;
    printf("The program will only read %d training samples.\n");
    }

  for( i = 0; i < number_of_trainingsamples; i++ )
    {
    for( j = 0; j < MAXINPUT; j++ )
      {
      if( fscanf(fp, " %f", &tinput[i][j] ) == EOF )
        {
        printf("Unexpexted EOF in scanf while reading data for input layer.\n");
        exit(-1);
        }
      }
    for( j = 0; j < MAXOUTPUT; j++ )
      {
      if( fscanf(fp, " %f", &toutput[i][j] ) == EOF )
        {
        printf("Unexpexted EOF in scanf while reading data for output layer.\n");
        exit(-1);
        }
      }
    }
}


// *****************************************************
// ***** FORWARD PROPAGATION *****
// *****************************************************


void ForwardPropagation( int sample_index )
{
  int i,j;
  FLOAT sum;

  // set input layer

  for( i = 0; i < MAXINPUT; i++ )
    {
    ilayer[i] = tinput[sample_index][i];
    }

  // other layers

  for( j = 0; j < MAXHIDDEN; j++ )
    {
    sum = 0.0;
    for( i = 0;  i < MAXINPUT; i++ )
      {
      sum = sum + (w1[i][j]*ilayer[i]);
      }
    hlayer[j] = Sigmoid(sum);
    }

  for( j = 0; j < MAXOUTPUT; j++ )
    {
    sum = 0.0;
    for( i = 0;  i < MAXHIDDEN; i++ )
      {
      sum = sum + (w2[i][j]*hlayer[i]);
      }
    olayer[j] = Sigmoid(sum);
    }

}


// *****************************************************
// ***** BACKPROPAGATION *****
// *****************************************************


// ***** Calculate deltas and error *****

FLOAT CalculateDeltasAndError( int sample_index )
{
  int i,j;
  FLOAT out,error,total_error;

  total_error = 0.0;

  // output layer

  for( i = 0; i < MAXOUTPUT; i++ )
    {
    // compute delta
    out       = olayer[i];
    error     = toutput[sample_index][i] - out;
    odelta[i] = out * (1-out) * error;
    total_error = total_error + (error * error);
    }

  // hidden layer

  for( i = 0; i < MAXHIDDEN; i++ )
    {
    // compute delta
    out   = hlayer[i];
    error = 0.0;
    for( j = 0; j < MAXOUTPUT; j++ )
      {
      error = error + ( w2[i][j] * odelta[j] );
      }
    hdelta[i] = out * (1-out) * error;
    }

  // return total error
  return( total_error );

}

// ***** Update Weights *****

void UpdateWeights( void )
{
  int i,j;

  for( i = 0; i < MAXOUTPUT; i++ )
    {
    for( int j = 0; j < MAXHIDDEN; j++ )
      {
      delta_w2[j][i] = learning_rate * odelta[i] * hlayer[j];
      w2[j][i] += delta_w2[j][i];
      }
    }

  for( i = 0; i < MAXHIDDEN; i++ )
    {
    for( j = 0; j < MAXINPUT; j++ )
      {
      delta_w1[j][i] = learning_rate * hdelta[i] * ilayer[j];
      w1[j][i] += delta_w1[j][i];
      }
    }
}

// ***** Toplevel Back Propagation *****

FLOAT BackPropagation( int sample_index )
{
  FLOAT error;

  error = CalculateDeltasAndError( sample_index );
  UpdateWeights();
  return( error );
}


// *****************************************************
// ***** MAIN *****
// *****************************************************

int main( int argc, char * argv[] )
{

  FILE* fp;
  int i,j, count;
  FLOAT local_error, global_error;

  if ( argc != 3 )
    {
    printf("Usage: %s <training file> <count>\n", argv[0] );
    exit(-1);
    }

  fp = fopen( argv[1], "r" );
  if( !fp )
    {
    printf("Could not open training file.\n");
    exit(-1);
    }

  count = atoi(argv[2]);


  initialize();

  LoadTrainingSet(fp);

  RandomizeWeights();

  for( j = 0; j < count; j++ )
    {
    global_error = 0.0;
    for(i = 0; i < number_of_trainingsamples; i++)
      {
      ForwardPropagation(i);
      local_error = BackPropagation(i);
      global_error += local_error;

      printf("\ntest (%5.3f %5.3f) -> %5.3f -> output %5.3f -> error %5.3f"
             " (%5.3f)\n",
              ilayer[0], ilayer[1], toutput[i][0], olayer[0],
              local_error, global_error
              );      
      ShowWeights();
      }
    }

  fclose(fp);
}
