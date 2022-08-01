// *****************************************************
//
// Backpropagation network with 1 hidden layer.
//
// - with learning rate and momentum
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

// layers

FLOAT ilayer[MAXINPUT];
FLOAT hlayer[MAXHIDDEN];
FLOAT olayer[MAXOUTPUT];

// weights

FLOAT w1[MAXINPUT][MAXHIDDEN];
FLOAT w2[MAXHIDDEN][MAXOUTPUT];

FLOAT delta_w1[MAXINPUT][MAXHIDDEN];
FLOAT delta_w2[MAXHIDDEN][MAXOUTPUT];

// deltas

FLOAT hdelta[MAXHIDDEN];
FLOAT odelta[MAXOUTPUT];

// training data

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
      w1[i][j] = ( (FLOAT) rand() ) / ( (FLOAT) RAND_MAX );

  for( i = 0; i < MAXHIDDEN; i++ )
    for( j = 0; j < MAXOUTPUT; j++ )
      w2[i][j] = ( (FLOAT) rand() ) / ( (FLOAT) RAND_MAX );

}


// *****************************************************
// AUXILIARY FUNCTIONS
// *****************************************************

FLOAT Sigmoid( FLOAT x )
{
  return( 1 / (1+exp(-x)) );
}

// ***** Display Weights *****

void ShowWeights( void )
{
  int i,j;

  printf("(");

  for( j = 0; j < MAXHIDDEN; j++ )
    {
    for( i = 0; i < MAXINPUT; i++ )
      printf(" %5.3f", w1[i][j]);
    }

  printf(") - (");

  for( j = 0; j < MAXOUTPUT; j++ )
    {
    for( i = 0; i < MAXHIDDEN; i++ )
      printf(" %5.3f", w2[i][j]);
    }

  printf(")\n");
}

// *****************************************************
// ***** TRAINING DATA *****
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

// ***** Update Weights (use momentum term) *****

void UpdateWeights( void )
{
  int i,j;

  for( i = 0; i < MAXOUTPUT; i++ )
    {
    for( int j = 0; j < MAXHIDDEN; j++ )
      {
      // calculate new delta
      delta_w2[j][i] = ( learning_rate * odelta[i] * hlayer[j] )
                       +   // add momentum term
                       ( momentum * delta_w2[j][i] );
      
      // update weight
      w2[j][i] += delta_w2[j][i];
      }
    }

  for( i = 0; i < MAXHIDDEN; i++ )
    {
    for( j = 0; j < MAXINPUT; j++ )
      {
      // calculate new delta
      delta_w1[j][i] = ( learning_rate * hdelta[i] * ilayer[j] )
                       +   // add momentum term
                       ( momentum * delta_w1[j][i] );
      
      // update weight
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
// ***** ONE CYCLE *****
// *****************************************************

void Cycle( int count )
{
  int i,j;
  FLOAT local_error, global_error;

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
}

// *****************************************************
// ***** TOPLEVEL COMMANDS *****
// *****************************************************

// ***** cycle *****

void TopCycle( void )
{
  int count;

  if( scanf(" %d", &count ) == EOF )
    {
    printf("Unexpexted EOF in scanf while reading count.\n");
    return;
    }

  Cycle(count);
}

// ***** Step *****

void TopStep( void )
{
  Cycle(1);
}

// ***** Learning Rate *****

void TopLearningRate( void )
{
}

// ***** Momentum *****

void TopMomentum( void )
{
}

// *****************************************************
// ***** MAIN *****
// *****************************************************

// ***** topmenu *****

void ShowMenu( void )
{
  printf("(R)andomize\n");
  printf("(S)tep\n");
  printf("(C)ycle\n");
  printf("\n");
  printf("(L)earning rate\n");
  printf("(M)omentum\n");
  printf("\n");
  printf("(Q)uit\n");
  printf("\n");
}

// ***** get command *****

int GetCommand( void )
{
  int c;

  for(;;)
    {
    printf("command: ");
    c = getchar();
    if( c != '\n' )
      return( toupper(c) );
    }
}

// ***** command loop *****

void CommandLoop( void )
{
  int c, flag;

  flag = 1;

  for(;;)
    {
    if ( flag ) ShowMenu();

    switch ( c = GetCommand() )
      {
      case '\n':                     flag = 0; break; // ignore newline

      case 'R' : RandomizeWeights(); flag = 1; break;

      case 'C' : TopCycle();         flag = 1; break;

      case 'S' : TopStep();          flag = 1; break;

      case 'L' : TopLearningRate();  flag = 1; break;

      case 'M' : TopMomentum();      flag = 1; break;

      case 'Q' : return;             break;

      default  : printf("%c is not a valid command\n",c); flag = 0; break;
      }
    }
}

// ***** main *****

int main( int argc, char * argv[] )
{

  FILE* fp;
  int count;

  // parse arguments

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

  // Toplevel

  initialize();

  LoadTrainingSet(fp);

  RandomizeWeights();

  CommandLoop();

  // Done

  fclose(fp);
}
