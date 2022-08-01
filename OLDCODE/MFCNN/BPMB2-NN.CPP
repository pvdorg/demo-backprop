// *****************************************************
//
// Backpropagation network with 1 hidden layer.
//
// - with learning rate and momentum
//
// *****************************************************

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef __cplusplus
}
#endif

#include "bpmb2-nn.h"

// ********************
// * global variables *
// ********************

FLOAT learning_rate = 0.3;
FLOAT momentum      = 0.9;

int cycle_count               = 0;

int number_of_trainingsamples = 0;
char current_trainingfile[81] = "";

// layers

FLOAT ilayer[MAXINPUT];
FLOAT hlayer[MAXHIDDEN];
FLOAT olayer[MAXOUTPUT];

// weights

FLOAT w1[MAXINPUT][MAXHIDDEN];
FLOAT w2[MAXHIDDEN][MAXOUTPUT];

FLOAT bias1[MAXHIDDEN]; 
FLOAT bias2[MAXOUTPUT];

// weight changes

FLOAT delta_w1[MAXINPUT][MAXHIDDEN];
FLOAT delta_w2[MAXHIDDEN][MAXOUTPUT];

FLOAT delta_bias1[MAXHIDDEN]; 
FLOAT delta_bias2[MAXOUTPUT];

// deltas

FLOAT hdelta[MAXHIDDEN];
FLOAT odelta[MAXOUTPUT];

// training set

FLOAT tinput[MAXTRAININGSAMPLES][MAXINPUT];
FLOAT toutput[MAXTRAININGSAMPLES][MAXOUTPUT];

// *****************************************************
// INITIALIZATION
// *****************************************************

void InitializeNetwork( void )
{
  int i,j;

  for( i = 0; i < MAXINPUT; i++ )
    {
    ilayer[i] = 0.0;

    for( j = 0; j < MAXHIDDEN; j++ )
      {
      delta_w1[i][j] = 0.0;
      }
    }

  for( i = 0; i < MAXHIDDEN; i++ )
    {
    hlayer[i] = 0.0;
    delta_bias1[i] = 0.0;

    for( j = 0; j < MAXOUTPUT; j++ )
      {
      delta_w2[i][j] = 0.0;
      }
    }

  for( i = 0; i < MAXOUTPUT; i++ )
    {
    olayer[i] = 0.0;
    delta_bias2[i] = 0.0;
    }

}

// ***** Randomization of weights *****

void RandomizeWeights( void )
{
  int i,j;

  for( i = 0; i < MAXINPUT; i++ )
    for( j = 0; j < MAXHIDDEN; j++ )
      w1[i][j]       = ( (FLOAT) rand() ) / ( (FLOAT) RAND_MAX );

  for( i = 0; i < MAXHIDDEN; i++ )
    {
    // w2
    for( j = 0; j < MAXOUTPUT; j++ )
      w2[i][j] = ( (FLOAT) rand() ) / ( (FLOAT) RAND_MAX );
    // bias1
    bias1[i]   = ( (FLOAT) rand() ) / ( (FLOAT) RAND_MAX );
    }

  for( i = 0; i < MAXOUTPUT; i++ )
    {
    // bias2
    bias2[i]   = ( (FLOAT) rand() ) / ( (FLOAT) RAND_MAX );
    }
}


// *****************************************************
// ***** AUXILIARY FUNCTIONS *****
// *****************************************************

FLOAT Sigmoid( FLOAT x )
{
  return( 1 / (1+exp(-x)) );
}

// *****************************************************
// ***** DISPLAY FUNCTIONS *****
// *****************************************************

// ***** Display Input-Output *****

void DisplayInput( void )
{
#ifdef DOSVERSION
  printf("(%5.3f", ilayer[0]);
  for( int i = 1; i < MAXINPUT; i++ )
    printf(" %5.3f", ilayer[i]);
  printf(")");
#endif
}

void DisplayOutput( void )
{
#ifdef DOSVERSION
  printf("(%5.3f", olayer[0]);
  for( int i = 1; i < MAXOUTPUT; i++ )
    printf(" %5.3f", olayer[i]);
  printf(")");
#endif
}

// ***** Display Weights *****

void DisplayWeights( void )
{
#ifdef DOSVERSION
  int i,j;

  printf("(");

  for( j = 0; j < MAXHIDDEN; j++ )
    {
    for( i = 0; i < MAXINPUT; i++ )
      printf(" %5.3f", w1[i][j]);
    printf(" [%5.3f]", bias1[j]);
    }

  printf(") - (");

  for( j = 0; j < MAXOUTPUT; j++ )
    {
    for( i = 0; i < MAXHIDDEN; i++ )
      printf(" %5.3f", w2[i][j]);
    printf(" [%5.3f]", bias2[j]);
    }

  printf(")\n");
#endif
}

// ***** Display Test Sample *****

void DisplaySample( int index )
{
#ifdef DOSVERSION
  int i;

  printf("(%5.3f", tinput[index][0]);
  for( i = 1; i < MAXINPUT; i++ )
    printf(" %5.3f", tinput[index][i]);
  printf(") -> (%5.3f", toutput[index][0]);
  for( i = 1; i < MAXOUTPUT; i++ )
    printf(" %5.3f", toutput[index][i]);
  printf(")");
#endif
}

// *****************************************************
// ***** TRAINING DATA *****
// *****************************************************

void LoadTrainingSet( const char * filename )
{
  int i,j, result;
  FILE * fp;

  // try to open the file

  fp = fopen( filename, "r" );
  if( !fp )
    {
#ifdef DOSVERSION
    printf("Could not read training set from %s.\n", filename);
#endif
    return;
    }

  // read count

  result = fscanf(fp, " %d", &number_of_trainingsamples );
  if( result != 1 )
    {
#ifdef DOSVERSION
    printf("Error while reading number of trainingsamples for data file.\n");
#endif
    return;
    }
  if( number_of_trainingsamples > MAXTRAININGSAMPLES )
    {
    number_of_trainingsamples = MAXTRAININGSAMPLES;
#ifdef DOSVERSION
    printf("The program will only read %d training samples.\n");
#endif
    }

  // read data

  for( i = 0; i < number_of_trainingsamples; i++ )
    {
    for( j = 0; j < MAXINPUT; j++ )
      {
      result = fscanf(fp, " %f", &tinput[i][j] );
      if( result != 1 )
	{
#ifdef DOSVERSION
	printf("Error while reading input layer (sample %d, input %d).\n",
		i, j);
#endif
	return;
	}
      }
    for( j = 0; j < MAXOUTPUT; j++ )
      {
      result = fscanf(fp, " %f", &toutput[i][j] );
      if( result != 1 )
	{
#ifdef DOSVERSION
	printf("Error while reading output layer (sample %d, output %d).\n",
	       i, j);
#endif
	return;
	}
      }
    }

  // close file and set global variable

  fclose(fp);
  strcpy(current_trainingfile,filename);
}

// ***** display *****

void DisplayTrainingSet( void )
{
#ifdef DOSVERSION
  int i,j;

  printf("\nNumber of samples = %d\n", number_of_trainingsamples);

  printf(" --------");
  for( j = 0; j < MAXINPUT; j++ ) printf("--------");
  printf("-");
  for( j = 0; j < MAXOUTPUT; j++ ) printf("--------");
  printf(" \n");

  for( i = 0; i < number_of_trainingsamples; i++ )
    {
    printf("| %5d |", i + 1);
    for( j = 0; j < MAXINPUT; j++ ) printf("| %5.3f ", tinput[i][j] );
    printf("|");
    for( j = 0; j < MAXOUTPUT; j++ ) printf("| %5.3f ", toutput[i][j] );
    printf("|\n");
    }

  printf(" --------");
  for( j = 0; j < MAXINPUT; j++ ) printf("--------");
  printf("-");
  for( j = 0; j < MAXOUTPUT; j++ ) printf("--------");
  printf(" \n");
#endif
}

// *****************************************************
// ***** ENVIRONMENT *****
// *****************************************************

// ***** load *****

void LoadEnvironment( char * filename )
{
  int i,j,icount,hcount,ocount, result;
  FILE * fp;

  // try to open the file

  fp = fopen( filename, "r" );
  if( !fp )
    {
#ifdef DOSVERSION
    printf("Could not open environment file %s for reading.\n", filename);
#endif
    return;
    }

  // read data (at this moment only the weights and bias)

  result = fscanf(fp, " %d %d", &icount, &hcount);
  if( result != 2 )
    {
#ifdef DOSVERSION
    printf("Error reading environment. Could not read dimensions.\n");
#endif
    return;
    }
#ifdef DOSVERSION
  if( icount != MAXINPUT )
    printf("Error: Dimension ilayer not compatible.\n");
  if( hcount != MAXHIDDEN )
    printf("Error: Dimension hlayer not compatible.\n");
#endif

  for(j = 0; j < MAXHIDDEN; j++ )
    {
    for( i = 0; i < MAXINPUT; i++ )
      {
      result = fscanf(fp," %f", &w1[i][j]);
      if( result != 1)
	{
#ifdef DOSVERSION
	printf("Error reading w1[%d][%d].\n",i,j);
#endif
	return;
	}
      }
    result = fscanf(fp," %f",&bias1[j]);
    if( result != 1)
      {
#ifdef DOSVERSION
      printf("Error reading bias1[%d].\n",j);
#endif
      return;
      }
    }


  result = fscanf(fp, " %d %d", &hcount, &ocount);
  if( result != 2 )
    {
#ifdef DOSVERSION
    printf("Error reading environment. Could not read dimensions.\n");
#endif
    return;
    }
#ifdef DOSVERSION
  if( hcount != MAXHIDDEN )
    printf("Error: Dimension hlayer not compatible.\n");
  if( ocount != MAXOUTPUT )
    printf("Error: Dimension olayer not compatible.\n");
#endif

  for(j = 0; j < MAXOUTPUT; j++ )
    {
    for( i = 0; i < MAXHIDDEN; i++ )
      {
      result = fscanf(fp," %f", &w2[i][j]);
      if( result != 1)
	{
#ifdef DOSVERSION
	printf("Error reading w2[%d][%d].\n",i,j);
#endif
	return;
	}
      }

    result = fscanf(fp," %f",&bias2[j]);
    if( result != 1)
      {
#ifdef DOSVERSION
      printf("Error reading bias2[%d].\n",j);
#endif
      return;
      }
    }

  // close file

  fclose(fp);
}

// ***** save *****

void SaveEnvironment( char * filename )
{
  int i,j;
  FILE * fp;

  // try to open the file

  fp = fopen( filename, "w" );
  if( !fp )
    {
#ifdef DOSVERSION
    printf("Could not open environment file %s for writing.\n", filename);
#endif
    return;
    }

  // write data (at this moment only the weights and bias

  fprintf(fp, "%d %d\n", MAXINPUT, MAXHIDDEN);
  for(j = 0; j < MAXHIDDEN; j++ )
    {
    for( i = 0; i < MAXINPUT; i++ )
      {
      fprintf(fp,"%f ",w1[i][j]);
      }
    fprintf(fp,"%f\n",bias1[j]);
    }

  fprintf(fp, "%d %d\n", MAXHIDDEN, MAXOUTPUT);
  for(j = 0; j < MAXOUTPUT; j++ )
    {
    for( i = 0; i < MAXHIDDEN; i++ )
      {
      fprintf(fp,"%f ",w2[i][j]);
      }
    fprintf(fp,"%f\n",bias2[j]);
    }

  // close file

  fclose(fp);
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
    sum = bias1[j];
    for( i = 0;  i < MAXINPUT; i++ )
      {
      sum = sum + (w1[i][j]*ilayer[i]);
      }
    hlayer[j] = Sigmoid(sum);
    }

  for( j = 0; j < MAXOUTPUT; j++ )
    {
    sum = bias2[j];
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
//
// Remark: for a bias neuron we don't need to calculate
//         a delta because deltas are used in the adjustment
//         of incoming connections.
//         A bias only has outgoing connections.
//

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
//
// Don't forget the weights for the bias.
//

void UpdateWeights( void )
{
  int i,j;

  // ***** connections to output layer *****

  for( i = 0; i < MAXOUTPUT; i++ )
    {
    // ***** bias weight *****

    // calculate new delta
    delta_bias2[i] = ( learning_rate * odelta[i] )
		       +   // add momentum term
		       ( momentum * delta_bias2[i]);
    // update weight
    bias2[i] += delta_bias2[i];

    // ***** normal weights *****

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

  // ***** connections to hidden layer *****

  for( i = 0; i < MAXHIDDEN; i++ )
    {
    // ***** bias weight *****

    // calculate new delta
    delta_bias1[i] = ( learning_rate * hdelta[i] )
		       +   // add momentum term
		       ( momentum * delta_bias1[i]);
    // update weight
    bias1[i] += delta_bias1[i];

    // ***** normal weights *****

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

void Cycle( int count, int printit )
{
  int i,j;
  FLOAT local_error, global_error;

  // header
#ifdef DOSVERSION
  if( printit ) {
    printf("\n");
    for(i = 0; i < number_of_trainingsamples; i++)
      {
      printf("+-error-");
      for( j = 0; j < MAXOUTPUT; j++ ) printf("--out---");
      }
    printf("++-global-+\n");
    }
#endif

  // main body
  for( j = 0; j < count; j++ )
    {
    cycle_count++;
    global_error = 0.0;
    for(i = 0; i < number_of_trainingsamples; i++)
      {
      ForwardPropagation(i);
      local_error = BackPropagation(i);
      global_error += local_error;

#ifdef DOSVERSION
      if( printit ) {
	printf("| %5.3f ", local_error);      
	DisplayOutput();
	printf(" ");
	}
#endif
      }
#ifdef DOSVERSION
    if( printit ) printf("|| %5.4f |\n", global_error);      
#endif
    }

  // footer
#ifdef DOSVERSION
  if( printit ) {
    for(i = 0; i < number_of_trainingsamples; i++)
      {
      printf("+-------");
      for( j = 0; j < MAXOUTPUT; j++ ) printf("--------");
      }
    printf("++--------+\n");
    }
#endif
}

