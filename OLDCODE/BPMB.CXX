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

// global constants

#define FLOAT float

#define MAXINPUT		2
#define MAXHIDDEN		2
#define MAXOUTPUT		1
#define MAXTRAININGSAMPLES	100

// global variables

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

void initialize( void )
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
  printf("(%5.3f", ilayer[0]);
  for( int i = 1; i < MAXINPUT; i++ )
    printf(" %5.3f", ilayer[i]);
  printf(")");
}

void DisplayOutput( void )
{
  printf("(%5.3f", olayer[0]);
  for( int i = 1; i < MAXOUTPUT; i++ )
    printf(" %5.3f", olayer[i]);
  printf(")");
}

// ***** Display Weights *****

void DisplayWeights( void )
{
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
}

// ***** Display Test Sample *****

void DisplaySample( int index )
{
  int i;

  printf("(%5.3f", tinput[index][0]);
  for( i = 1; i < MAXINPUT; i++ )
    printf(" %5.3f", tinput[index][i]);
  printf(") -> (%5.3f", toutput[index][0]);
  for( i = 1; i < MAXOUTPUT; i++ )
    printf(" %5.3f", toutput[index][i]);
  printf(")");
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
    printf("Could not read training set from %s.\n", filename);
    return;
    }

  // read count

  result = fscanf(fp, " %d", &number_of_trainingsamples );
  if( result != 1 )
    {
    printf("Error while reading number of trainingsamples for data file.\n");
    return;
    }
  if( number_of_trainingsamples > MAXTRAININGSAMPLES )
    {
    number_of_trainingsamples = MAXTRAININGSAMPLES;
    printf("The program will only read %d training samples.\n");
    }

  // read data

  for( i = 0; i < number_of_trainingsamples; i++ )
    {
    for( j = 0; j < MAXINPUT; j++ )
      {
      result = fscanf(fp, " %f", &tinput[i][j] );
      if( result != 1 )
        {
        printf("Error while reading input layer (sample %d, input %d).\n",
                i, j);
        return;
        }
      }
    for( j = 0; j < MAXOUTPUT; j++ )
      {
      result = fscanf(fp, " %f", &toutput[i][j] );
      if( result != 1 )
        {
        printf("Error while reading output layer (sample %d, output %d).\n",
               i, j);
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
    printf("Could not open environment file %s for reading.\n", filename);
    return;
    }

  // read data (at this moment only the weights and bias)

  result = fscanf(fp, " %d %d", &icount, &hcount);
  if( result != 2 )
    {
    printf("Error reading environment. Could not read dimensions.\n");
    return;
    }
  if( icount != MAXINPUT )
    printf("Error: Dimension ilayer not compatible.\n");
  if( hcount != MAXHIDDEN )
    printf("Error: Dimension hlayer not compatible.\n");

  for(j = 0; j < MAXHIDDEN; j++ )
    {
    for( i = 0; i < MAXINPUT; i++ )
      {
      result = fscanf(fp," %f", &w1[i][j]);
      if( result != 1)
        {
        printf("Error reading w1[%d][%d].\n",i,j);
        return;
        }
      }
    result = fscanf(fp," %f",&bias1[j]);
    if( result != 1)
      {
      printf("Error reading bias1[%d].\n",j);
      return;
      }
    }


  result = fscanf(fp, " %d %d", &hcount, &ocount);
  if( result != 2 )
    {
    printf("Error reading environment. Could not read dimensions.\n");
    return;
    }
  if( hcount != MAXHIDDEN )
    printf("Error: Dimension hlayer not compatible.\n");
  if( ocount != MAXOUTPUT )
    printf("Error: Dimension olayer not compatible.\n");


  for(j = 0; j < MAXOUTPUT; j++ )
    {
    for( i = 0; i < MAXHIDDEN; i++ )
      {
      result = fscanf(fp," %f", &w2[i][j]);
      if( result != 1)
        {
        printf("Error reading w2[%d][%d].\n",i,j);
        return;
        }
      }

    result = fscanf(fp," %f",&bias2[j]);
    if( result != 1)
      {
      printf("Error reading bias2[%d].\n",j);
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
    printf("Could not open environment file %s for writing.\n", filename);
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

void Cycle( int count )
{
  int i,j;
  FLOAT local_error, global_error;

  // header
  printf("\n");
  for(i = 0; i < number_of_trainingsamples; i++)
    {
    printf("+-error-");
    for( j = 0; j < MAXOUTPUT; j++ ) printf("--out---");
    }
  printf("++-global-+\n");

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

      printf("| %5.3f ", local_error);      
      DisplayOutput();
      printf(" ");      
      }
    printf("|| %5.4f |\n", global_error);      
    }

  // footer
  for(i = 0; i < number_of_trainingsamples; i++)
    {
    printf("+-------");
    for( j = 0; j < MAXOUTPUT; j++ ) printf("--------");
    }
  printf("++--------+\n");

}

// *****************************************************
// ***** TOPLEVEL AUXILIARY FUNCTIONS *****
// *****************************************************

// ***** topmenu *****

void ShowMenu( void )
{
  printf("--------------------------------------------------------------\n");
  printf("Training file : %s\n", current_trainingfile);
  printf("--------------------------------------------------------------\n");
  printf("(W)eights ...\n");
  printf("(S)tep\n");
  printf("(C)ycle\n");
  printf("(T)rainingset ...\n");
  printf("(E)nvironment ...\n");
  printf("(V)ariables ...\n");
  printf("(Q)uit\n");
  printf("--------------------------------------------------------------\n");
  printf("Cycle: %5d ----- Learning Rate: %5.3f ----- Momentum: %5.3f \n",
         cycle_count, learning_rate, momentum
        );
  printf("--------------------------------------------------------------\n");
}

// ***** get command *****

int GetCommand( const char * message = "" )
{
  int c;

  printf("Command %s : ", message);

  for(;;)
    {
    c = getchar();
    if( c != '\n' )
      return( toupper(c) );
    }
}

// *****************************************************
// ***** TOPLEVEL COMMANDS *****
// *****************************************************

// ***** RANDOMIZATION *****

void TWDisplayWeights( void )
{
  int i,j;

  printf("\nConnections from input to hidden layer\n");

  for( j = 0; j < MAXHIDDEN; j++ )
    {
    for( i = 0; i < MAXINPUT; i++ )
      printf(" %5.3f", w1[i][j]);
    printf(" [%5.3f]\n", bias1[j]);
    }

  printf("Connections from hidden layer to output\n");

  for( j = 0; j < MAXOUTPUT; j++ )
    {
    for( i = 0; i < MAXHIDDEN; i++ )
      printf(" %5.3f", w2[i][j]);
    printf(" [%5.3f]\n", bias2[j]);
    }
}

// ***** 

void TWRandomize( void )
{
  printf("\nRandomizing Weights ...\n");
  RandomizeWeights();
  TWDisplayWeights();
  cycle_count = 0;
}

// *****

void TopWeights( void )
{
  int c;

  for(;;)
    {
    switch ( c = GetCommand("[(R)andomize / (D)isplay]") )
      {
      case 'R' : TWRandomize();       return; break;

      case 'D' : TWDisplayWeights();  return; break;

      default  : printf("%c is not a valid command\n",c); break;
      }
    }
}

// ***** cycle *****

void TopCycle( void )
{
  int count, result;

  printf("\nGive number of cycles : ");
  result = scanf(" %d", &count );
  if( result != 1 )
    {
    printf("Error while reading number of cycles.\n");
    return;
    }

  Cycle(count);
}

// ***** Step *****

void TopStep( void )
{
  Cycle(1);
}

// ***** TRAINING SET *****

void TSLoad( void )
{
  int result;
  char filename[81];

  printf("\nGive filename : ");

  result = scanf(" %80s", &filename );
  if( result != 1 )
    {
    printf("Error while reading filename.\n");
    return;
    }

  printf("Trying to load %s\n", filename );
  LoadTrainingSet( filename );
}

// *****

void TopTrainingSet( void )
{
  int c;

  for(;;)
    {
    switch ( c = GetCommand("[(L)oad / (D)isplay]") )
      {
      case 'L' : TSLoad();              return; break;

      case 'D' : DisplayTrainingSet();  return; break;

      default  : printf("%c is not a valid command\n",c); break;
      }
    }
}

// ***** ENVIRONMENT *****

void TELoad( void )
{
  int result;
  char filename[81];

  printf("\nGive filename : ");

  result = scanf(" %80s", &filename );
  if( result != 1 )
    {
    printf("Error while reading filename.\n");
    return;
    }

  printf("Trying to load %s\n", filename );
  LoadEnvironment( filename );
  TWDisplayWeights();
}

// *****

void TESave( void )
{
  int result;
  char filename[81];

  printf("\nGive filename : ");

  result = scanf(" %80s", &filename );
  if( result != 1 )
    {
    printf("Error while reading filename.\n");
    return;
    }

  printf("Trying to save environment into file %s\n", filename );
  SaveEnvironment( filename );
}

// *****

void TopEnvironment( void )
{
  int c;

  for(;;)
    {
    switch ( c = GetCommand("[(L)oad / (S)ave]") )
      {
      case 'L' : TELoad();  return; break;

      case 'S' : TESave();  return; break;

      default  : printf("%c is not a valid command\n",c); break;
      }
    }
}

// ***** VARIABLES *****

void TopLearningRate( void )
{
  int result;

  printf("\nGive new learning rate (old value = %5.3f) : ", learning_rate);

  result = scanf(" %f", &learning_rate );
  if( result != 1 )
    {
    printf("Error while reading learning_rate.\n");
    return;
    }
}

// *****

void TopMomentum( void )
{
  int result;

  printf("\nGive new momentum (old value = %5.3f) : ", momentum);

  result = scanf(" %f", &momentum );
  if( result != 1 )
    {
    printf("Error while reading momentum.\n");
    return;
    }
}

// *****

void TopVariables( void )
{
  int c;

  for(;;)
    {
    switch ( c = GetCommand("[(L)earning Rate / (M)omentum]") )
      {
      case 'L' : TopLearningRate();  return; break;

      case 'M' : TopMomentum();      return; break;

      default  : printf("%c is not a valid command\n",c); break;
      }
    }
}

// *****************************************************
// ***** MAIN *****
// *****************************************************

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
      case 'W' : TopWeights();       flag = 1; break;
      case 'S' : TopStep();          flag = 1; break;
      case 'C' : TopCycle();         flag = 1; break;

      case 'T' : TopTrainingSet();   flag = 1; break;
      case 'E' : TopEnvironment();   flag = 1; break;
      case 'V' : TopVariables();     flag = 1; break;

      case 'Q' : return;             break;

      default  : printf("%c is not a valid command\n",c); flag = 0; break;
      }
    }
}

// ***** main *****

int main( int argc, char * argv[] )
{
  // parse arguments

  if ( argc != 2 )
    {
    printf("Usage: %s <training file>\n", argv[0] );
    exit(-1);
    }

  // Toplevel

  initialize();

  LoadTrainingSet(argv[1]);

  RandomizeWeights();

  CommandLoop();

  return(0);

}
