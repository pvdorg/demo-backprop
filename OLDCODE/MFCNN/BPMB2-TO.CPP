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

#include <math.h>
#include <string.h>

#ifdef __cplusplus
}
#endif

#include "bpmb2-nn.h"

// *****************************
// ***** TOPLEVEL COMMANDS *****
// *****************************

//////////////////////////////////////////////////////////

#ifdef DOSVERSION

// *****************************************************
// ***** TOPLEVEL AUXILIARY FUNCTIONS *****
// *****************************************************

// ***** topmenu *****

void ShowMenu( void )
{
  printf("---------------------------------------------------------------------------\n");
  printf("Training file : %s\n", current_trainingfile);
  printf("---------------------------------------------------------------------------\n");
  printf("(W)eights ...\n");
  printf("(C)ycle\n");
  printf("(T)rainingset ...\n");
  printf("(E)nvironment ...\n");
  printf("(V)ariables ...\n");
  printf("(G)raphics ...\n");
  printf("(Q)uit\n");
  printf("---------------------------------------------------------------------------\n");
  printf("Cycle: %5d ---- Learning Rate: %5.3f ---- Momentum: %5.3f ---- Step: %4.2f\n",
	 cycle_count, learning_rate, momentum, step
	);
  printf("---------------------------------------------------------------------------\n");
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

  Cycle(count,1);
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

      case 'G' : TopGraphics();      flag = 1; break;

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

#endif
