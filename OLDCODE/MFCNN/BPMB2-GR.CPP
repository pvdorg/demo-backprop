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
#include <graph.h>
#include <malloc.h>

#ifdef __cplusplus
}
#endif

// *****************************************************
// ***** GRAPHICS *****
// *****************************************************

#define MAX_X           16.0
#define MAX_Y           12.0

#define BLACK           0
#define BLUE            1
#define GREEN           2
#define CYAN            3
#define RED             4
#define MAGENTA         5
#define ORANGE          6
#define WHITE           7
#define LIGHTBLACK      8
#define LIGHTBLUE       9
#define LIGHTGREEN      10
#define LIGHTCYAN       11
#define LIGHTRED        12
#define LIGHTMAGENTA    13
#define YELLOW          14
#define HIGHWHITE       15

// *****

void SetColor( FLOAT val )
{
  if( val < 0.1 )
    _setcolor( BLACK );
  else if( val < 0.2 )
    _setcolor( LIGHTBLACK );
  else if( val < 0.3 )
    _setcolor( RED );
  else if( val < 0.4 )
    _setcolor( LIGHTRED );
  else if( val < 0.5 )
    _setcolor( YELLOW );
  else if( val < 0.6 )
    _setcolor( GREEN );
  else if( val < 0.7 )
    _setcolor( LIGHTBLUE );
  else if( val < 0.8 )
    _setcolor( BLUE );
  else if( val < 0.9 )
    _setcolor( MAGENTA );
  else
    _setcolor( WHITE );
}

// *****

void GrInit( void )
{
  _setvideomode( _MAXRESMODE );
  _setwindow( 1, - MAX_X, - MAX_Y, MAX_X, MAX_Y );
  _moveto_w( -MAX_X, 0 );
  _lineto_w( MAX_X, 0 );
  _moveto_w( 0, -MAX_Y );
  _lineto_w( 0, MAX_Y );
}

void GrEnd( void )
{
  fgetc( stdin );
  fgetc( stdin );
  _setvideomode( _DEFAULTMODE );
}

// *****

void DrawInputPoint( FLOAT x, FLOAT y, FLOAT val )
{
  FLOAT delta = 0.1;

  SetColor( val );
  if( val < 0.1 ) {
    _setcolor( BLACK );
    _rectangle_w( _GFILLINTERIOR, x-delta, y+delta, x+delta, y-delta );
    _setcolor( LIGHTBLACK );
    _moveto_w( x-delta, y+delta );
    _lineto_w( x+delta, y+delta );
    _lineto_w( x+delta, y-delta );
    _lineto_w( x-delta, y-delta );
    _lineto_w( x-delta, y+delta );
  } else
    _rectangle_w( _GFILLINTERIOR, x-delta, y+delta, x+delta, y-delta );
}

void DrawTrainingSet( void )
{
  for( int i = 0; i < number_of_trainingsamples; i++ )
    DrawInputPoint( tinput[i][0], tinput[i][1], toutput[i][0] );
}

// *****

void DrawPoint( FLOAT x, FLOAT y, FLOAT val )
{
  SetColor( val );
  _setpixel_w( x, y );
}

// *****

FLOAT CalculateForward( FLOAT x, FLOAT y )
{
  int i,j;
  FLOAT sum;

  // set input layer

  ilayer[0] = x;
  ilayer[1] = y;

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

  return( olayer[0] );
}

void DrawForward( void )
{
  FLOAT x,y;

  for( x = -MAX_X; x < MAX_X; x += step )
    for( y = -MAX_Y; y < MAX_Y; y += step )
      DrawPoint( x, y, CalculateForward(x,y) );
}

// *****

void GrDrawing( void )
{
  GrInit();
  DrawTrainingSet();
  DrawForward();
  DrawTrainingSet();
  GrEnd();
}

// ***** cycle *****

void GrCycle( void )
{
  int count, result;

  printf("\nGive number of cycles : ");
  result = scanf(" %d", &count );
  if( result != 1 )
    {
    printf("Error while reading number of cycles.\n");
    return;
    }

  GrInit();
  for( int i = 0; i < count; i++ ) {
    Cycle(1,0);
    DrawTrainingSet();
    DrawForward();
    DrawTrainingSet();
    }
  GrEnd();
}

// *****

void GrFileOut( void )
{
  long imsize, j;
  char __huge * buffer;
  char __huge * cp;
  int count, cycle_step, result;
  FILE * OutFile;
  char filename[81];

  // get parameters

  printf("\nGive number of images : ");
  result = scanf(" %d", &count );
  if( result != 1 )
    {
    printf("Error while reading number of cycles.\n");
    return;
    }
  printf("\nGive number of cycles/image : ");
  result = scanf(" %d", &cycle_step );
  if( result != 1 )
    {
    printf("Error while reading number of cycles/image.\n");
    return;
    }
  printf("\nGive filename : ");
  result = scanf(" %80s", filename );
  if( result != 1 )
    {
    printf("Error while reading filename.\n");
    return;
    }

  // prepare output file

  OutFile = fopen( filename , "wb" );
  if( !OutFile )
    return;
  fwrite( &count, sizeof(int), 1, OutFile );

  // calculate image size

  GrInit();
  imsize = _imagesize_w( -MAX_X, MAX_Y, MAX_X, -MAX_Y );
//  printf( "image_size = %ld\n", imsize );
  buffer = (char __huge *) _halloc( imsize, sizeof(char) );

  // make images

  for( int i = 0; i < count; i++ ) {
    // iterate
    Cycle(cycle_step,0);
    // draw image
    GrInit();
    DrawTrainingSet();
    DrawForward();
    DrawTrainingSet();
//    _clearscreen( _GCLEARSCREEN );
    printf( "bpmb2 %s : %d", current_trainingfile, cycle_count );
    // capture image
    _getimage_w( -MAX_X, MAX_Y, MAX_X, -MAX_Y, buffer );
    if( _grstatus() )
      {
      _hfree( buffer );
      GrEnd();
      fprintf( stderr, "An error occurred during getimage in step %d\n", i+1 );
      return;
      }
    // write image to file
//    fwrite( buffer, imsize / 2, 2, OutFile );
    for( j = 0, cp = buffer; j < imsize; j++, cp++ ) 
      fputc( *cp, OutFile );
    }

  // all done

  _hfree( buffer );
  fclose( OutFile );
  GrEnd();
}

void GrFileIn( void )
{
  long imsize, j;
  char __huge * buffer;
  char __huge * cp;
  int count, result;
  FILE * InFile;
  char filename[81];

  printf("\nGive filename : ");
  result = scanf(" %80s", filename );
  if( result != 1 )
    {
    printf("Error while reading filename.\n");
    return;
    }

  // Prepare input

  InFile = fopen( filename , "rb" );
  if( !InFile )
    return;

  result = fread( &count, sizeof(int), 1, InFile );
  if( result != 1 )
    {
    printf("Error while reading number of cycles.\n");
    return;
    }

  // calculate image size

  GrInit();
  imsize = _imagesize_w( -MAX_X, MAX_Y, MAX_X, -MAX_Y );
//  printf( "image_size = %ld\n", imsize );
  buffer = (char __huge *) _halloc( imsize, sizeof(char) );

  // read images

  for( int i = 0; i < count; i++ ) {
    for( j = 0, cp = buffer; j < imsize; j++, cp++ ) 
      *cp = fgetc( InFile );
    _putimage_w( -MAX_X, MAX_Y, buffer, _GPSET );
    if( _grstatus() < 0 )
      {
      _hfree( buffer );
      GrEnd();
      return;
      }
    }

  // done

  _hfree( buffer );
  fclose( InFile );
  GrEnd();
}

// *****

void GrSettings( void )
{
  int c;

  for(;;)
    {
    switch ( c = GetCommand(
    "[(1) Step 0.1 / (2) Step 0.2 / (3) Step 0.25 / (4) Step 0.5]") )
      {
      case '1' : step = 0.1;  return; break;
      case '2' : step = 0.2;  return; break;
      case '3' : step = 0.25; return; break;
      case '4' : step = 0.5;  return; break;

      default  : printf("%c is not a valid command\n",c); break;
      }
    }
}

// *****

void TopGraphics( void )
{
  int c;

  for(;;)
    {
    switch ( c = GetCommand("[(C)ycle / (D)rawing / File(I)n / File(O)ut / (S)ettings]") )
      {
      case 'C' : GrCycle();    return;
      case 'D' : GrDrawing();  return;
      case 'I' : GrFileIn();     return;
      case 'O' : GrFileOut();     return;
      case 'S' : GrSettings(); return;

      default  : printf("%c is not a valid command\n",c); break;
      }
    }
}

