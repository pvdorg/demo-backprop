//
//  bpOFX.cpp
//
//  Created by Peter Van Damme on 04-October-2012.
//  Updated by Peter Van Damme on 1-August-2022.
//  Copyright (c) 2012-2022 Peter Van Damme. All rights reserved.
//

#include "ofMain.h"

#include "bp.h"

// *****************************************************
//
// OFX graphics functions for showing the bp-network
//
// *****************************************************

#define USE_COLOR   0

#define USE_PIXELARRAY 1

// *****************************************************
// ***** Pixel array functions *****
// *****************************************************

#if USE_PIXELARRAY

unsigned char	*ourPixels;	 // this will be used to store our pixel array
ofTexture ourTexture;	 // this will be used to draw our pixels

//*****  Pixel by Pixel method that sets a pixel RGB in an x, y position in a pixel array *****
void ourSetPixel(int horizontal,int vertical,
                 unsigned char R,unsigned char G,unsigned char B,
                 int arrayWidth,
                 unsigned char pixels[])
{
	int thisPixel = 3*(arrayWidth * vertical +horizontal);
	pixels[thisPixel]=R;
	pixels[thisPixel+1]=G;
	pixels[thisPixel+2]=B;
}

#endif

// *****************************************************
// ***** Coordinate functions *****
// *****************************************************

#define MAX_X           16.0
#define MAX_Y           12.0
#define STEP            1

// ***** set real values at runtime ****

int canvasWidth;
int canvasHeight;
int originX;
int originY;
FLOAT scaleX;
FLOAT scaleY;

void setCanvas( int width, int height )
{
  canvasWidth = width;
  canvasHeight = height;
  originX = canvasWidth / 2;
  originY = canvasHeight / 2;
  scaleX = canvasWidth / (MAX_X * 2);
  scaleY = canvasHeight / (MAX_Y * 2);
  
#if USE_PIXELARRAY
	// create an ofTexture to hold our image info
	ourTexture = * new ofTexture();
  // allocate memory for the ofTextue (This is RGB no Alpha)	
	ourTexture.allocate(width, height, GL_RGB);
  // create a pixel array to keep our  pixel data (RGB, no Alpha)...
  ourPixels = new unsigned char[width*height*3];
#endif
  

}

int xPixel(FLOAT x)
{
  return( originX + (int)(x * scaleX) );
}

int yPixel(FLOAT y)
{
  return( originY + (int)(y * scaleY) );
}

FLOAT xValue(int x)
{
  return( ((float)(x - originX)) / scaleX );
}

FLOAT yValue(int y)
{
  return( ((float)(y - originY)) / scaleY );
}


// *****************************************************
// ***** Color functions *****
// *****************************************************

ofColor BLACK(0,0,0);
ofColor HIGHWHITE(255,255,255);

ofColor YELLOW(255,255,0);

#if USE_COLOR

ofColor BLUE( 000,000,255);
ofColor GREEN(000,255,000);
ofColor CYAN( 000,255,255);
ofColor RED(  255,000,000);
ofColor MAGENTA(255,000,255);
ofColor ORANGE( 255,128,000);
ofColor WHITE(       127,127,127);
ofColor LIGHTBLACK(   63, 63, 63);
ofColor LIGHTBLUE(     0,  0,127);
ofColor LIGHTGREEN(    0,127,  0);
ofColor LIGHTCYAN(     0,127,127);
ofColor LIGHTRED(    127,  0,  0);
ofColor LIGHTMAGENTA(127,  0,127);

ofColor GetColor( FLOAT val )
{
  ofColor color;
  
  if( val < 0.1 )
    color = BLACK;
  else if( val < 0.2 )
    color = LIGHTBLACK;
  else if( val < 0.3 )
    color = RED;
  else if( val < 0.4 )
    color = LIGHTRED;
  else if( val < 0.5 )
    color = YELLOW;
  else if( val < 0.6 )
    color = GREEN;
  else if( val < 0.7 )
    color = LIGHTBLUE;
  else if( val < 0.8 )
    color = BLUE;
  else if( val < 0.9 )
    color = MAGENTA;
  else
    color = WHITE;
  
  return( color );
}

#else

ofColor GetColor( FLOAT val )
{
  ofColor color;
  
  if( val < 0.0 )
    color = BLACK;
  else if( val > 1.0 )
    color = HIGHWHITE;
  else
  {
    val = ((int)(val * 255 / 10)) * 10;
    color = ofColor(val,val,val);
  }
  
  return( color );
}

unsigned char GetGrey( FLOAT val )
{
  unsigned char grey ;
  
  if( val < 0.0 )
    grey = 0;
  else if( val > 1.0 )
    grey = 255;
  else
  {
    grey = ((int)(val * 255 / 10)) * 10;
  }
  
  return( grey );
}

#endif

// *****************************************************
// ***** Draw functions *****
// *****************************************************

void DrawPoint( FLOAT x, FLOAT y, FLOAT val )
{
  ofColor color = GetColor( val );
  ofRect( xPixel(x), yPixel(y), 1, 1);
}

// ***** Training Set *****

static void DrawInputPoint( FLOAT x, FLOAT y, FLOAT val )
{
  int delta = 2;
  int size = 2*delta + 1;
  
  int xTopLeft = xPixel(x)-delta;
  int yTopLeft = yPixel(y)-delta;
  int xBottomRight = xTopLeft + size;
  int yBottomRight = yTopLeft + size;

  ofSetColor( GetColor( val ) );
  ofRect( xTopLeft, yTopLeft, size, size );
  
  // border
  ofSetColor( YELLOW );
  ofLine( xTopLeft, yTopLeft, xBottomRight, yTopLeft );
  ofLine( xBottomRight, yTopLeft, xBottomRight, yBottomRight );
  ofLine( xBottomRight, yBottomRight, xTopLeft, yBottomRight );
  ofLine( xTopLeft, yBottomRight, xTopLeft, yTopLeft );
  

}

void DrawTrainingSet( void )
{
  ofSetColor( YELLOW );
  ofLine( 0, originY, canvasWidth, originY );
  ofLine( originX, 0, originX, canvasHeight );

  for( int i = 0; i < number_of_trainingsamples; i++ )
    DrawInputPoint( tinput[i][0], tinput[i][1], toutput[i][0] );
}

//--------------------------------------------------------------
// ***** visualize neural net *****
//--------------------------------------------------------------

void DrawNeuralNet( void )
{
  int x,y;
  
  for( x = 0; x < canvasWidth; x += STEP )
    for( y = 0; y < canvasHeight; y += STEP )
    {
      FLOAT val = CalculateOutput(xValue(x),yValue(y));
#if USE_PIXELARRAY
      unsigned char grey = GetGrey(val);
      ourSetPixel(x, y, grey, grey, grey, canvasWidth, ourPixels);
#else
      ofSetColor( GetColor( val ) );
      ofRect( x, y, 1, 1);
#endif
    }

#if USE_PIXELARRAY
  ofSetColor(HIGHWHITE);
  // This loads the pixels into the OF texture..	
  ourTexture.loadData(ourPixels, canvasWidth, canvasHeight, GL_RGB);
  // display the result on the screen .	
	ourTexture.draw(0,0);
#endif

}

