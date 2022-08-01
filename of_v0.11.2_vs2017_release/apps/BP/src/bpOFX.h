//
//  bpOFX.h
//
//  Created by Peter Van Damme on 04-October-2012.
//  Updated by Peter Van Damme on 1-August-2022.
//  Copyright (c) 2012-2022 Peter Van Damme. All rights reserved.
//

#ifndef bpOFX_h
#define bpOFX_h

extern int canvasWidth;
extern int canvasHeight;

void setCanvas( int width, int height );
int xPixel(FLOAT x);
int yPixel(FLOAT y);
FLOAT xValue(int x);
FLOAT yValue(int y);

void DrawTrainingSet( void );
void DrawNeuralNet( void );

#endif
