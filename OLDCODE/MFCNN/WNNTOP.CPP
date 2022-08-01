// Windows Neural Networks Learning Environment
// Peter Van Damme, 1993

#include <afxwin.h>

#include "wnn.h"
#include "bpmb2-nn.h"

// ***************************
// * Wnn Main Window Methods *
// ***************************

// *****************************
// ***** FILE MENU METHODS *****
// *****************************

// ***** LOAD TRAINING SET *****

void WnnMainWindow::OnFileOpen( void )
{
#ifdef DOSVERSION
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
#endif

  LoadTrainingSet( "or.dat" );
}

// ***** EXIT *****

void WnnMainWindow::OnFileExit( void )
{
  DestroyWindow();
}

// *****************************
// ***** FILE MENU METHODS *****
// *****************************

// ***** Toplevel commands *****

void WnnMainWindow::OnStep()
{
  Cycle(1,1);
}

