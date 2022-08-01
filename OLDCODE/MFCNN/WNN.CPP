// Windows Neural Networks Learning Environment
// Peter Van Damme, 1993

#include <afxwin.h>
#include "wnnmenu.h"

#include "wnn.h"
#include "bpmb2-nn.h"

/////////////////////////////////////////////////////////////////////////////

// theApp:
// Just creating this application object runs the whole application.
//
WnnApp theApp;

// ***************************
// * Wnn Application Methods *
// ***************************

BOOL WnnApp::InitInstance()
{
  m_pMainWnd = new WnnMainWindow();
  m_pMainWnd->ShowWindow( m_nCmdShow );
  m_pMainWnd->UpdateWindow();

  InitializeNetwork();
  RandomizeWeights();
  
  return TRUE;
}

// ***************************
// * Wnn Main Window Methods *
// ***************************

WnnMainWindow::WnnMainWindow()
{
  LoadAccelTable( "WnnAccelTable" );
  Create( NULL, APPLICATION_NAME,
          WS_OVERLAPPEDWINDOW, rectDefault, NULL,
          "WnnMenu" );
}

// ***** OnPaint *****

void WnnMainWindow::OnPaint()
{
  CString s = "Welcome to Windows Neural Networks Learning Environment.";
  CPaintDC dc( this );
  CRect rect;

  GetClientRect( rect );
  dc.SetTextAlign( TA_BASELINE | TA_CENTER );
  dc.SetTextColor( ::GetSysColor( COLOR_WINDOWTEXT ) );
  dc.SetBkMode( TRANSPARENT );
  dc.TextOut( ( rect.right / 2 ), ( rect.bottom / 2 ), s, s.GetLength() );
}

// ***** OnAbout *****

void WnnMainWindow::OnAbout()
{
  CModalDialog about( "AboutBox", this );
  about.DoModal();
}

// ***** OnNotImplemented *****

void WnnMainWindow::OnNotImplemented()
{
  MessageBox( "Command not yet implemented!",
              APPLICATION_NAME, MB_ICONEXCLAMATION | MB_OK );
}

// ***** message map *****

BEGIN_MESSAGE_MAP(WnnMainWindow, CFrameWnd)
  ON_WM_PAINT()
  ON_COMMAND(WNNM_ABOUT, OnAbout)
  
  ON_COMMAND(WNNM_NEW,      OnNotImplemented)
  ON_COMMAND(WNNM_OPEN,     OnFileOpen)
  ON_COMMAND(WNNM_SAVE,     OnNotImplemented)
  ON_COMMAND(WNNM_SAVEAS,   OnNotImplemented)
  ON_COMMAND(WNNM_EXIT,     OnFileExit)

  ON_COMMAND(WNNM_UNDO,     OnNotImplemented)
  ON_COMMAND(WNNM_CUT,      OnNotImplemented)
  ON_COMMAND(WNNM_COPY,     OnNotImplemented)
  ON_COMMAND(WNNM_PASTE,    OnNotImplemented)
  ON_COMMAND(WNNM_DEL,      OnNotImplemented)

  ON_COMMAND(WNNM_WEIGHTS,      OnNotImplemented)
  ON_COMMAND(WNNM_STEP,         OnStep)
  ON_COMMAND(WNNM_CYCLE,        OnNotImplemented)
  ON_COMMAND(WNNM_TRAININGSET,  OnNotImplemented)
  ON_COMMAND(WNNM_ENVIRONMENT,  OnNotImplemented)
  ON_COMMAND(WNNM_VARIABLES,    OnNotImplemented)
  ON_COMMAND(WNNM_GRAPHICS,     OnNotImplemented)
END_MESSAGE_MAP()

