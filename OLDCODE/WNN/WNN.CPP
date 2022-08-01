/*--------------------------------------------------------
   WNN.C -- Window Neural Net Learning Environment
		 (c) Peter Van Damme, 1993
  --------------------------------------------------------*/

#include <afxwin.h>

#include "wnnmenu.h"
#include "wnn.h"

// ***************************
// * Wnn Application Methods *
// ***************************

BOOL WnnApp::InitInstance()
{
  m_pMainWnd = new WnnMainWindow();
  m_pMainWnd->ShowWindow( m_nCmdShow );
  m_pMainWnd->UpdateWindow();
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
  dc.SetBkMode( TRANSPARENT );
  dc.TextOut( rect.right / 2, rect.bottom / 2, s, s.GetLength() );
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
  
  ON_COMMAND(WNNM_NEW,  OnNotImplemented)
  ON_COMMAND(WNNM_OPEN, OnNotImplemented)
  ON_COMMAND(WNNM_SAVE, OnNotImplemented)
  ON_COMMAND(WNNM_SAVEAS,   OnNotImplemented)
  ON_COMMAND(WNNM_EXIT, OnNotImplemented)

  ON_COMMAND(WNNM_UNDO, OnNotImplemented)
  ON_COMMAND(WNNM_CUT,  OnNotImplemented)
  ON_COMMAND(WNNM_COPY, OnNotImplemented)
  ON_COMMAND(WNNM_PASTE,    OnNotImplemented)
  ON_COMMAND(WNNM_DEL,  OnNotImplemented)

  ON_COMMAND(WNNM_WEIGHTS,  OnNotImplemented)
  ON_COMMAND(WNNM_STEP, OnNotImplemented)
  ON_COMMAND(WNNM_CYCLE,    OnNotImplemented)
  ON_COMMAND(WNNM_TRAININGSET,  OnNotImplemented)
  ON_COMMAND(WNNM_ENVIRONMENT,  OnNotImplemented)
  ON_COMMAND(WNNM_VARIABLES,    OnNotImplemented)
  ON_COMMAND(WNNM_GRAPHICS, OnNotImplemented)

END_MESSAGE_MAP()

// *****************************************************
// ***** TOPLEVEL AUXILIARY FUNCTIONS *****
// *****************************************************

#if PETZOLD

// ***** status *****

void ShowStatus( HDC hdc, short cxChar, short cxCaps, short cyChar )
{
  char szBuffer[100];
  short x = cxChar;
  short y = cyChar;
  
  TextOut(hdc,x,y,szBuffer,sprintf(szBuffer,
    "---------------------------------------------------------------------------"));
  TextOut(hdc,x,y*2,szBuffer,sprintf(szBuffer,
    "Training file : %s", current_trainingfile));
  TextOut(hdc,x,y*3,szBuffer,sprintf(szBuffer,
    "---------------------------------------------------------------------------"));
  TextOut(hdc,x,y*4,szBuffer,sprintf(szBuffer,
    "Cycle: %5d ---- Learning Rate: %5.3f ---- Momentum: %5.3f ---- Step: %4.2f",
	 cycle_count, learning_rate, momentum, step
	));
  TextOut(hdc,x,y*5,szBuffer,sprintf(szBuffer,
    "---------------------------------------------------------------------------"));
}

// ****************************
// * Windows global variables *
// ****************************

char szAppName[] = "WinNeuralNet" ;
char szMenuName[] = "WnnMenu" ;

// ***********
// * WndProc *
// ***********

long FAR PASCAL _export WndProc (HWND hwnd, UINT message, UINT wParam,
							  LONG lParam)
{
  static short cxChar, cxCaps, cyChar;
  char         szBuffer[100];
  HDC          hdc;
  PAINTSTRUCT  ps;
  TEXTMETRIC   tm;
  RECT         rect;
  HMENU        hMenu;

  switch (message)
    {
    case WM_COMMAND :
      hMenu = GetMenu(hwnd);
      switch( wParam )
        {
        case WNNM_NEW :
        case WNNM_OPEN :
        case WNNM_SAVE :
        case WNNM_SAVEAS :
          MessageBeep(0);
          return(0);

        case WNNM_EXIT :
          SendMessage( hwnd, WM_CLOSE, 0, 0L );
          return(0);

        case WNNM_UNDO :
        case WNNM_CUT :
        case WNNM_COPY :
        case WNNM_PASTE :
        case WNNM_DEL :
          MessageBeep(0);
          return(0);

        case WNNM_WEIGHTS :
        case WNNM_STEP :
        case WNNM_CYCLE :
        case WNNM_TRAININGSET :
        case WNNM_ENVIRONMENT :
        case WNNM_VARIABLES :
        case WNNM_GRAPHICS :
          MessageBox( hwnd, "Command not yet implemented!",
                      szAppName, MB_ICONEXCLAMATION | MB_OK );
          return(0);
        }
      break;
      
    case WM_PAINT:
      hdc = BeginPaint (hwnd, &ps) ;
      GetClientRect (hwnd, &rect) ;
      DrawText (hdc, "Hello, WNN!", -1, &rect,
		     DT_SINGLELINE | DT_CENTER | DT_VCENTER) ;
      ShowStatus(hdc,cxChar,cxCaps,cyChar);
      SetTextAlign(hdc, TA_RIGHT | TA_TOP);
      EndPaint (hwnd, &ps) ;
      return 0 ;

    case WM_CREATE:
      hdc = GetDC( hwnd );
      GetTextMetrics( hdc, &tm );
      cxChar = tm.tmAveCharWidth;
      cxCaps = (tm.tmPitchAndFamily & 1 ? 3 : 2 ) * cxChar / 2;
      cyChar = tm.tmHeight + tm.tmExternalLeading;
      ReleaseDC( hwnd, hdc );
      return( 0 );
      
    case WM_DESTROY:
      PostQuitMessage (0) ;
      return 0 ;
    }

  return DefWindowProc (hwnd, message, wParam, lParam) ;
}

// ********
// * MAIN *
// ********

long FAR PASCAL _export WndProc (HWND, UINT, UINT, LONG) ;

int PASCAL WinMain (HANDLE hInstance, HANDLE hPrevInstance,
		    LPSTR lpszCmdParam, int nCmdShow)
{
  HWND        hwnd ;
  MSG         msg ;
  WNDCLASS    wndclass ;

  if (!hPrevInstance)
    {
    wndclass.style         = CS_HREDRAW | CS_VREDRAW ;
    wndclass.lpfnWndProc   = WndProc ;
    wndclass.cbClsExtra    = 0 ;
    wndclass.cbWndExtra    = 0 ;
    wndclass.hInstance     = hInstance ;
    wndclass.hIcon         = LoadIcon (NULL, IDI_APPLICATION) ;
    wndclass.hCursor       = LoadCursor (NULL, IDC_ARROW) ;
    wndclass.hbrBackground = GetStockObject (WHITE_BRUSH) ;
    wndclass.lpszMenuName  = szMenuName ;
    wndclass.lpszClassName = szAppName ;
 
    RegisterClass (&wndclass) ;
    }

  hwnd = CreateWindow (
	   szAppName,         // window class name
	   "The Windows Neural Net Learning Tool",     // window caption
	   WS_OVERLAPPEDWINDOW,     // window style
	   CW_USEDEFAULT,           // initial x position
	   CW_USEDEFAULT,           // initial y position
	   CW_USEDEFAULT,           // initial x size
	   CW_USEDEFAULT,           // initial y size
	   NULL,                    // parent window handle
	   NULL,                    // window menu handle
	   hInstance,               // program instance handle
	   NULL) ;                  // creation parameters

  ShowWindow (hwnd, nCmdShow) ;
  UpdateWindow (hwnd) ;

  while (GetMessage (&msg, NULL, 0, 0))
    {
    TranslateMessage (&msg) ;
    DispatchMessage (&msg) ;
    }
  return msg.wParam ;
}

#endif
