//
// WNN.H
//

#ifndef WNN_H
#define WNN_H

// windows classes

#define APPLICATION_NAME "Windows Neural Networks Learning Environment"

// *********************
// * Application Class *
// *********************

class WnnApp : public CWinApp
{
public:
  BOOL InitInstance();
};

// ***************
// * Main Window *
// ***************

class WnnMainWindow : public CFrameWnd
{
public:
  WnnMainWindow();

  afx_msg void OnPaint();
  afx_msg void OnAbout();
  afx_msg void OnNotImplemented();

  afx_msg void OnFileOpen();
  afx_msg void OnFileExit();

  afx_msg void OnStep();

  DECLARE_MESSAGE_MAP()
};


#endif

