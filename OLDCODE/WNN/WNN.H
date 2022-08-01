//
// WNN.H
//

#ifndef WNN_H
#define WNN_H

// windows classes

#define APPLICATION_NAME "Windows Neural Networks Learning Environment"

class WnnApp : public CWinApp
{
public:
  BOOL InitInstance();
};

class WnnMainWindow : public CFrameWnd
{
public:
  WnnMainWindow();

  afx_msg void OnPaint();
  afx_msg void OnAbout();
  afx_msg void OnNotImplemented();

  DECLARE_MESSAGE_MAP()
};

// global constants

#define FLOAT float

#define MAXINPUT                2
#define MAXHIDDEN               2
#define MAXOUTPUT               1
#define MAXTRAININGSAMPLES      100

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

// graphics

FLOAT step = 0.1;



#endif
