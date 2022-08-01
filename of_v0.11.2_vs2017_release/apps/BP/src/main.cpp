#include "ofMain.h"
#include "testApp.h"
#include "ofAppGlutWindow.h"

//========================================================================
int main( ){

  //ofAppGlutWindow window;
  
  // 4:3 ratio
  //int w = 1024; int h = 768;
  //int w =  800; int h = 600;
  //int w =  640; int h = 480; // screenflow export to youtube & vimeo
  int w =  480; int h = 360; // youtube
	ofSetupOpenGL(w, h, OF_WINDOW);			// <-------- setup the GL context
//	ofSetupOpenGL(&window, w, h, OF_WINDOW);			// <-------- setup the GL context

	// this kicks off the running of my app
	// can be OF_WINDOW or OF_FULLSCREEN
	// pass in width and height too:
	ofRunApp( new testApp());

}
