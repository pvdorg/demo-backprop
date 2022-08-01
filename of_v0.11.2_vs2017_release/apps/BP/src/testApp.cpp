#include "testApp.h"

#include "bp.h"
#include "bpOFX.h"

#define SHOW_FRAMENUMBER 1

//--------------------------------------------------------------

void testApp::setup(){
  //ofSetFrameRate(60);
  
  // nice to have width and height as globals and not have to call ofGetWidth() all the time
  width = ofGetWidth();
	height = ofGetHeight();
  setCanvas(width, height);
  
  //***** get training samples *****
  
  // training file
  const char * training_file = "C:\\PVDDATA\\of_v0.11.2_vs2017_release\\apps\\BP\\src\\XOR.DAT";
  
  FILE* fp;
  fp = fopen( training_file, "r" );
  if( !fp )
  {
    printf("Could not open training file.\n");
    exit();
  }

  // ***** setup neural network *****
  
  initialize();
  
  LoadTrainingSet(fp);
  
  RandomizeWeights();

}

//--------------------------------------------------------------

void testApp::update(){
  
  // ***** do one learning cycle
  Cycle(1);
  
}

//--------------------------------------------------------------

void testApp::draw(){

  /*
   

   char s[1000];
   sprintf(s, "test (%5.3f %5.3f) -> %5.3f -> output %5.3f -> error %5.3f (%5.3f)",
   ilayer[0], ilayer[1], toutput[i][0], olayer[0],
   local_error, global_error
   );
   
   ofDrawBitmapString(ofToString(s), 10, 30 + (i * 15));
   */
  
  DrawNeuralNet();
  
  DrawTrainingSet();
  
#if SHOW_FRAMENUMBER
  ofSetColor(255,255,0);
  ofDrawBitmapString(ofToString(ofGetFrameNum()), 10, 15);
//  ofDrawBitmapString(ofToString(ofGetFrameNum()) + " - " + ofToString(ofGetFrameRate())+"fps", 10, 15);
#endif
  
}

//--------------------------------------------------------------
bool key_active = false;
void testApp::keyPressed(int key){
  //cout << "keyPressed " << (char)key << endl;
  key_active = true;
}

//--------------------------------------------------------------
void testApp::keyReleased(int key){
  //cout << "keyReleased " << (char)key << endl;
  key_active = false;
}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y ){
  //    cout << "mouseMoved: " << x << ", " << y << endl;
}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){
  //    cout << "mouseDragged: " << x << ", " << y << " button: " << button << endl;
}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){
  //    cout << "mousePressed: " << x << ", " << y << " button: " << button << endl;
}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){
  //cout << "mouseReleased: " << x << ", " << y << " button: " << button << endl;
  
  FLOAT val = 1.0;
  if(key_active) val = 0.0;
  AddTrainingSample(xValue(x),yValue(y),val);
}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){
  
}

//--------------------------------------------------------------
void testApp::gotMessage(ofMessage msg){
  
}

//--------------------------------------------------------------
void testApp::dragEvent(ofDragInfo dragInfo){ 
  
}