#pragma once

#include "ofMain.h"
#include "SharedData.h"

class DisplayApp : public ofBaseApp
{
public:
  void setup() override {}
  void update() override {}
  void draw() override
  {
    ofBackground(0);
    ofSetColor(255, 0, 0);
    ofDrawRectangle(sharedRect);
  }

  void keyPressed(int key) override
  {
    if (key == 'f')
    {
      ofToggleFullscreen();
    }
    else if (key == OF_KEY_ESC)
    {
      ofExit();
    }
  }

  void keyReleased(int) override {}
  void mouseMoved(int, int) override {}
  void mouseDragged(int, int, int) override {}
  void mousePressed(int, int, int) override {}
  void mouseReleased(int, int, int) override {}
  void mouseEntered(int, int) override {}
  void mouseExited(int, int) override {}
  void windowResized(int, int) override {}
  void dragEvent(ofDragInfo) override {}
  void gotMessage(ofMessage) override {}
};