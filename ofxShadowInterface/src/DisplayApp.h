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
    // カメラ解像度 (guiWindowでの描画基準) と displayWindow の現在サイズからスケール係数を計算
    const float camW = 1280.0f;
    const float camH = 720.0f;

    // アスペクト比を保持してスケーリング
    float scaleX = ofGetWidth() / camW;
    float scaleY = ofGetHeight() / camH;
    float scale = std::min(scaleX, scaleY); // 小さい方のスケールを使用

    // 中央に配置するためのオフセット計算
    float offsetX = (ofGetWidth() - camW * scale) * 0.5f;
    float offsetY = (ofGetHeight() - camH * scale) * 0.5f;

    // スケーリング後の矩形を生成（中央配置）
    ofRectangle scaledRect(
        sharedRect.x * scale + offsetX,
        sharedRect.y * scale + offsetY,
        sharedRect.getWidth() * scale,
        sharedRect.getHeight() * scale);

    ofSetColor(255, 0, 0);
    ofDrawRectangle(scaledRect);
    ofDrawBitmapString("RedRect: " + ofToString(scaledRect.x) + ", " + ofToString(scaledRect.y) + ", " + ofToString(scaledRect.width) + ", " + ofToString(scaledRect.height), 10, 10);
    ofDrawBitmapString("ofGetWidth(): " + ofToString(ofGetWidth()) + ", ofGetHeight(): " + ofToString(ofGetHeight()), 10, 30);
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