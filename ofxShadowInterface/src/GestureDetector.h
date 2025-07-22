#pragma once

#include "ofMain.h"
#include "ofxOpenCv.h"
#include "ofxCv.h"
#include <vector>
#include <string>

class GestureDetector
{
public:
  GestureDetector();

  // ピンチ判定に用いる距離を設定
  void setPinchDistance(int maxDist, int minDist = 0);

  // 与えられた輪郭からジェスチャーを検出して返す
  // 返り値: { gestureName, centerPoint }
  std::vector<std::pair<std::string, ofPoint>> detect(const std::vector<cv::Point> &contour);

  // 描画用の補助情報
  const std::vector<ofPoint> &getContourPts() const { return contourPts; }
  const std::vector<ofPoint> &getHullPts() const { return hullPts; }
  const std::vector<ofPoint> &getTipsPts() const { return tipsPts; }

  // 情報をクリア
  void clear();

private:
  int pinchDist;
  int minPinchDist;

  // 検出に使用した各種ポイント
  std::vector<ofPoint> contourPts;
  std::vector<ofPoint> hullPts;
  std::vector<ofPoint> tipsPts;
};