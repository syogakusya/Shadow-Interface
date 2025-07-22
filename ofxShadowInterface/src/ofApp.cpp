#include "ofApp.h"
#include <algorithm>
#include "SharedData.h"
#include "GestureDetector.h"

//--------------------------------------------------------------
void ofApp::setup()
{
  cam.setDeviceID(1);
  cam.setup(1280, 720);
  thresh = 40;
  pinchDist = 150;
  minPinchDist = 10;
  minArea = 2000;
  maxArea = 50000;
  minDistance = 100;
  shadowMargin = 30;
  maxContours = 3;
  bgCaptured = false;
  colorImg.allocate(cam.getWidth(), cam.getHeight());
  grayImg.allocate(cam.getWidth(), cam.getHeight());
  bgGray.allocate(cam.getWidth(), cam.getHeight());
  diffImg.allocate(cam.getWidth(), cam.getHeight());
  warpedImg.allocate(cam.getWidth(), cam.getHeight());
  isCalibrating = false;
  threshParam.set("Thresh", thresh, 1, 255);
  pinchParam.set("PinchDist", pinchDist, 1, 1000);
  minPinchParam.set("Min PinchDist", minPinchDist, 1, 200);
  minAreaParam.set("Min Area", minArea, 100, 10000);
  maxAreaParam.set("Max Area", maxArea, 10000, 1000000);
  minDistanceParam.set("Min Distance", minDistance, 0, 300);
  shadowMarginParam.set("Shadow Margin", shadowMargin, 0, 100);
  maxContoursParam.set("Max Contours", maxContours, 1, 10);
  gui.setup();
  gui.add(shadowHandStatus.setup("Shadow Hand Status", ""));
  gui.add(threshParam);
  gui.add(pinchParam);
  gui.add(minPinchParam);
  gui.add(autoThreshParam.set("Auto Thresh", true));
  gui.add(minAreaParam);
  gui.add(maxAreaParam);
  gui.add(minDistanceParam);
  gui.add(shadowMarginParam);
  gui.add(maxContoursParam);
  rect.setFromCenter(200, 200, 60, 60);
  dragging = false;
  pinchActive = false;
}

//--------------------------------------------------------------
void ofApp::update()
{
  cam.update();
  if (!cam.isFrameNew())
    return;
  colorImg.setFromPixels(cam.getPixels());
  grayImg = colorImg;

  // もしキャリブレーション済みなら、グレースケール画像にも台形補正を適用
  if (quadPoints.size() == 4)
  {
    cv::Mat gMat = ofxCv::toCv(grayImg);
    cv::Mat gWarped;
    cv::warpPerspective(gMat, gWarped, perspective, gMat.size());
    grayImg.setFromPixels(gWarped.data, gWarped.cols, gWarped.rows);
  }

  thresh = threshParam;
  pinchDist = pinchParam;
  minPinchDist = minPinchParam;
  minArea = minAreaParam;
  maxArea = maxAreaParam;
  minDistance = minDistanceParam;
  shadowMargin = shadowMarginParam;
  maxContours = maxContoursParam;
  if (bgCaptured)
  {
    cv::Mat g = ofxCv::toCv(grayImg);
    cv::Mat bg = ofxCv::toCv(bgGray);

    // 現在の矩形位置とその影響範囲を背景画像で置き換え（背景差分前に実行）
    cv::Rect cvRect(rect.x, rect.y, rect.width, rect.height);

    // 影の範囲も考慮して拡張（矩形の周囲も除外）
    cv::Rect expandedRect(
        std::max(0, cvRect.x - shadowMargin),
        std::max(0, cvRect.y - shadowMargin),
        std::min(g.cols - std::max(0, cvRect.x - shadowMargin), cvRect.width + shadowMargin * 2),
        std::min(g.rows - std::max(0, cvRect.y - shadowMargin), cvRect.height + shadowMargin * 2));

    if (expandedRect.x >= 0 && expandedRect.y >= 0 &&
        expandedRect.x + expandedRect.width <= g.cols &&
        expandedRect.y + expandedRect.height <= g.rows)
    {
      bg(expandedRect).copyTo(g(expandedRect));
    }

    cv::Mat diff;
    cv::absdiff(g, bg, diff);

    // 背景差分なしの場合の前処理を強化
    if (autoThreshParam)
    {
      // ガウシアンブラーで滑らかにしてからOtsu
      cv::GaussianBlur(diff, diff, cv::Size(5, 5), 0);
      double otsu = cv::threshold(diff, diff, 0, 255,
                                  cv::THRESH_BINARY | cv::THRESH_OTSU);
      thresh = static_cast<int>(otsu);
      threshParam = thresh;
    }
    else
    {
      cv::GaussianBlur(diff, diff, cv::Size(5, 5), 0);
      cv::threshold(diff, diff, thresh, 255, cv::THRESH_BINARY);
    }

    // モルフォロジー演算を強化
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(diff, diff, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(diff, diff, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
    diffImg.setFromPixels(diff.data, diff.cols, diff.rows);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(diff, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (!contours.empty())
    {
      // 手らしい輪郭を複数選択（面積、形状、位置を考慮）
      std::vector<std::pair<std::vector<cv::Point>, double>> validContours;

      for (auto &contour : contours)
      {
        double area = cv::contourArea(contour);
        if (area < minArea || area > maxArea)
          continue; // 面積フィルタ

        // 輪郭の重心を計算
        cv::Moments M = cv::moments(contour);
        if (M.m00 == 0)
          continue;
        cv::Point2f centroid(M.m10 / M.m00, M.m01 / M.m00);

        // 矩形から離れているかチェック（手は矩形から離れているはず）
        double distFromRect = cv::norm(centroid - cv::Point2f(rect.getCenter().x, rect.getCenter().y));
        if (distFromRect < minDistance)
          continue; // 矩形に近すぎる場合は除外

        // 形状の複雑さをチェック（手は複雑な形状）
        double perimeter = cv::arcLength(contour, true);
        double complexity = perimeter * perimeter / area; // 複雑さの指標

        // スコア計算（面積と複雑さのバランス）
        double score = area * (complexity / 20.0) * (distFromRect / 200.0);

        if (score > 1000) // 最小スコア閾値
        {
          validContours.push_back({contour, score});
        }
      }

      // スコア順でソート
      std::sort(validContours.begin(), validContours.end(),
                [](const auto &a, const auto &b)
                { return a.second > b.second; });

      // 複数の有効な輪郭を処理（最大数を制限）
      gestures.clear();
      centers.clear();
      bool anyPinch = false;
      bool anyPinchInRect = false;

      int processedCount = 0;
      for (auto &validContour : validContours)
      {
        if (processedCount >= maxContours)
          break;
        if (cv::contourArea(validContour.first) > 3000)
        {
          // 輪郭を滑らかにする
          std::vector<cv::Point> smoothedContour;
          cv::approxPolyDP(validContour.first, smoothedContour, 5, true);

          gestureDetector.setPinchDistance(pinchDist, minPinchDist);
          auto res = gestureDetector.detect(smoothedContour);

          for (auto &p : res)
          {
            gestures.push_back(p.first);
            centers.push_back(p.second);
            if (p.first == "pinch")
            {
              anyPinch = true;
              if (rect.inside(p.second))
                anyPinchInRect = true;
            }
          }
          processedCount++;
        }
      }

      // 有効な輪郭が見つかった場合
      if (!gestures.empty())
      {
        if (anyPinch)
        {
          // ピンチが矩形内にあればドラッグ開始 / 継続
          if (anyPinchInRect)
            dragging = true;

          if (dragging && !centers.empty())
          {
            // 矩形内のピンチのみを抽出
            std::vector<ofPoint> pinchedCenters;
            for (size_t i = 0; i < gestures.size(); ++i)
            {
              if (gestures[i] == "pinch" && rect.inside(centers[i]))
              {
                pinchedCenters.push_back(centers[i]);
              }
            }

            if (!pinchedCenters.empty())
            {
              // 複数ピンチの重心で矩形を移動
              ofPoint centroid(0, 0);
              for (auto &p : pinchedCenters)
              {
                centroid += p;
              }
              centroid /= pinchedCenters.size();
              ofLog() << "centroid: " << centroid << endl;

              // 移動のみ（サイズ変更は削除）
              rect.setFromCenter(centroid, rect.getWidth(), rect.getHeight());
            }
          }
        }
        else
        {
          // ピンチが無ければドラッグ解除
          dragging = false;
        }
        // ステータス表示は最初のジェスチャのみ
        if (!gestures.empty())
          shadowHandStatus.setup(gestures[0]);
      }
      else
      {
        gestureDetector.clear();
      }
    }
  }
  else
  {
    gestureDetector.clear();
  }

  // 共有矩形を最新位置で更新
  sharedRect = rect;
}

//--------------------------------------------------------------
void ofApp::draw()
{
  ofSetColor(255);
  if (quadPoints.size() == 4)
  {
    cv::Mat src = ofxCv::toCv(colorImg);
    cv::Mat dst;
    cv::warpPerspective(src, dst, perspective, src.size());
    warpedImg.setFromPixels(dst.data, dst.cols, dst.rows);
    warpedImg.draw(0, 0);
  }
  else
  {
    colorImg.draw(0, 0);
    for (auto &p : quadPoints)
    {
      ofSetColor(0, 255, 0);
      ofDrawCircle(p, 5);
    }
  }
  ofSetColor(255, 0, 0);
  ofDrawRectangle(rect);

  // ピンチ地点の描画を改善
  for (size_t i = 0; i < gestures.size(); ++i)
  {
    if (gestures[i] == "pinch")
    {
      // 矩形内のピンチは緑、矩形外は黄色
      if (rect.inside(centers[i]))
      {
        ofSetColor(0, 255, 0); // 緑
      }
      else
      {
        ofSetColor(255, 255, 0); // 黄色
      }
      ofDrawCircle(centers[i], 10);

      // ピンチ中心に小さい白い点
      ofSetColor(255, 255, 255);
      ofDrawCircle(centers[i], 3);
    }
  }

  // draw detected contour
  const auto &contourPts = gestureDetector.getContourPts();
  if (!contourPts.empty())
  {
    ofSetColor(0, 255, 0);
    ofPolyline poly;
    for (auto &p : contourPts)
      poly.addVertex(p.x, p.y);
    poly.close();
    poly.draw();
  }

  // convex hull
  const auto &hullPts = gestureDetector.getHullPts();
  if (!hullPts.empty())
  {
    ofNoFill();
    ofSetColor(0, 0, 255);
    ofPolyline hullPoly;
    for (auto &p : hullPts)
      hullPoly.addVertex(p);
    hullPoly.close();
    hullPoly.draw();
    ofFill();
  }

  // fingertip tips
  const auto &tipsPts = gestureDetector.getTipsPts();
  for (auto &p : tipsPts)
  {
    // ピンチ中心から近い指先は赤、そうでなければオレンジ
    bool nearPinch = false;
    for (size_t i = 0; i < gestures.size(); ++i)
    {
      if (gestures[i] == "pinch")
      {
        float dist = ofDist(p.x, p.y, centers[i].x, centers[i].y);
        if (dist < pinchDist * 0.7f)
        { // ピンチ距離の70%以内
          nearPinch = true;
          break;
        }
      }
    }

    if (nearPinch)
    {
      ofSetColor(255, 0, 0); // 赤（ピンチに使用中）
    }
    else
    {
      ofSetColor(255, 165, 0); // オレンジ（通常の指先）
    }
    ofDrawCircle(p, 4);
  }

  gui.draw();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key)
{
  if (key == 'b')
  {
    bgGray = grayImg;
    bgCaptured = true;
  }
  else if (key == 'c')
  {
    quadPoints.clear();
    isCalibrating = true;
  }
  else if (key == 'r')
  {
    bgCaptured = false;
    dragging = false;
    pinchActive = false;
    rect.setFromCenter(200, 200, 60, 60);
  }
  else if (key == OF_KEY_ESC)
  {
    ofExit();
  }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y)
{
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button)
{
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button)
{
  if (isCalibrating)
  {
    quadPoints.push_back(ofPoint(x, y));
    if (quadPoints.size() == 4)
    {
      std::vector<cv::Point2f> src(4), dst(4);
      for (int i = 0; i < 4; ++i)
        src[i] = cv::Point2f(quadPoints[i].x, quadPoints[i].y);
      dst[0] = cv::Point2f(0, 0);
      dst[1] = cv::Point2f(cam.getWidth(), 0);
      dst[2] = cv::Point2f(cam.getWidth(), cam.getHeight());
      dst[3] = cv::Point2f(0, cam.getHeight());
      perspective = cv::getPerspectiveTransform(src, dst);
      isCalibrating = false;
    }
  }
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button)
{
}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y)
{
}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y)
{
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h)
{
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg)
{
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo)
{
}
