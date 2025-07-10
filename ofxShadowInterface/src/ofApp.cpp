#include "ofApp.h"
#include <algorithm>
#include "SharedData.h"

//--------------------------------------------------------------
void ofApp::setup()
{
  cam.setDeviceID(1);
  cam.setup(1280, 720);
  thresh = 40;
  pinchDist = 150;
  bgCaptured = false;
  colorImg.allocate(cam.getWidth(), cam.getHeight());
  grayImg.allocate(cam.getWidth(), cam.getHeight());
  bgGray.allocate(cam.getWidth(), cam.getHeight());
  diffImg.allocate(cam.getWidth(), cam.getHeight());
  warpedImg.allocate(cam.getWidth(), cam.getHeight());
  isCalibrating = false;
  threshParam.set("Thresh", thresh, 1, 255);
  pinchParam.set("PinchDist", pinchDist, 1, 1000);
  gui.setup();
  gui.add(shadowHandStatus.setup("Shadow Hand Status", ""));
  gui.add(threshParam);
  gui.add(pinchParam);
  gui.add(autoThreshParam.set("Auto Thresh", true));
  rect.setFromCenter(200, 200, 60, 60);
  dragging = false;
  pinchActive = false;

  box2d.init();
  box2d.setGravity(0, 10);
  box2d.createGround();
  box2d.setFPS(60.0);

  shadowHandBodyCreated = false;

  for (int i = 0; i < 5; i++)
  {
    auto box = std::make_shared<ofxBox2dRect>();
    box->setPhysics(3.0, 0.53, 0.1);
    box->setup(box2d.getWorld(), ofRandom(100, 700), ofRandom(100, 300), 40, 40);
    boxes.push_back(box);
  }
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
  if (bgCaptured)
  {
    cv::Mat g = ofxCv::toCv(grayImg);
    cv::Mat bg = ofxCv::toCv(bgGray);
    cv::Mat diff;
    cv::absdiff(g, bg, diff);
    if (autoThreshParam)
    {
      // Otsu で自動決定（返り値が推奨しきい値）
      double otsu = cv::threshold(diff, diff, 0, 255,
                                  cv::THRESH_BINARY | cv::THRESH_OTSU);
      thresh = static_cast<int>(otsu); // GUI スライダーにも反映したいなら
      threshParam = thresh;            // ←これで表示が追従
    }
    else
    {
      cv::threshold(diff, diff, thresh, 255, cv::THRESH_BINARY);
    }
    cv::morphologyEx(diff, diff, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 2);
    diffImg.setFromPixels(diff.data, diff.cols, diff.rows);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(diff, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (!contours.empty())
    {
      auto it = std::max_element(contours.begin(), contours.end(), [](const auto &a, const auto &b)
                                 { return cv::contourArea(a) < cv::contourArea(b); });
      if (cv::contourArea(*it) > 3000)
      {
        // save contour for drawing
        contourPts.clear();
        for (auto &pt : *it)
          contourPts.emplace_back(pt.x, pt.y);

        auto res = detectPinch(*it);
        gesture = res.first;
        center = res.second;
        shadowHandStatus.setup(gesture);

        if (!shadowHandBodyCreated)
        {
          createShadowHandBody(*it);
        }
        else
        {
          updateShadowHandBody(*it);
        }

        if (gesture == "pinch")
        {
          if (!pinchActive)
          {
            if (rect.inside(center))
              dragging = true;
            pinchActive = true;
          }
          if (dragging)
            rect.setFromCenter(center, rect.getWidth(), rect.getHeight());
        }
        else
        {
          if (pinchActive)
          {
            dragging = false;
            pinchActive = false;
          }
        }
      }
      else
      {
        contourPts.clear();
        hullPts.clear();
        tipsPts.clear();
        if (shadowHandBodyCreated)
        {
          shadowHandBody.reset();
          shadowHandBodyCreated = false;
        }
      }
    }
    else
    {
      contourPts.clear();
      hullPts.clear();
      tipsPts.clear();
      if (shadowHandBodyCreated)
      {
        shadowHandBody.reset();
        shadowHandBodyCreated = false;
      }
    }
  }

  // 共有矩形を最新位置で更新
  sharedRect = rect;

  box2d.update();
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
  if (gesture == "pinch")
  {
    ofSetColor(255, 255, 0);
    ofDrawCircle(center, 8);
  }

  // draw detected contour
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
  for (auto &p : tipsPts)
  {
    ofSetColor(255, 0, 0);
    ofDrawCircle(p, 6);
  }

  for (auto &box : boxes)
  {
    ofSetColor(255, 0, 0);
    box->draw();
  }

  if (shadowHandBody)
  {
    ofSetColor(0, 255, 255, 100);
    shadowHandBody->draw();
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

std::pair<std::string, ofPoint> ofApp::detectPinch(const std::vector<cv::Point> &contour)
{
  cv::Moments M = cv::moments(contour);
  int cx = 0, cy = 0;
  if (M.m00 != 0)
  {
    cx = int(M.m10 / M.m00);
    cy = int(M.m01 / M.m00);
  }
  std::vector<int> hullIdx;
  cv::convexHull(contour, hullIdx, false, false);
  if (hullIdx.size() < 3)
    return {"fist", {float(cx), float(cy)}};
  std::sort(hullIdx.begin(), hullIdx.end());
  std::vector<cv::Vec4i> defects;
  try
  {
    cv::convexityDefects(contour, hullIdx, defects);
  }
  catch (...)
  {
    return {"fist", {float(cx), float(cy)}};
  }
  return {"fist", {float(cx), float(cy)}};
}

void ofApp::createShadowHandBody(const std::vector<cv::Point> &contour)
{
  if (contour.size() < 3)
    return;

  std::vector<ofPoint> simplifiedContour;
  for (const auto &pt : contour)
  {
    simplifiedContour.emplace_back(pt.x, pt.y);
  }

  shadowHandBody = std::make_shared<ofxBox2dPolygon>();
  shadowHandBody->setPhysics(1.0, 0.3, 0.1);
  shadowHandBody->setup(box2d.getWorld(), simplifiedContour);
  shadowHandBodyCreated = true;
}

void ofApp::updateShadowHandBody(const std::vector<cv::Point> &contour)
{
  {
    if (!shadowHandBody || contour.size() < 3)
      return;

    std::vector<ofPoint> simplifiedContour;
    for (const auto &pt : contour)
    {
      simplifiedContour.emplace_back(pt.x, pt.y);
    }

    shadowHandBody->updatePolygon(simplifiedContour);
  }
  if (defects.empty())
    return {"fist", {float(cx), float(cy)}};
  std::vector<cv::Point> tips;
  for (auto &d : defects)
  {
    if (d[3] / 256 < 10)
      continue;
    tips.push_back(contour[d[0]]);
    tips.push_back(contour[d[1]]);
  }
  std::vector<cv::Point> uniqueTips;
  for (auto &p : tips)
  {
    if (std::find_if(uniqueTips.begin(), uniqueTips.end(), [&](const auto &q)
                     { return p == q; }) == uniqueTips.end())
      uniqueTips.push_back(p);
  }
  if (uniqueTips.size() >= 2)
  {
    cv::Point p1 = uniqueTips[0];
    cv::Point p2 = uniqueTips[1];
    double dist = cv::norm(p1 - p2);
    ofPoint c((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
    hullPts.clear();
    tipsPts.clear();

    std::vector<cv::Point> cvHull;
    cv::convexHull(contour, cvHull);
    for (auto &hp : cvHull)
      hullPts.emplace_back(hp.x, hp.y);

    for (auto &tip : uniqueTips)
      tipsPts.emplace_back(tip.x, tip.y);

    if (dist < pinchDist)
      return {"pinch", c};
    else
      return {"open", c};
  }
  return {"fist", {float(cx), float(cy)}};
}
