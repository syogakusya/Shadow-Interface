#include "GestureDetector.h"
#include <algorithm>

GestureDetector::GestureDetector() : pinchDist(150), minPinchDist(0) {}

void GestureDetector::setPinchDistance(int maxDist, int minDist)
{
  pinchDist = maxDist;
  minPinchDist = minDist;
}

std::vector<std::pair<std::string, ofPoint>> GestureDetector::detect(const std::vector<cv::Point> &contour)
{
  contourPts.clear();
  for (auto &pt : contour)
    contourPts.emplace_back(pt.x, pt.y);

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
    return {{"fist", {float(cx), float(cy)}}};

  std::sort(hullIdx.begin(), hullIdx.end());
  std::vector<cv::Vec4i> defects;
  try
  {
    cv::convexityDefects(contour, hullIdx, defects);
  }
  catch (...)
  {
    return {{"fist", {float(cx), float(cy)}}};
  }
  if (defects.empty())
    return {{"fist", {float(cx), float(cy)}}};

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

  // 描画用データ更新
  hullPts.clear();
  tipsPts.clear();
  std::vector<cv::Point> cvHull;
  cv::convexHull(contour, cvHull);
  for (auto &hp : cvHull)
    hullPts.emplace_back(hp.x, hp.y);
  for (auto &tip : uniqueTips)
    tipsPts.emplace_back(tip.x, tip.y);

  // 複数ピンチ対応
  std::vector<std::pair<std::string, ofPoint>> result;
  std::vector<bool> used(uniqueTips.size(), false);
  for (size_t i = 0; i < uniqueTips.size(); ++i)
  {
    for (size_t j = i + 1; j < uniqueTips.size(); ++j)
    {
      double dist = cv::norm(uniqueTips[i] - uniqueTips[j]);
      if (dist >= minPinchDist && dist < pinchDist)
      {
        ofPoint c((uniqueTips[i].x + uniqueTips[j].x) / 2, (uniqueTips[i].y + uniqueTips[j].y) / 2);
        result.push_back({"pinch", c});
        used[i] = used[j] = true;
      }
    }
  }

  // ピンチ以外の指先は open として返す
  for (size_t i = 0; i < uniqueTips.size(); ++i)
  {
    if (!used[i])
    {
      result.push_back({"open", ofPoint(uniqueTips[i].x, uniqueTips[i].y)});
    }
  }

  if (result.empty())
  {
    result.push_back({"fist", {float(cx), float(cy)}});
  }

  for (auto &r : result)
  {
    ofLog() << r.first << " " << r.second << endl;
  }

  return result;
}

void GestureDetector::clear()
{
  contourPts.clear();
  hullPts.clear();
  tipsPts.clear();
}