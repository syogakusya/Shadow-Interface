import cv2
import numpy as np

# ------------------------------
# パラメータ (初期値)
# ------------------------------
THRESH = 40           # 背景差分の二値化しきい値
PINCH_DISTANCE = 100   # ピンチ判定する指先間距離(px)
MAX_PINCH = 400       # トラックバーの上限値

# ------------------------------
# トラックバー用コールバック (何もしない)
# ------------------------------

def _nothing(x):
    pass

# ------------------------------
# ユーティリティ
# ------------------------------

def detect_pinch(contour: np.ndarray) -> tuple[str, tuple[int, int], dict]:
    """輪郭からジェスチャ( 'pinch' / 'open' / 'fist') と指先重心を返す
    convexityDefects がエラーを吐く場合（自己交差輪郭など）は 'fist' を返す.
    """
    # 輪郭の近似処理で精度向上
    epsilon = 0.02 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    
    # 輪郭の中心を計算（フォールバック用）
    M = cv2.moments(contour)
    cx, cy = (0, 0)
    if M['m00']:
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])

    # 可視化情報を格納する辞書
    debug_info = {
        'hull_points': None,
        'defects': None,
        'finger_tips': [],
        'unique_tips': [],
        'valid_defects': []
    }

    hull_ids = cv2.convexHull(contour, returnPoints=False)
    if hull_ids is None or len(hull_ids) < 4:
        return 'fist', (cx, cy), debug_info

    # 凸包の点座標も取得（可視化用）
    hull_points = cv2.convexHull(contour, returnPoints=True)
    debug_info['hull_points'] = hull_points

    # OpenCV のバグ回避: hull インデックスが昇順でないと例外が出ることがある
    hull_ids = hull_ids.squeeze()
    hull_ids = np.sort(hull_ids)

    try:
        defects = cv2.convexityDefects(contour, hull_ids)
    except cv2.error:
        # 自己交差などで失敗したらこぶし扱い
        return 'fist', (cx, cy), debug_info

    if defects is None:
        return 'fist', (cx, cy), debug_info

    debug_info['defects'] = defects

    finger_tips = []
    valid_defects = []
    
    for s, e, f, d in defects[:, 0]:
        depth = d / 256
        if depth < 15:  # 深さの閾値を厳しく
            continue
            
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        
        # 角度チェック：指先らしい角度か判定
        angle = np.degrees(np.arccos(np.clip(
            np.dot(np.subtract(start, far), np.subtract(end, far)) /
            (np.linalg.norm(np.subtract(start, far)) * np.linalg.norm(np.subtract(end, far))),
            -1, 1
        )))
        
        if 30 < angle < 150:  # 指先らしい角度範囲
            valid_defects.append((start, end, far, depth))
            finger_tips.extend([start, end])

    debug_info['finger_tips'] = finger_tips
    debug_info['valid_defects'] = valid_defects

    # 重複除去（距離が近い点をまとめる）
    unique_tips = []
    for tip in finger_tips:
        is_duplicate = False
        for unique_tip in unique_tips:
            if np.linalg.norm(np.subtract(tip, unique_tip)) < 20:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_tips.append(tip)

    debug_info['unique_tips'] = unique_tips

    # 指先が2本見えているか？
    if len(unique_tips) >= 2:
        # 距離が最も近い2本を選択
        min_dist = float('inf')
        best_pair = None
        
        for i in range(len(unique_tips)):
            for j in range(i + 1, len(unique_tips)):
                dist = np.linalg.norm(np.subtract(unique_tips[i], unique_tips[j]))
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (unique_tips[i], unique_tips[j])
        
        if best_pair:
            p1, p2 = best_pair
            center = tuple(((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2))
            if min_dist < PINCH_DISTANCE:
                return 'pinch', center, debug_info
            else:
                return 'open', center, debug_info

    return 'fist', (cx, cy), debug_info


class CalibrationArea:
    def __init__(self):
        self.corners = []  # 4つの角の座標
        self.calibrated = False
        
    def add_corner(self, point):
        if len(self.corners) < 4:
            self.corners.append(point)
            if len(self.corners) == 4:
                self.calibrated = True
                self._sort_corners()
                
    def reset(self):
        self.corners = []
        self.calibrated = False
        
    def _sort_corners(self):
        # 左上、右上、右下、左下の順にソート
        self.corners = sorted(self.corners, key=lambda p: p[1])  # y座標でソート
        top_two = sorted(self.corners[:2], key=lambda p: p[0])   # 上2つをx座標でソート
        bottom_two = sorted(self.corners[2:], key=lambda p: p[0])  # 下2つをx座標でソート
        self.corners = [top_two[0], top_two[1], bottom_two[1], bottom_two[0]]
        
    def normalize_point(self, point):
        if not self.calibrated:
            return None
            
        # 透視変換用の4点を設定
        src_points = np.float32(self.corners)
        dst_points = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        
        # 透視変換行列を計算
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 点を変換
        point_homogeneous = np.array([[[point[0], point[1]]]], dtype=np.float32)
        normalized = cv2.perspectiveTransform(point_homogeneous, matrix)
        
        return (float(normalized[0][0][0]), float(normalized[0][0][1]))
        
    def draw(self, frame, fill=True):
        # キャリブレーション領域を描画
        for i, corner in enumerate(self.corners):
            cv2.circle(frame, corner, 10, (255, 255, 0), -1)
            cv2.putText(frame, str(i+1), (corner[0]+15, corner[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
        if len(self.corners) >= 2:
            # 線を描画
            for i in range(len(self.corners)):
                next_i = (i + 1) % len(self.corners)
                if next_i < len(self.corners):
                    cv2.line(frame, self.corners[i], self.corners[next_i], (255, 255, 0), 2)
                    
        if self.calibrated and fill:
            # キャリブレーション完了時は塗りつぶし
            pts = np.array(self.corners, np.int32)
            cv2.fillPoly(frame, [pts], (255, 255, 0, 50))


class DraggableRect:
    def __init__(self, pos=(200, 200), size=60, color=(0, 0, 255)):
        self.pos = list(pos)  # 中心座標
        self.size = size
        self.color = color
        self.dragging = False

    def contains(self, point):
        x, y = point
        px, py = self.pos
        half = self.size // 2
        return px - half <= x <= px + half and py - half <= y <= py + half

    def start_drag(self):
        self.dragging = True

    def stop_drag(self):
        self.dragging = False

    def update(self, point):
        if self.dragging:
            self.pos[0], self.pos[1] = point

    def draw(self, frame):
        px, py = self.pos
        half = self.size // 2
        cv2.rectangle(frame, (px - half, py - half), (px + half, py + half), self.color, -1)


def create_fullscreen_frame():
    import subprocess, re
    try:
        result = subprocess.run(
            ['system_profiler', 'SPDisplaysDataType'], capture_output=True, text=True, timeout=1
        )
        m = re.search(r'Resolution:\s*(\d+)\s*x\s*(\d+)', result.stdout)
        if m:
            width, height = map(int, m.groups())
            return np.zeros((height, width, 3), dtype=np.uint8)
    except Exception:
        pass
    return np.zeros((1440, 2560, 3), dtype=np.uint8)


# ------------------------------
# メインループ
# ------------------------------

def main():
    global THRESH, PINCH_DISTANCE  # トラックバーで書き換える

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: カメラを開けません")
        return

    # マウスクリック用のグローバル変数
    calibration = CalibrationArea()
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            calibration.add_corner((x, y))

    # デバッグウィンドウ設定
    cv2.namedWindow('Debug', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Debug', mouse_callback)
    cv2.createTrackbar('Thresh', 'Debug', THRESH, 255, _nothing)
    cv2.createTrackbar('PinchDist', 'Debug', PINCH_DISTANCE, MAX_PINCH, _nothing)
    
    # UIウィンドウ設定
    cv2.namedWindow('UI', cv2.WINDOW_NORMAL)

    bg_gray = None  # 背景画像
    obj = DraggableRect()
    pinch_active = False
    ui_fullscreen = False
    normalized_debug = False
    perspective_matrix = None
    
    ui_frame_normal = None
    ui_frame_fullscreen = None

    print("[INFO] 'b': 背景キャプチャ / 'r': リセット / 'c': キャリブレーションリセット / 'f': フルスクリーン切替 / ESC: 終了")


    while True:
        ret, frame = cap.read()
        if not ret:
            break


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            bg_gray = gray.copy()
            print("[INFO] 背景を保存しました")
        elif key == ord('r'):
            bg_gray = None
            obj.stop_drag()
            pinch_active = False
            normalized_debug = False
            perspective_matrix = None
            print("[INFO] リセットしました")
        elif key == ord('c'):
            calibration.reset()
            print("[INFO] キャリブレーションをリセットしました")
        elif key == ord('f'):
            ui_fullscreen = not ui_fullscreen
            if ui_fullscreen:
                cv2.setWindowProperty('UI', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                ui_frame_fullscreen = create_fullscreen_frame()
            else:
                cv2.setWindowProperty('UI', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        elif key == 27:  # ESC
            break

        if bg_gray is not None:
            # トラックバー値を取得してパラメータ更新
            THRESH = cv2.getTrackbarPos('Thresh', 'Debug')
            if THRESH < 1:
                THRESH = 1  # 0 にすると全白になりやすい
            PINCH_DISTANCE = cv2.getTrackbarPos('PinchDist', 'Debug')
            if PINCH_DISTANCE < 1:
                PINCH_DISTANCE = 1

            # ガウシアンブラーで背景差分の精度向上
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            bg_blur = cv2.GaussianBlur(bg_gray, (5, 5), 0)
            
            diff = cv2.absdiff(gray_blur, bg_blur)
            _, mask = cv2.threshold(diff, THRESH, 255, cv2.THRESH_BINARY)
            
            # より強力なノイズ除去
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.medianBlur(mask, 5)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                contour = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                if area > 5000:  # 面積閾値を上げてノイズ除去を強化
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                    gesture, center, debug_info = detect_pinch(contour)
                    print(f"gesture: {gesture}, center: {center}")
                    
                    # 座標正規化
                    normalized_center = calibration.normalize_point(center)
                    if normalized_center:
                        print(f"normalized: ({normalized_center[0]:.3f}, {normalized_center[1]:.3f})")
                        if calibration.calibrated and not normalized_debug:
                            src_points = np.float32(calibration.corners)
                            h, w = frame.shape[:2]
                            dst_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
                            perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                            normalized_debug = True

                    # デバッグ用の描画
                    # 凸包を描画（青色）
                    if debug_info['hull_points'] is not None:
                        cv2.drawContours(frame, [debug_info['hull_points']], -1, (255, 0, 0), 2)

                    # 凸性欠損を描画
                    if debug_info['defects'] is not None and debug_info['valid_defects']:
                        for start, end, far, depth in debug_info['valid_defects']:
                            # 凸性欠損の線を描画（白色）
                            cv2.line(frame, start, end, (255, 255, 255), 2)
                            # 凸性欠損の最深点を描画（赤色）
                            cv2.circle(frame, far, 6, (0, 0, 255), -1)

                    # 検出された指先を描画（緑色）
                    for tip in debug_info['unique_tips']:
                        cv2.circle(frame, tip, 8, (0, 255, 0), -1)

                    # ピンチ操作（正規化座標を使用）
                    if gesture == 'pinch' and calibration.calibrated and normalized_center:
                        # 正規化座標を画面サイズに変換
                        screen_height, screen_width = frame.shape[:2]
                        ui_center = (int(normalized_center[0] * screen_width), 
                                   int(normalized_center[1] * screen_height))
                        
                        if not pinch_active:
                            # ピンチ開始
                            if obj.contains(ui_center):
                                obj.start_drag()
                            pinch_active = True
                        obj.update(ui_center)
                        cv2.circle(frame, center, 8, (0, 255, 255), -1)
                        
                        # 正規化座標を画面に表示
                        norm_text = f"Norm: ({normalized_center[0]:.3f}, {normalized_center[1]:.3f})"
                        cv2.putText(frame, norm_text, (center[0] + 20, center[1] - 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    else:
                        if pinch_active:
                            # ピンチ終了
                            obj.stop_drag()
                            pinch_active = False

        # デバッグウィンドウ用フレーム作成
        if normalized_debug and perspective_matrix is not None:
            # 透視変換後は正規化された領域をカメラサイズで表示
            h, w = frame.shape[:2]
            debug_frame = cv2.warpPerspective(frame, perspective_matrix, (w, h))
        else:
            debug_frame = frame.copy()
            calibration.draw(debug_frame, fill=False)
        
        calib_status = f"Calibration: {len(calibration.corners)}/4 corners"
        if calibration.calibrated:
            calib_status += " - READY"
        cv2.putText(debug_frame, calib_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        if ui_fullscreen:
            # フルスクリーン時は実際のウィンドウサイズを取得してバッファを調整
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect('UI')
            except AttributeError:
                # OpenCV < 4.5.1 などで未対応の場合はフォールバック
                win_h, win_w = frame.shape[:2]

            if ui_frame_fullscreen is None or ui_frame_fullscreen.shape[0] != win_h or ui_frame_fullscreen.shape[1] != win_w:
                ui_frame_fullscreen = np.zeros((win_h, win_w, 3), dtype=np.uint8)

            ui_frame = ui_frame_fullscreen
        else:
            if ui_frame_normal is None or ui_frame_normal.shape[:2] != frame.shape[:2]:
                ui_frame_normal = np.zeros_like(frame)
            ui_frame = ui_frame_normal
        ui_frame.fill(0)
        
        obj.draw(ui_frame)
        if pinch_active and calibration.calibrated:
            cv2.circle(ui_frame, obj.pos, 15, (0, 255, 255), 3)
        cv2.imshow('Debug', debug_frame)
        cv2.imshow('UI', ui_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()