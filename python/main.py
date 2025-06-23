import cv2
import numpy as np

# ------------------------------
# パラメータ (初期値)
# ------------------------------
THRESH = 40           # 背景差分の二値化しきい値
PINCH_DISTANCE = 40   # ピンチ判定する指先間距離(px)
MAX_PINCH = 200       # トラックバーの上限値

# ------------------------------
# トラックバー用コールバック (何もしない)
# ------------------------------

def _nothing(x):
    pass

# ------------------------------
# ユーティリティ
# ------------------------------

def detect_pinch(contour: np.ndarray) -> tuple[str, tuple[int, int]]:
    """輪郭からジェスチャ( 'pinch' / 'open' / 'fist') と指先重心を返す
    convexityDefects がエラーを吐く場合（自己交差輪郭など）は 'fist' を返す.
    """
    # 輪郭の中心を計算（フォールバック用）
    M = cv2.moments(contour)
    cx, cy = (0, 0)
    if M['m00']:
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])

    hull_ids = cv2.convexHull(contour, returnPoints=False)
    if hull_ids is None or len(hull_ids) < 3:
        return 'fist', (cx, cy)

    # OpenCV のバグ回避: hull インデックスが昇順でないと例外が出ることがある
    hull_ids = hull_ids.squeeze()
    hull_ids = np.sort(hull_ids)

    try:
        defects = cv2.convexityDefects(contour, hull_ids)
    except cv2.error:
        # 自己交差などで失敗したらこぶし扱い
        return 'fist', (cx, cy)

    if defects is None:
        return 'fist', (cx, cy)

    finger_tips = []
    for s, e, f, d in defects[:, 0]:
        if d / 256 < 10:  # 深さが浅い凸凹は無視
            continue
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        finger_tips.extend([start, end])

    # 重複除去
    finger_tips = list({(x, y) for (x, y) in finger_tips})

    # 指先が2本見えているか？
    if len(finger_tips) >= 2:
        # 2本に限定
        finger_tips = finger_tips[:2]
        p1, p2 = finger_tips
        dist = np.linalg.norm(np.subtract(p1, p2))
        center = tuple(((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2))
        if dist < PINCH_DISTANCE:
            return 'pinch', center
        else:
            return 'open', center

    return 'fist', (cx, cy)


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


# ------------------------------
# メインループ
# ------------------------------

def main():
    global THRESH, PINCH_DISTANCE  # トラックバーで書き換える

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: カメラを開けません")
        return

    # ウィンドウ・トラックバー設定
    cv2.namedWindow('ShadowInterface')
    cv2.createTrackbar('Thresh', 'ShadowInterface', THRESH, 255, _nothing)
    cv2.createTrackbar('PinchDist', 'ShadowInterface', PINCH_DISTANCE, MAX_PINCH, _nothing)

    bg_gray = None  # 背景画像
    obj = DraggableRect()
    pinch_active = False

    print("[INFO] 'b' キー: 背景キャプチャ / 'r': リセット / ESC: 終了")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # ミラー表示
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            bg_gray = gray.copy()
            print("[INFO] 背景を保存しました")
        elif key == ord('r'):
            bg_gray = None
            obj.stop_drag()
            pinch_active = False
            print("[INFO] リセットしました")
        elif key == 27:  # ESC
            break

        if bg_gray is not None:
            # トラックバー値を取得してパラメータ更新
            THRESH = cv2.getTrackbarPos('Thresh', 'ShadowInterface')
            if THRESH < 1:
                THRESH = 1  # 0 にすると全白になりやすい
            PINCH_DISTANCE = cv2.getTrackbarPos('PinchDist', 'ShadowInterface')
            if PINCH_DISTANCE < 1:
                PINCH_DISTANCE = 1

            diff = cv2.absdiff(gray, bg_gray)
            _, mask = cv2.threshold(diff, THRESH, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, None, iterations=2)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                contour = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                if area > 3000:  # ノイズ除去
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                    gesture, center = detect_pinch(contour)

                    if gesture == 'pinch':
                        if not pinch_active:
                            # ピンチ開始
                            if obj.contains(center):
                                obj.start_drag()
                            pinch_active = True
                        obj.update(center)
                        cv2.circle(frame, center, 8, (0, 255, 255), -1)
                    else:
                        if pinch_active:
                            # ピンチ終了
                            obj.stop_drag()
                            pinch_active = False

        obj.draw(frame)
        cv2.imshow('ShadowInterface', frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 