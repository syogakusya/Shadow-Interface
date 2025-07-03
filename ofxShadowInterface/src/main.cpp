#include "ofMain.h"
#include "ofApp.h"
#include "DisplayApp.h"
#include "SharedData.h"

// 共有矩形の実体定義
ofRectangle sharedRect;

//========================================================================
int main()
{
	// GUI ウィンドウ（カメラ + GUI）
	ofGLFWWindowSettings guiSettings;
	guiSettings.setSize(800, 600);
	guiSettings.setPosition(glm::vec2(50, 50));
	guiSettings.resizable = true;
	auto guiWindow = ofCreateWindow(guiSettings);

	// 表現ウィンドウ（黒背景 + 赤四角形）
	ofGLFWWindowSettings displaySettings;
	displaySettings.setSize(1024, 768);
	displaySettings.setPosition(glm::vec2(900, 50));
	displaySettings.resizable = true;
	displaySettings.shareContextWith = guiWindow; // OpenGL コンテキスト共有
	auto displayWindow = ofCreateWindow(displaySettings);

	// アプリケーションのインスタンス
	auto guiApp = std::make_shared<ofApp>();
	auto displayApp = std::make_shared<DisplayApp>();

	// それぞれのウィンドウにアプリを割り当てる
	ofRunApp(guiWindow, guiApp);
	ofRunApp(displayWindow, displayApp);

	cout << "b, r, ESC : ブラックレスポンス, リセット, 終了" << endl;

	ofRunMainLoop();
}
