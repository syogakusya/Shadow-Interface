# ShadowInterface

手の影を用いて画面上のオブジェクトを "つまむ / 掴む" 操作ができるデモアプリです。

## セットアップ

1. Python 3.8 以降をインストール
2. 仮想環境を推奨（例: `python -m venv venv` & `venv\Scripts\activate`）
3. 依存ライブラリをインストール

```bash
pip install -r requirements.txt
```

## 使い方

```bash
python main.py
```

1. 起動後、背景に手を入れず `b` キーを押して背景をキャプチャしてください。
2. 影が検出され始めると緑色の輪郭が表示されます。
3. 親指と人差し指で "つまむ" ジェスチャ（2 点間が狭い）をすると赤い四角形を掴めます。
4. 指を離すとドロップします。
5. `Esc` キーで終了、`r` キーで背景を再キャプチャ。

## コード概要

- `main.py` : 画像処理とジェスチャ判定、オブジェクト操作のメインスクリプト

## 参考

- OpenCV convexityDefects による指先検出
- 背景差分で影領域を抽出
