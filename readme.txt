pythonファイルの説明

\agent:基礎的なエージェントについて記述
\Diffusion phenomena:拡散方程式（使わない）
\Regression:回帰問題（使用）
    \Logistic:ロジスティック回帰(PowerPointでの動画用)
    \Non_strongly:練習用（使用しない）
    \ronbun:TAC2018用シミュレーション（各フォルダが数値実験に対応）　
    \shuron:修士論文用シミュレーション（各フォルダが数値実験に対応）
    \sq_mean:練習用(使用しない)
\jikken_~~~~:練習用(test)

共通事項
main.py:実行ファイル (これをrunしてください)
    n:エージェント数
    m:状態の次元
make_communication:通信グラフ作成


--------------------------------------------------------------
1. anacondaをインストール
https://qiita.com/suJiJi/items/665f24b823506df0b5ab

2. pycharm や spyderなどのIDEをインストール

2. python3.7をインストール
conda create -n py37 python=3.7 anaconda
conda info -e

3．python3.7の仮想環境に入る
activate py37

4. 各種パッケージをインストール

conda install ***

***に以下のもの

numpy
sympy
matplotlib
networkx
progressbar2
pandas

scikit-learn    <-- 機械学習データセット用
-c conda-forge ffmpeg <-- 動画作成用

5. cvxpyをインストール
conda install -c cvxgrp cvxpy

----------
初回実行時にpythonのversionを指定

左上タブのFile
-->Settings
-->Project Interpreterの欄右端の歯車ボタン
-->Add
-->左端のConda Environmentを選択
-->Existing environment
-->C:\Users\na-ha\Anaconda3\envs\py37\python.exe
Make available to all projectsにチェックを入れる

