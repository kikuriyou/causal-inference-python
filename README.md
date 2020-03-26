# 環境構築
- Python用の統計的因果推論モジュールをいじるために必要な環境構築の説明です。
- Dockerクライアントをインストール済みであれば、あとはこの後に説明する手順を追っていけば計算環境に入れます。
- 未インストールの場合は、使用OSに応じたDockerクライアントをインストールしてください。
    - [Get started with Docker Desktop for Mac](https://docs.docker.com/docker-for-mac/)
    - [Get started with Docker Desktop for Windows](https://docs.docker.com/docker-for-windows/)
    - [Get Docker Engine - Community for Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

- 以下、実行手順です。

    `docker.sh` があるディレクトリに移動します。
    ```
    cd docker/
    ```

    Dockerイメージをビルドします。初めてのビルドだと時間がかかります。
    ```
    ./docker.sh build
    ```

    Dockerコンテナを起動します。port番号はここでは `8888` にしています。
    ```
    ./docker.sh start 8888
    ```
    GPUがない場合は以下のようなメッセージが出ますが問題ありません。
    ```
    ./docker.sh: line 56: lspci: command not found
    There is not any gpu.
    ```

    Dockerコンテナに入ります。
    ```
    ./docker.sh enter
    ```
    以下のように表示されていればOKです。
    ```
    root@d77a7cb17cbc:~/user/project# 
    ```
    入ったところのディレクトリに `run_jupyter.sh` というファイルがあるので実行します。先ほど `./docker.sh start [port-num]` で指定したport番号と同じ番号を入力します。
    ```
    ./run_jupyter.sh 8888
    ```
    以下のような表示が出るので、`token=xxxxxx` の xxxxxx の部分をコピーします。この部分は Jupyter Lab にアクセスするためのパスワードになります。
    ```
    To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-19-open.html
    Or copy and paste one of these URLs:
        http://d77a7cb17cbc:8888/?token=e94c6e58980b6d93a207927ef6d4294d778d8e82759925c9
     or http://127.0.0.1:8888/?token=e94c6e58980b6d93a207927ef6d4294d778d8e82759925c9
    ```
    Jupyter Lab を開くためにブラウザで `localhost:8888` にアクセスします。初回アクセス時はパスワードが求められるので先ほどコピーしたパスワードを入力してログインします。

    Jupyter Lab の画面の左側にディレクトリが表示されているので、`scripts` に入ります。
    いくつかのスクリプトや notebook があるので、ここでは `sample_tvcm.ipynb` を開きます。
    あとは自由に notebook をいじってください。










## 


