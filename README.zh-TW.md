# Splash Nemotron fork 中文說明

這是 [incoai/splash](https://github.com/incoai/splash) 的 fork，在 Splash 上加了
NVIDIA Nemotron 3.5 兩顆模型：

- **Bolt**（74B-A9B）：自動下載 NVIDIA 的 LiLiCorr 草稿模型，支援看圖。
- **Super VL**（120B-A12B）：要自己下載 target、草稿與 mmproj，支援看圖。
- 兩顆都能從 SSD 串流 expert（記憶體不夠放全部 expert 時）。
- 兩顆都能看影片，並照官方做法修剪影片 token（EVS）。

其餘功能和上游相同；上游的使用說明見 [README.md](README.md)。目前基於上游 10-31 的
`main`（1.5.0 之後），**還沒合併上游 1.6.0**。上游的桌面 App 和 Homebrew 版不含這些
改動，要從原始碼編譯。

## 分支

| 分支 | 內容 |
| --- | --- |
| `nemotron`（預設） | 全部改動＋README |
| `main` | 只追上游，不改 |
| `nemotron-3.5-bolt` | Bolt 與草稿、看圖（[incoai/splash#633](https://github.com/incoai/splash/issues/633)） |
| `bolt-streaming` | 在前一個之上：SSD 串流 expert |
| `bench-tight-memory` | 在前一個之上：`backend-benchmark` 小記憶體暖機 |
| `super-vl` | 在前一個之上：Super VL、官方 NVFP4 GGUF（NVFP4G64） |
| `nemotron-video` | 在前一個之上：影片輸入與 EVS 修剪 |

## 編譯與啟動

需要 macOS 26.4 以上、Xcode 26 以上（含 Metal Toolchain）、Python 3.12–3.14。

```bash
git clone -b nemotron https://github.com/ansoncat/splash-nemotron.git
cd splash-nemotron
make -j4
./splash serve --model nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF
```

第一次啟動會裝 Python 套件、下載模型與草稿。`make` 若說缺 Metal Toolchain，執行
`xcodebuild -downloadComponent MetalToolchain`。Super 的下載與啟動指令見
[README 的 Nemotron 3.5 段](README.md#nemotron-35-this-fork)。

## 影片

- 送法：訊息裡放 `video_url` 或 `input_video`，內容是 base64 的 MP4／QuickTime data URL。
- 取樣：每秒 1 幀、最多 128 幀（`--video-fps`、`--video-max-frames`）。
- 修剪：預設照官方剪掉 70% 的影片 token（`--video-pruning 0.7`；`0` 不剪）。
  2 分鐘的片約 6.2K token，不剪是 16.8K。token 數與官方 processor 相同。
- 限制：單一影片最大 256 MiB、解碼最多 120 秒；只解取樣幀所在的 GOP，30–60 分鐘的片
  約 20 秒。超過約 96 MiB 的影片 base64 後會超過預設請求上限，要加
  `--max-request-size 1G`。軟體解碼、不看旋轉、不收 WebM、沒有音訊。

## 實測（M5 Pro 20 核 GPU、64 GB，維護者自測，不是上游數字）

Bolt decode 對 llama.cpp（greedy、thinking 關、同場次交錯、關 prefix cache）：

| 提示 | llama.cpp | Splash | 倍數 |
| --- | ---: | ---: | ---: |
| 程式碼 | 45–46 | 115–120 | 2.5–2.6× |
| 英文散文 | 34–40 | 62–68 | 1.6–2.0× |
| 繁中散文 | 30–32 | 49–55 | 1.6–1.9× |

影片修剪 `0` 對 `0.7`（Bolt、greedy）：

| 題組 | thinking | 題數 | 不剪 | 剪 0.7 | prompt token |
| --- | --- | ---: | ---: | ---: | --- |
| MVBench 三任務（3–20 秒） | 關 | 600 | 58.0% | 57.7% | −57% |
| Video-MME long（30–60 分鐘） | 開 | 102 | 41.2% | 45.1% | −61% |

兩組差別都不顯著（McNemar p 0.89、0.45）。Video-MME 有三分之一題在 16,384 token 內
還沒想完，算錯；也只用了 256 MiB 以下的片。

其他數字（Super、expert 串流、各量化版）見 [README 的 Performance 段](README.md#nemotron-35-on-this-fork)。
