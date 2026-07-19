# overhead カメラ予測器による動的 pick

推論レイテンシ（PE gap）分だけ cube を時間前進させた観測を policy に渡すことで、「フレームを撮った瞬間の cube 位置」ではなく「実行される瞬間の cube 位置」を狙う仕組み。RTC の `--predict-cube` として `pixi run eval` / `pixi run sim-eval` から使える（[cli/README.md](https://github.com/Octpus-VLA/reactive-vla/blob/main/cli/README.md#推論)、[docs/rtc-sim-rollout.md](rtc-sim-rollout.md)）。

ここで使う Tier 1/2/3 は LeRobot の API 名ではなく、プロジェクト内の設計用語。この予測器は Tier 3（cube の位置・速度から動的に狙い先を決める予測的リプラン）にあたる。

---

## 時間前進の実装: `image_shift` / `latent_warp` / `latent_flow`

`predictor`(`third_party/lerobot/src/lerobot/predictors/`)は、推論レイテンシ(PE gap)分だけ cube を**進めた観測**を policy に渡すことで「実行時刻の cube 位置を狙う」迎撃を担う。**速度をどう推定し、どこで前進させるか**を `PredictorConfig.mode` で選ぶ:

| mode | 速度推定 | どこで進めるか | 実装 | submodule |
|---|---|---|---|---|
| `image_shift`(デフォルト) | 色追跡(HSV 重心)の**単一速度** | **RGB ピクセル**上で cube を切り貼り。policy が再エンコード。 | `predictors/shift.py` (`shift_cube_in_frame`) | 不要 |
| `latent_warp` | 色追跡の**単一速度** | **vision パッチトークン**上で cube トークンを**剛体並進**。 | `predictors/latent_warp.py` (`warp_token_grid`) | 不要 |
| `latent_flow` | **dense optical flow** の**per-patch 速度** | パッチトークンを**各々の flow で前進**(grid_sample)。色判定なし・剛体仮定なし。 | `predictors/optical_flow.py` + `latent_warp.py` (`warp_token_grid_by_flow`) | 不要 |

このうち `latent_warp` / `latent_flow` の2つは、近年の潜在世界モデルが示す「ピクセルより**凍結エンコーダの特徴空間**で時間を進める方が学習しやすく頑健」という流れに沿う（末尾の参考文献を参照）。`image_shift` はその対照となる素朴な実装で、**RGB ピクセル**上で直接 cube を切り貼りする。

- `latent_warp` は `shift_cube_in_frame` の潜在双子。**色マスクで cube を見つけ、重心速度 1 個で丸ごと平行移動**する軽量版。コンベアの cube がほぼ剛体並進ならこれで足りる。
- `latent_flow` は **色を一切使わず、連続フレーム間の dense optical flow から per-patch 速度を推定**し、各トークンを自分の flow で前進させる(`grid_sample` の backward warp)。flow バックエンドは差し替え可能で、同じ `estimate` 契約を実装すれば別のモデルも挿せる。

### dense flow の per-patch 前進(`latent_flow`)

`latent_flow` の流れ: 連続フレームを `DenseFlowEstimator`(OpenCV DIS、状態保持)に通して **dense flow 場 `(H,W,2)`** を得る → lead 時間ぶんに換算(`flow · lead_s/dt`)→ パッチグリッドに平均プールして **per-patch 変位**にする → `warp_token_grid_by_flow` が `grid_sample` の backward sample(target `q` を `q − flow[q]` からサンプル)で各トークンを自分の flow で前進。`flow_motion_threshold` を超えるパッチだけ warp し、静止領域(背景・待機アーム)はそのまま残す。

### 注入点(SmolVLA)

両 latent モードは SmolVLA のエンコード経路に**デフォルト無効**のフックを 1 つ足して実現している。

| 変更点 | ファイル | 内容 |
|---|---|---|
| トークン warp の核 | `predictors/latent_warp.py` | `warp_token_grid`(剛体: cube トークンを `offset_tokens` 並進＋背景 median 埋め＋OOB drop)と `warp_token_grid_by_flow`(dense: per-patch flow を `grid_sample` で前進＋motion mask)。 |
| dense flow 推定 | `predictors/optical_flow.py` (`DenseFlowEstimator`) | 連続フレームの OpenCV DIS/Farneback flow。初回フレームは `None`。エピソード毎に reset。 |
| エンコーダ内フック | `policies/smolvla/smolvlm_with_expert.py` (`embed_image`) | vision encoder 出力(connector の **pixel-shuffle 前**、row-major `H·W` の SigLIP 特徴グリッド)に `warp_fn` を任意適用。`warp_fn=None` で従来どおり。 |
| カメラ別 warp の配線 | `policies/smolvla/modeling_smolvla.py` (`embed_prefix` / `SmolVLAPolicy.set_latent_warp`) | present 画像特徴の順に warp 関数リストを割り当て。`set_latent_warp` は context manager で 1 推論の間だけ設定し、終了時に必ず復元(リークなし)。 |
| RTC 配線 | `rollout/inference/rtc.py` (`_time_advanced_obs` / `_latent_warp_context` / `_build_latent_warp_fns`) | image_shift はピクセル編集、latent は `predict_action_chunk` を `set_latent_warp(...)` で包む。predictor/flow は 1 ループ 1 回だけ実行。マスク/offset/flow 場は `resize_imgs_with_padding` を通してエンコーダ入力空間へ射影し、パッチグリッドと整合させる。 |
| 設定キー | `predictors/config.py` | `mode`(`image_shift`/`latent_warp`/`latent_flow`)・`latent_mask_threshold`・`flow_algorithm`(`dis`/`farneback`)・`flow_motion_threshold`。**デフォルトは `image_shift`** で既存挙動不変。 |

connector の pixel-shuffle **前**の素の SigLIP グリッド(row-major)で warp するため、トークン再サンプリングの内部仕様に依存しない。有効化は例えば `--inference.predictor.enabled=true --inference.predictor.mode=latent_flow --inference.predictor.camera=overall`。

> 注: warp は `grid_sample`/`nonzero` などデータ依存の制御を含むため `torch.compile` 下では graph break しうる(eager 運用想定)。predictor/flow 自体が numpy/OpenCV ベースで元々非 compile 経路。

---

## 物理的な限界

時間前進で観測を補正しても、**アーム最大速度で把持作業域に届かないベルト速度**では取れない。可達域のベルト方向長さ `L`、ベルト速 `s`、アーム移動時間 `Δt_arm` として、取れる上限は概ね `s_max ≈ L / Δt_arm`。`s > s_max` は予測をどう改善しても不可能で、対策はロボットをベルトに寄せる／迎撃点を上流に取る／アームを速くする側になる。

---

## 参考

- Black, Galliker, Levine. *Real-Time Chunking (RTC)*, 2025. https://arxiv.org/abs/2506.07339 — チャンク境界の滑らかさを担保。反応性は別レイヤ。
- Liu ほか. *Bidirectional Decoding (BID)*, ICLR 2025. https://arxiv.org/abs/2408.17355 — チャンク実行中は直近観測への反応が落ちる(動く標的で失敗)。
- Islam ほか. *Constant-Time Replanning off a Conveyor*, RSS 2020. https://arxiv.org/abs/2101.07148 — 「確信を待つ/一発計画/逐次 replan」で 34.6/16/69.2%。早く動き出し安く replan。
- Akinola ほか. *Dynamic Grasping with Reachability and Motion Awareness*, IROS 2021. https://arxiv.org/abs/2103.10562 — eye-to-hand 外部カメラで予測、終端は近接カメラ、という役割分担。
- *Intercepting the Future (AHEAD)*, 2026. https://arxiv.org/abs/2606.02486 — frozen VLA の特徴空間で、optical flow 由来の per-token 速度/加速度に条件付けして未来パッチトークンを予測(flow-matching latent dynamics + adaptive horizon)。`latent_warp` の学習版の参照先。
- Hu ほか. *LaDi-WM: Latent Diffusion World Model*, 2025. https://arxiv.org/abs/2505.11528 — DINO 幾何 + CLIP 意味の特徴空間で latent diffusion 予測。「ピクセルより特徴空間で進める方が学習しやすい」を示す。AHEAD の前身。
- Zhou ほか. *DINO-WM: Zero-Shot Latent World Model*, 2024. https://arxiv.org/abs/2411.04983 — 凍結 DINOv2 パッチ特徴上の latent dynamics(reconstruction 不要)。「frozen encoder 特徴で時間を進める」原型。
