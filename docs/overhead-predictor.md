# overhead カメラを「いつ取りに行くか」のゲートに使う

ベルトコンベア上を動く cube を SO-101 で pick するとき、上方の `overall`(overhead / eye-to-hand)カメラを **「cube がいつアームの前に来るか」を判断するゲート** として使うための設計メモです。`supervisor-trigger.md` の Tier 2/3 と地続きですが、ここでは **学習不要(training-free)** で効く部分だけに絞ります。

ここで使う Tier 1/2/3 は LeRobot の API 名ではなく、プロジェクト内の設計用語です。

---

## 1. 狙い(この2点だけ)

1. **学習時**: cube がまだアームの前(把持作業域)に無いフレームに `pick` 指示を当てて学習させるのをやめる。→ policy を「cube が見えている → 掴む」だけに絞る。
2. **推論時**: cube が見えていない/まだ来ていないのに policy を実行してアームを動かすのを避ける。→ overhead が「来た」と言うまでアームは待機。

どちらも **新しいネットワーク・追加学習・条件付けトークンは不要**。学習側はデータのフィルタ、推論側は実行ゲート 1 個で済みます。

> 迎撃点を狙う(predict-and-intercept)・予測ベクトルを policy に条件付けする、といった「**どこを狙うか(WHERE)**」の話は学習を伴うので §6 の将来課題に分離した。本メモは「**いつ動くか(WHEN)**」だけを扱う。

---

## 2. いまの問題

- detector(`red_cube_speed.py`)は **「いつ replan するか」しか変えていない**。`replan_now` / `effective_chunk_size_threshold` は observation を送り直すタイミングのノブで、**アームの実行を止める仕組みは無い**(`base.py:38` の `DetectorOutput`)。
- policy(SmolVLA)はテレオペデータをそのまま BC 学習している。動データには **cube がアーム(front/wrist)カメラに映っていない待機フレーム** が含まれる。テレオペ中はそこでアームをほぼ動かしていないので、policy は粗く hold を覚えてはいるが、
  - **同じ「空フレーム」に hold と動き出しの両方のラベルが付く**(いつ来るかは wrist 画像に写っていない)→ BC が平均化して動き出しが鈍る・タイミングがぶれる。
  - 「アーム前に無いのに `pick` 指示で学習」は、観測に手がかりが無いまま行動を教える格好になり、上の曖昧さの温床になる。

→ ②は **実行ゲート**で、①は **学習データのフィルタ**で素直に消せる。

---

## 3. 方針: overhead を実行ゲートにする(training-free)

### 3.1 推論時 — WAIT → ENGAGE の実行ゲート

```text
  overhead frame ──► detector: cube は把持作業域(engage 線の内側)にいるか?
                        │
            いない ─────┤──► WAIT: アームは ready/home pose を保持(policy を実行しない)
                        │
            いる ───────┴──► ENGAGE: 既存 policy をそのまま実行(RTC で chunk 補充)
```

- WAIT 中は policy の action を **dispatch せず**、ready pose を保持する hold コマンドだけ送る。
- ENGAGE に入ったら何も変えず既存の RTC ループに渡す。**policy 側は無改造**。
- ゲートは overhead detector の出力だけで決まる(`red_cube_not_visible` か、`center_px` が engage 線を越えたか)。

### 3.2 学習時 — 待機フレームを落として train=test を揃える

- データセットを **「cube がアームカメラに写っている(= 把持作業域にいる)フレーム」だけ** に絞って学習する。空・待機フレームは action loss から除外。
- こうすると **学習も推論も「cube が見えている」状態しか policy に通らない**ので分布が一致し、§2 の曖昧さ(空フレームのマルチモーダル・ラベル)が消える。待機の hold は policy ではなく §3.1 のゲートが担当する。
- 完全に捨てるのが不安なら **強くダウンサンプル**でもよい(待機を少し残すと急な再ゲート時に滑らか)。比率はアブレーションで決める。

---

## 4. 「アーム前にいる」をどう判定するか

overhead 画像上に **engage 線(把持作業域の上流境界)を一度キャリブレーション**し、

- `red_cube_not_visible`(`red_cube_speed.py:68`)→ 当然 WAIT。
- `center_px` が engage 線の内側 → ENGAGE。
- (任意)既存の `speed_px_s` から **少しだけ先読み**して engage 線の手前 `v·Δt_lead` で解放すると、アーム始動の遅れ(`Δt_total`)を相殺できる。`Δt_lead` は小さめの固定値から始め、過大だと空振りするので clip する。

学習側のフィルタも **同じ engage 線**で「アームカメラに cube が写っている/作業域にいる」を定義すれば、§3.1 と §3.2 の判定が一致する。

---

## 5. コードへの対応づけ(最小)

| 変更点 | ファイル | 内容 |
|---|---|---|
| ゲート出力を追加 | `third_party/lerobot/src/lerobot/detectors/base.py` | `DetectorOutput` に `in_engage_zone: bool`(または `should_engage`)を追加。既存フィールドは維持。 |
| engage 判定 | `third_party/lerobot/src/lerobot/detectors/red_cube_speed.py` | `red_cube_not_visible` / `center_px` vs engage 線で `in_engage_zone` を決める。任意で `speed_px_s` による先読み。 |
| 推論ゲート(RTC) | `third_party/lerobot/src/lerobot/rollout/inference/rtc.py` の実行ループ(`_evaluate_detector` 付近・queue 補充の手前) | `in_engage_zone` が False の間は policy を実行せず ready pose を保持。True で既存経路に復帰。 |
| 推論ゲート(async) | `third_party/lerobot/src/lerobot/async_inference/robot_client.py` | 同上を supervisor 経路にも適用。 |
| 学習フィルタ | データ前処理スクリプト(親リポジトリ側) | 同じ engage 線で「作業域外/cube 不可視」フレームを除外 or ダウンサンプルしてから学習。 |
| 設定キー | `detectors/config.py` | engage 線・先読み `Δt_lead` 上限・フィルタ比率。**デフォルトは従来挙動(ゲート無効)**。 |

> CLAUDE.md の方針に従い、LeRobot 内部の変更は submodule(`third_party/lerobot`)内でコミットしてから親の submodule ポインタを更新する。ゲート/フィルタはデフォルト無効で既存挙動を壊さない。

---

## 5.5 時間前進の実装: `image_shift` / `latent_warp` / `latent_flow`(WHERE 側)

`predictor`(`third_party/lerobot/src/lerobot/predictors/`)は、推論レイテンシ(PE gap)分だけ cube を**進めた観測**を policy に渡すことで「実行時刻の cube 位置を狙う」迎撃(WHERE)を担う。**速度をどう推定し、どこで前進させるか**を `PredictorConfig.mode` で選ぶ:

| mode | 速度推定 | どこで進めるか | 実装 | submodule |
|---|---|---|---|---|
| `image_shift`(デフォルト) | 色追跡(HSV 重心)の**単一速度** | **RGB ピクセル**上で cube を切り貼り。policy が再エンコード。 | `predictors/shift.py` (`shift_cube_in_frame`) | 不要 |
| `latent_warp` | 色追跡の**単一速度** | **vision パッチトークン**上で cube トークンを**剛体並進**。 | `predictors/latent_warp.py` (`warp_token_grid`) | 不要 |
| `latent_flow` | **dense optical flow** の**per-patch 速度** | パッチトークンを**各々の flow で前進**(grid_sample)。色判定なし・剛体仮定なし。 | `predictors/optical_flow.py` + `latent_warp.py` (`warp_token_grid_by_flow`) | 不要 |

3 つとも近年の潜在世界モデル(下記 §8)が示す「ピクセルより**凍結エンコーダの特徴空間**で時間を進める方が学習しやすく頑健」という流れに沿う。

- `latent_warp` は `shift_cube_in_frame` の潜在双子。**色マスクで cube を見つけ、重心速度 1 個で丸ごと平行移動**する軽量版。コンベアの cube がほぼ剛体並進ならこれで足りる。
- `latent_flow` は **色を一切使わず、連続フレーム間の dense optical flow から per-patch 速度を推定**し、各トークンを自分の flow で前進させる(`grid_sample` の backward warp)。AHEAD の「optical flow 条件付きで未来パッチトークンを予測」を、学習済み latent dynamics の代わりに**古典 flow(OpenCV DIS)＋解析的前進**で実現した版。flow バックエンドは差し替え可能で、将来 SEA-RAFT/NeuFlow 等の学習済みモデルを同じ `estimate` 契約で挿せる(その場合は submodule 追加を相談)。

### dense flow の per-patch 前進(`latent_flow`)

`latent_flow` の流れ: 連続フレームを `DenseFlowEstimator`(OpenCV DIS、状態保持)に通して **dense flow 場 `(H,W,2)`** を得る → lead 時間ぶんに換算(`flow · lead_s/dt`)→ パッチグリッドに平均プールして **per-patch 変位**にする → `warp_token_grid_by_flow` が `grid_sample` の backward sample(target `q` を `q − flow[q]` からサンプル)で各トークンを自分の flow で前進。`flow_motion_threshold` を超えるパッチだけ warp し、静止領域(背景・待機アーム)はそのまま残す(AHEAD の saliency mask 相当)。

### 注入点(SmolVLA)

両 latent モードは SmolVLA のエンコード経路に**デフォルト無効**のフックを 1 つ足して実現する。

| 変更点 | ファイル | 内容 |
|---|---|---|
| トークン warp の核 | `predictors/latent_warp.py` | `warp_token_grid`(剛体: cube トークンを `offset_tokens` 並進＋背景 median 埋め＋OOB drop)と `warp_token_grid_by_flow`(dense: per-patch flow を `grid_sample` で前進＋motion mask)。 |
| dense flow 推定 | `predictors/optical_flow.py` (`DenseFlowEstimator`) | 連続フレームの OpenCV DIS/Farneback flow。初回フレームは `None`。エピソード毎に reset。学習済み backend は同じ `estimate` 契約で差し替え可。 |
| エンコーダ内フック | `policies/smolvla/smolvlm_with_expert.py` (`embed_image`) | vision encoder 出力(connector の **pixel-shuffle 前**、row-major `H·W` の SigLIP 特徴グリッド)に `warp_fn` を任意適用。`warp_fn=None` で従来どおり。 |
| カメラ別 warp の配線 | `policies/smolvla/modeling_smolvla.py` (`embed_prefix` / `SmolVLAPolicy.set_latent_warp`) | present 画像特徴の順に warp 関数リストを割り当て。`set_latent_warp` は context manager で 1 推論の間だけ設定し、終了時に必ず復元(リークなし)。 |
| RTC 配線 | `rollout/inference/rtc.py` (`_time_advanced_obs` / `_latent_warp_context` / `_build_latent_warp_fns`) | image_shift はピクセル編集、latent は `predict_action_chunk` を `set_latent_warp(...)` で包む。predictor/flow は 1 ループ 1 回だけ実行。マスク/offset/flow 場は `resize_imgs_with_padding` を通してエンコーダ入力空間へ射影し、パッチグリッドと整合させる。 |
| 設定キー | `predictors/config.py` | `mode`(`image_shift`/`latent_warp`/`latent_flow`)・`latent_mask_threshold`・`flow_algorithm`(`dis`/`farneback`)・`flow_motion_threshold`。**デフォルトは `image_shift`** で既存挙動不変。 |

connector の pixel-shuffle **前**の素の SigLIP グリッド(row-major)で warp するため、トークン再サンプリングの内部仕様に依存しない。有効化は例えば `--inference.predictor.enabled=true --inference.predictor.mode=latent_flow --inference.predictor.camera=overall`。

> 注: warp は `grid_sample`/`nonzero` などデータ依存の制御を含むため `torch.compile` 下では graph break しうる(eager 運用想定)。predictor/flow 自体が numpy/OpenCV ベースで元々非 compile 経路。

---

## 6. 評価と将来課題

**評価(まずこの2点)**

- **無駄動作の削減**: ゲート ON/OFF で、cube 到来前のアーム移動量・誤把持の有無を比較。
- **成功率 vs ベルト速度**: ゲートだけで取りこぼしがどこまで改善するか。学習フィルタ ON/OFF も比較。

**将来課題(学習を伴う・本メモの範囲外)**

- **WHERE(迎撃)**: §5.5 の 3 モードは training-free な迎撃。`image_shift`/`latent_warp` は解析的な定速外挿(`p_pred = p + v·Δt`)、`latent_flow` は古典 dense flow の per-patch 前進。まずこれらで「後追い bias」がどこまで減るかを測る。
- **学習済み flow backend**: 古典 flow(DIS)が遮蔽・低テクスチャで崩れる場合は、`DenseFlowEstimator` と同じ `estimate` 契約で SEA-RAFT/NeuFlow 等の**学習済みモデルを submodule 追加**して差し替える(本体・RTC 統合は無改修)。コンベアの単一剛体には過剰なので必要になってから。
- **学習済み latent dynamics(AHEAD-lite)**: 定速/定 flow 前進では足りない(非線形運動・appearance 変化)場合は、同じ注入点に flow velocity 条件付きの小さな forward model(flow-matching か小 transformer)で未来パッチトークンを予測する版を差し替える。vendor 済みの V-JEPA2(`policies/vla_jepa`)を backbone に流用可。
- **条件付け**: overhead 由来の位置/速度/到達時刻を SmolVLA の state トークンに注入し、hindsight relabel + LoRA で微調整。空フレームを捨てずに「条件付き hold/動き出し」を学ぶ版。
- いずれも学習コストと reachability の検証が要るので、まず training-free ゲート + 解析/flow warp で効果を測ってから判断する。

---

## 7. 物理的な天井(正直な限界)

ゲートで始動を最適化しても、**アーム最大速度で把持作業域に届かないベルト速度**では取れない。可達域のベルト方向長さ `L`、ベルト速 `s`、アーム移動時間 `Δt_arm` として、取れる上限は概ね `s_max ≈ L / Δt_arm`。`s > s_max` はゲートでも予測でも不可能で、対策はロボットをベルトに寄せる / 迎撃点を上流に取る / アームを速くする側。評価で `s_max` を明示する。

---

## 8. 参考(要点のみ)

- Black, Galliker, Levine. *Real-Time Chunking (RTC)*, 2025. https://arxiv.org/abs/2506.07339 — チャンク境界の滑らかさを担保。反応性は別レイヤ。
- Liu ほか. *Bidirectional Decoding (BID)*, ICLR 2025. https://arxiv.org/abs/2408.17355 — チャンク実行中は直近観測への反応が落ちる(動く標的で失敗)。
- Islam ほか. *Constant-Time Replanning off a Conveyor*, RSS 2020. https://arxiv.org/abs/2101.07148 — 「確信を待つ/一発計画/逐次 replan」で 34.6/16/69.2%。早く動き出し安く replan。
- Akinola ほか. *Dynamic Grasping with Reachability and Motion Awareness*, IROS 2021. https://arxiv.org/abs/2103.10562 — eye-to-hand 外部カメラで予測、終端は近接カメラ、という役割分担。
- *Intercepting the Future (AHEAD)*, 2026. https://arxiv.org/abs/2606.02486 — frozen VLA の特徴空間で、optical flow 由来の per-token 速度/加速度に条件付けして未来パッチトークンを予測(flow-matching latent dynamics + adaptive horizon)。`latent_warp` の学習版の参照先。
- Hu ほか. *LaDi-WM: Latent Diffusion World Model*, 2025. https://arxiv.org/abs/2505.11528 — DINO 幾何 + CLIP 意味の特徴空間で latent diffusion 予測。「ピクセルより特徴空間で進める方が学習しやすい」を示す。AHEAD の前身。
- Zhou ほか. *DINO-WM: Zero-Shot Latent World Model*, 2024. https://arxiv.org/abs/2411.04983 — 凍結 DINOv2 パッチ特徴上の latent dynamics(reconstruction 不要)。「frozen encoder 特徴で時間を進める」原型。
</content>
</invoke>
