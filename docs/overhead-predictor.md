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

## 6. 評価と将来課題

**評価(まずこの2点)**

- **無駄動作の削減**: ゲート ON/OFF で、cube 到来前のアーム移動量・誤把持の有無を比較。
- **成功率 vs ベルト速度**: ゲートだけで取りこぼしがどこまで改善するか。学習フィルタ ON/OFF も比較。

**将来課題(学習を伴う・本メモの範囲外)**

- **WHERE(迎撃)**: ゲートは「いつ動くか」を直すだけで、**「動く cube を追い越せない後追い(chase)bias」は残る**。速いベルトでは把持時刻の cube 位置を狙う predict-and-intercept が要る(`p_pred = p + v·Δt`)。
- **条件付け**: overhead 由来の位置/速度/到達時刻を SmolVLA の state トークンに注入し、hindsight relabel + LoRA で微調整。空フレームを捨てずに「条件付き hold/動き出し」を学ぶ版。
- いずれも学習コストと reachability の検証が要るので、まず本メモの training-free ゲートで効果を測ってから判断する。

---

## 7. 物理的な天井(正直な限界)

ゲートで始動を最適化しても、**アーム最大速度で把持作業域に届かないベルト速度**では取れない。可達域のベルト方向長さ `L`、ベルト速 `s`、アーム移動時間 `Δt_arm` として、取れる上限は概ね `s_max ≈ L / Δt_arm`。`s > s_max` はゲートでも予測でも不可能で、対策はロボットをベルトに寄せる / 迎撃点を上流に取る / アームを速くする側。評価で `s_max` を明示する。

---

## 8. 参考(要点のみ)

- Black, Galliker, Levine. *Real-Time Chunking (RTC)*, 2025. https://arxiv.org/abs/2506.07339 — チャンク境界の滑らかさを担保。反応性は別レイヤ。
- Liu ほか. *Bidirectional Decoding (BID)*, ICLR 2025. https://arxiv.org/abs/2408.17355 — チャンク実行中は直近観測への反応が落ちる(動く標的で失敗)。
- Islam ほか. *Constant-Time Replanning off a Conveyor*, RSS 2020. https://arxiv.org/abs/2101.07148 — 「確信を待つ/一発計画/逐次 replan」で 34.6/16/69.2%。早く動き出し安く replan。
- Akinola ほか. *Dynamic Grasping with Reachability and Motion Awareness*, IROS 2021. https://arxiv.org/abs/2103.10562 — eye-to-hand 外部カメラで予測、終端は近接カメラ、という役割分担。
</content>
</invoke>
