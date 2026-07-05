# 非同期レイテンシ実験（sync vs RTC）

動くコンベア上の cube を掴む reactive-VLA タスクで、**推論レイテンシが成功率にどう効くか**を sim 上で測った実験の記録。結論を先に言うと: **同期（レイテンシ無し）では 100% 掴めるが、非同期（実機相当のレイテンシ）では実質 0% になり、RTC のパラメータ調整では回復しない**。レイテンシ補償（未来位置予測 = Tier 3）が必須であることを定量的に示す。

## 対象ポリシー / 条件

- ポリシー: `lerobot/smolvla_base` を **シム収集データ `OctpusVLA/sim_pickplace_belt_010`**（白アーム・`box_top` 単一カメラ・belt 0.10・50 エピソード）で 40000 steps ファインチューニング。学習 loss 0.82→0.003、**同条件の同期評価で 10/10**。
  （`box_top` はこの実験当時のカメラ名。現在のコードでは `overview` にリネーム済み — 同じ外部視点カメラを指す。）
- 評価: `pixi run sim-eval`、入力カメラ `box_top`(現 `overview`) `→ observation.images.camera1`、`--success-criterion place_in_box`、`--end-on-drop`（cube がベルトから落ちたら/箱に入ったら終了）、各 10 ロールアウト、`--device cuda`。
- 同期 = `--rtc` なし（推論中は sim が凍結）。非同期 = `--rtc`（推論を別スレッドで回し、その間も sim=ベルトが進む。[docs/rtc-sim-rollout.md](rtc-sim-rollout.md) 参照）。

## 実験1: 同期 vs 非同期（belt 0.10）

| モード | 成功率 | 平均成功ステップ |
|---|---|---|
| 同期 (`--rtc` なし) | **10/10 = 100%** | 101 |
| 非同期 (`--rtc`) | **≈ 0/10 = 0%** | — |

非同期の失敗エピソードはほぼ全て **~150 ステップ = cube がベルト端まで進んで落下**（一度も掴めず）。同期は推論中にベルトが止まるので「見た瞬間の cube」をそのまま掴めて 100%。**この差はレイテンシそのもの**（GPU 推論なので CPU 遅延の話ではない）。

## 実験2: RTC の execution_horizon スイープ（非同期・belt 0.10）

「RTC のチャンク実行スケジュール（execution_horizon）を変えれば非同期でも掴めるのでは？」を検証。結果は **全 horizon で 0/10**。

| execution_horizon | 成功 | avg real_delay |
|---|---|---|
| 8 | 0/10 | 11.0 |
| 12 | 0/10 | 12.0 |
| 16 | 0/10 | 11.9 |
| 24 | 0/10 | 11.9 |
| 32 | 0/10 | 11.1 |
| 48 | 0/10 | 11.9 |

- **`avg real_delay ≈ 11–12` が horizon に依らず一定** = 推論レイテンシ（制御ステップ換算）そのもので、horizon では縮まらない。
- 本質: ポリシーは常に「~11 ステップ前の cube 位置」に向けて動くため、非同期では推論中も cube が進み、グリッパが**動く的の後ろに着地して空振り**する。チャンクのスケジュールを変えても参照観測が古い事実は変わらない。
- 生データ: `outputs/experiments/rtc_sweep_0703_1024/results.md`。

## 実験3: ベルト速度スイープ（sync vs 非同期）

「非同期レイテンシは**どの速度から**効くか」を、両モードで belt_speed を振って測る。

<!-- BELT_SWEEP_RESULTS -->
（実行中: `jobs/experiments/belt_speed_sweep.pbs`。完了後にここへ表を追記する。予想: 同期は全速度で高成功率を保つ一方、非同期は速度が上がるほど成功率が落ちる。境界速度が「予測なしで掴める上限」。）

## 結論

- **レイテンシ補償（未来位置予測）無しでは、RTC のパラメータ調整だけでは動くベルトを掴めない。**
- 同期 100% は「レイテンシ無し」という非現実的な上限。実機相当の非同期では大きく落ちる — これがリアクティブ VLA の核心課題。
- 次段は **Tier 3 predictor**（overhead カメラで cube 速度を推定し、推論レイテンシ分だけ未来位置へずらして入力＝`--predict-cube`）。同じ非同期条件で成功率が戻れば、予測の効果を直接示せる。

## 再現方法

```bash
# execution_horizon スイープ（非同期・belt 0.10）
bash jobs/experiments/rtc_sweep.pbs           # or qsub（バッチ）
# ベルト速度スイープ（sync vs 非同期）
bash jobs/experiments/belt_speed_sweep.pbs    # or qsub（バッチ）
```
GPU ノードで実行（推論が CUDA、ログインノードは CPU で極端に遅い）。結果は `outputs/experiments/<name>_<ts>/results.md` に表で残る。単発は例えば:
```bash
pixi run sim-eval --policy <ckpt> --belt-speed 0.10 --success-criterion place_in_box \
  --policy-camera overview --rtc --device cuda --episodes 10 --repo-id rollout_sim_boxtop_rtc   # 非同期
pixi run sim-eval --policy <ckpt> --belt-speed 0.10 --success-criterion place_in_box \
  --policy-camera overview --device cuda --episodes 10 --repo-id rollout_sim_boxtop_sync        # 同期
```
（`--policy-camera` は既定が `overview` だが、将来デフォルトが変わっても再現できるよう明示している。）
`--repo-id` を付けると動画＋ `eval_summary.json` が rollout ディレクトリに残る。
