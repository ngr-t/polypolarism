# Polars DataFrame 静的型検査フレームワーク

## ― レコード多相性に基づく列・型推論チェッカー（MVP指示書）

### 1. 背景と問題意識

* Python の型アノテーションは DataFrame に対して表現力が極端に弱い
* 特に

  * 列の存在制約
  * 列の追加・削除・変換
  * join / groupby などの集合演算
    を **静的に検証できない**
* Pandera 等はランタイム検証に強いが、
  **レコード多相性（row polymorphism）を使った静的検証はできない**

本プロジェクトは
**Polars を対象に、レコード多相性に近い型体系を持つ静的検査ツール**
を実装することを目的とする。

---

### 2. ゴール定義

#### 最終ゴール（ビジョン）

* Polars の DataFrame パイプラインに対して

  * 列の存在
  * dtype（nullable / list / struct / categorical 等を含む）
  * join / groupby / agg の整合性
    を **実行せずに検証**できる

#### MVPゴール

* Polars 専用
* 静的検査（CLIツール）
* ANSI SQL 相当の操作が型検証できる

  * **join**
  * **group_by().agg()**
* 検証対象は **軸C**

  * dtype
  * nullable
  * List / Struct / その他 Polars の主要 dtype

---

### 3. 非ゴール（MVPではやらない）

* Pandas / Spark 対応
* ランタイム検証（Panderaの代替はしない）
* IDE プラグイン
* 動的列名・正規表現列（将来拡張を見据えた設計のみ）

---

### 4. ツール形態

* **静的検査 CLI**

  * Python コードを実行しない
  * AST 解析 + 型推論
* CI で落とせる出力を提供
* 型チェッカー（mypy/pyright）との統合はしない
  → 独立したセマンティックチェッカー

---

### 5. 基本設計方針

#### 5.1 二層型体系

1. **Frame 型（DataFrame 型）**

   * 列集合 + 各列の dtype
2. **Expr 型（Polars Expr 型）**

   * `pl.Expr` の計算結果の dtype

すべての操作は
**Expr 型推論 → Frame 型更新**
として定義する。

---

### 6. 型の内部表現

#### 6.1 DataType

* Polars の dtype を忠実にモデル化する
* 最初から拡張可能な設計にする

最低限含めるもの：

* Int64, Int32, UInt*, Float32/64
* Utf8
* Boolean
* Date, Datetime(tz), Duration
* Decimal(p, s)
* Categorical / Enum
* List[T]
* Struct{ field -> T }
* Nullability（Optional[T] or Nullable[T]）

#### 6.2 FrameType

```text
FrameType:
  known_columns: Dict[str, DataType]
  rest: Optional[RowVar]  # MVPでは None だが将来拡張用に保持
```

---

### 7. 型注釈（Schema DSL）

#### 7.1 注釈の役割

* Python の型注釈を **このツール専用 DSL** として使う
* mypy 等に理解させる必要はない

#### 7.2 使用例

```python
from dfcheck import DF

def f(
  users: DF["{user_id:Int64, country:Utf8, age:Int64?}"],
  purchases: DF["{user_id:Int64, amount:Float64}"]
):
    ...
```

#### 7.3 DSL仕様（MVP）

* `{col:Type, ...}`
* `Type?` → nullable
* `List[T]`
* `Struct{a:T, b:U}`

※ 固定列名のみ（パターン列は将来拡張）

---

### 8. 静的解析の基本戦略

* Python AST を解析
* 関数単位で以下を行う：

  1. 引数注釈から初期 FrameType を構築
  2. 代入・メソッドチェーンを追跡
  3. 各操作で FrameType を更新
  4. return 時点の FrameType を取得
  5. 出力注釈があれば照合、なければ推論結果をレポート

※ 一部の変数だけ注釈があっても、推論は全変数に伝播する

---

### 9. 対象操作（MVP）

#### 9.1 実装必須

* `join`
* `group_by().agg()`

#### 9.2 補助的に必要

* `pl.col`
* `pl.lit`
* `.alias`
* 算術・比較・論理演算
* `cast`
* `when/then/otherwise`

---

### 10. join の型推論仕様

#### 入力

* 左 FrameType L
* 右 FrameType R
* join keys（`on` / `left_on` / `right_on`）
* join type（inner / left / right / full）

#### 検証

* join key の dtype が一致しない場合は **エラー**

#### 出力 FrameType

* **inner**

  * 両側の列を保持
* **left**

  * 右由来列は nullable
* **right**

  * 左由来列は nullable
* **full**

  * 両側由来列が nullable

#### 列名衝突

* Polars のデフォルト挙動に寄せる
* 右側に suffix を付与（MVPでは決め打ち）
* key 列の扱いも Polars のデフォルトに固定

---

### 11. group_by().agg() の型推論仕様

#### 出力列

* group keys

  * dtype は入力と同じ
* agg expr

  * 1 expr = 1 出力列
  * alias がない場合は Polars の命名規則に従う（MVPでは警告推奨）

#### 集約関数の型シグネチャ（例）

* `sum(Int64) -> Int64`
* `mean(Int64) -> Float64`
* `count(*) -> UInt64`
* `n_unique(T) -> UInt64`
* `list(T) -> List[T]`
* `first(T) -> T`
* nullable の扱いは関数ごとに定義

※ この **集約関数シグネチャ表** が最重要資産

---

### 12. Expr 型推論（MVP）

* `pl.col("x")`

  * FrameType から取得
  * 存在しなければエラー
* `pl.lit(v)`

  * リテラル型
* 演算子

  * 型昇格ルール表に基づく
* `cast`

  * 指定 dtype
* `when/then/otherwise`

  * then/otherwise を unify
  * nullable は増える方向

---

### 13. 実装構成案（そのままタスク分割可能）

```
dfcheck/
  types.py        # DataType, FrameType
  dsl.py          # DF["..."] パーサ
  expr_infer.py   # Expr 型推論
  ops/
    join.py
    groupby.py
  analyzer.py     # AST解析・データフロー
  checker.py      # 宣言 vs 推論の照合
  cli.py
docs/
  spec.md
```

---

### 14. 将来拡張を見据えた設計上の注意

* FrameType に `rest` を残す（row polymorphism 拡張点）
* 操作は registry 化（後から pivot/explode を追加可能に）
* dtype は enum ではなくクラス階層で表現

---

### 15. 成功条件（MVP完了の定義）

* join + groupby/agg を含む Polars パイプラインを

  * 実行せずに
  * 列存在・dtype・nullableまで含めて
  * 正しく検証できる
* エラーは「どの操作で・どの列が・なぜ不正か」を説明できる

## 実装手順

1. 開発環境の設定
  uv を使ってよしなに。
2. ディレクトリ構成作成
  ここまでで一度コミット。
3. 全体テストケース作成
  Polars のチュートリアルから今回の該当するケースを引っ張ってきたりして、通過するべき型アノテーションつきスクリプトとするべきでないするべきでないアノテーションつきスクリプトをそれぞれ5つほど定義してほしい。
3. 初期設計
ここからはt-wada スタイルで TDD を行ってほしい。
4. 単体テスト設計
  テストを書いた時点で難しそうであれば設計を変えても良い。
5. 実装

こまめにブランチを切り、コミットを行うこと。