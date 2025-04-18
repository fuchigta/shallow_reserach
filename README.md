# ShallowResearch

ウェブサイトの構造化されたリサーチを自動化するツール

## 概要

ShallowResearch は、ウェブサイト（特に技術ドキュメントや API 仕様など）を自動的にクロールし、各ページの内容を要約して構造化されたマークダウンドキュメントを生成するツールです。主な目的は、大量のドキュメントを短時間で理解し、重要なポイントを抽出することです。

## 特徴

- **自動サイトマップ抽出**: ナビゲーション要素から自動的にサイトマップを生成
- **並列クローリング**: 複数ページを同時に処理
- **テキスト要約**: Gemini 2.0 Flash による AI 要約
- **コード例抽出**: ページ内のコード例を抽出・保存
- **最終レポート生成**: すべての要約を統合した最終レポートを生成（マークダウンと HTML 形式）
- **マルチプロバイダー対応**: Google AI, OpenAI, Azure OpenAI, Anthropic
- **API キーの設定状況から自動的にプロバイダーを選択**
- **OpenAI 互換サービスの柔軟な設定（環境変数対応）**

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/fuchigta/shallow_reserach.git
cd shallow_reserach

# 依存関係のインストール
pip install -r requirements.txt

# Playwrightブラウザのインストール
playwright install chromium
```

## 使用方法

基本的な使用方法:

```bash
python shallow_research.py https://example.com/docs
```

API キーは環境変数またはコマンドラインオプションで設定できます。プロバイダーを指定しない場合は、利用可能な API キーから自動的に選択されます:

```bash
# 環境変数で設定（優先順位順）
export GOOGLE_API_KEY=your_api_key    # Google AI
export OPENAI_API_KEY=your_api_key    # OpenAI
export ANTHROPIC_API_KEY=your_api_key # Anthropic
export AZURE_OPENAI_API_KEY=your_api_key # Azure OpenAI

python shallow_research.py https://example.com/docs  # 自動選択
```

各 LLM プロバイダーの使用例:

```bash
# Google AI (デフォルト)
export GOOGLE_API_KEY=your_api_key
python shallow_research.py https://example.com/docs

# OpenAI GPT-4
export OPENAI_API_KEY=your_api_key
python shallow_research.py https://example.com/docs -p openai

# Azure OpenAI
export AZURE_API_KEY=your_api_key
python shallow_research.py https://example.com/docs -p azure \
  --api-base "https://your-resource.openai.azure.com" \
  --api-version "2024-02-15-preview" \
  --deployment-name "your-deployment"

# Anthropic Claude
export ANTHROPIC_API_KEY=your_api_key
python shallow_research.py https://example.com/docs -p anthropic
```

OpenAI 互換サービスの設定は、環境変数でも指定できます：

```bash
# OpenAI互換サービスの設定（環境変数）
export OPENAI_API_BASE="https://your-api-endpoint"    # APIエンドポイント
export OPENAI_API_VERSION="2024-02-15-preview"       # APIバージョン（Azure）
export OPENAI_DEPLOYMENT_NAME="your-deployment-name"  # デプロイメント名（Azure）
```

その他のオプション:

```bash
# 出力ディレクトリを指定
python shallow_research.py https://example.com/docs -o my_research

# 同時実行数とレート制限を調整
python shallow_research.py https://example.com/docs -c 5 -r 0.5

# 異なるLLMモデルを使用
python shallow_research.py https://example.com/docs -m gpt-4-turbo-preview

# 詳細出力モード
python shallow_research.py https://example.com/docs -v

# すべてのページを強制的に再処理
python shallow_research.py https://example.com/docs -f

# 最終要約のみを生成
python shallow_research.py https://example.com/docs --final-only
```

## 出力ファイル

ツールは以下のファイルを生成します:

- `sitemap.json`: 抽出されたサイトマップ
- `page_N_*.md`: 各ページの要約（マークダウン形式）
- `final_summary.md`: すべてのページの要約を統合した最終レポート
- `final_summary.html`: HTML フォーマットの最終レポート

## コマンドラインオプション

| オプション          | 説明                                                  |
| ------------------- | ----------------------------------------------------- |
| `url`               | 調査対象の URL（必須）                                |
| `-o, --output`      | 出力ディレクトリ（デフォルト: research_output）       |
| `-c, --concurrency` | 同時実行数（デフォルト: 3）                           |
| `-r, --rate-limit`  | リクエスト間の待機時間（秒）（デフォルト: 1）         |
| `-p, --provider`    | LLM プロバイダ（google/openai/azure/anthropic）       |
| `-m, --model`       | 使用する LLM モデル（プロバイダごとのデフォルトあり） |
| `-k, --api-key`     | LLM プロバイダの API キー                             |
| `-v, --verbose`     | 詳細出力モード                                        |
| `-f, --force`       | すべてのページを強制的に再処理                        |
| `--final-only`      | 最終要約のみを生成                                    |
| `--api-base`        | OpenAI 互換サービスのベース URL                       |
| `--api-version`     | Azure OpenAI の API バージョン                        |
| `--deployment-name` | Azure OpenAI のデプロイメント名                       |

## デフォルトの LLM モデル

各プロバイダのデフォルトモデル:

- Google AI: `gemini-1.5-flash`
- OpenAI: `gpt-4-turbo-preview`
- Anthropic: `claude-3-opus-20240229`
- Azure OpenAI: `gpt-4`

## アーキテクチャ

ShallowResearch は以下のコンポーネントで構成されています:

1. **サイトマップ抽出**: Playwright を使用してウェブサイトのナビゲーション構造を解析
2. **コンテンツ抽出**: 各ページからメインコンテンツとコード例を抽出
3. **テキスト要約**: Langchain を使用して Gemini 2.0 Flash で AI 要約を生成
4. **レポート生成**: 要約を統合して最終レポートを作成

## 拡張ポイント

このツールは以下の方向に拡張可能です:

- **知識グラフの構築**: ページ間の関連性を表現する構造の追加
- **検索機能の実装**: 生成された要約データに対するローカル検索エンジン
- **対話モード**: 特定のページや概念について質問できるインタラクティブモード
- **定期更新機能**: ドキュメントの変更を検出し要約を更新する機能
- **異なる LLM プロバイダー**: OpenAI、Anthropic など他の LLM プロバイダーの使用

## 必要条件

- Python 3.8+
- Google AI API キー（Gemini 2.0 Flash にアクセスできるもの）
- インターネット接続

## ライセンス

MIT ライセンス

## 謝辞

このプロジェクトは以下のライブラリを使用しています:

- [Playwright](https://playwright.dev/): モダンなウェブブラウザ自動化
- [Langchain](https://www.langchain.com/): LLM 統合フレームワーク
- [Google Generative AI](https://ai.google.dev/): Gemini API へのアクセス
- [Rich](https://rich.readthedocs.io/): リッチなコンソール出力
