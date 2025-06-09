#!/usr/bin/env python3
"""
Shallow Research - ウェブサイトの構造化されたリサーチを自動化するツール
"""

import os
import sys
import argparse
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from urllib.parse import urlparse
from pydantic import BaseModel, Field

from playwright.async_api import async_playwright, Page
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID

from dotenv import load_dotenv
load_dotenv()

# 定数
DEFAULT_OUTPUT_DIR = "research_output"
DEFAULT_CONCURRENCY = 3
DEFAULT_RATE_LIMIT = 1  # リクエスト間の秒数
MAX_RETRIES = 3
RETRY_DELAY = 2
PATH_RESTRICTION_ENABLED = True  # パス制限のデフォルト値

# プロバイダ設定
LLM_PROVIDERS = ["google", "openai", "anthropic", "azure"]  # 優先順位順
LLM_API_KEY_ENV = {
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY"
}
DEFAULT_LLM_MODELS = {
    "google": "gemini-2.5-flash-preview-05-20",
    "openai": "gpt-4-turbo-preview",
    "anthropic": "claude-3-opus-20240229",
    "azure": "gpt-4"  # Azure OpenAIのデフォルトモデル
}

# OpenAI互換サービスのデフォルト設定
OPENAI_ENV_VARS = {
    "api_base": "OPENAI_API_BASE",
    "api_version": "OPENAI_API_VERSION",
    "deployment_name": "OPENAI_DEPLOYMENT_NAME"
}
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"  # OpenAIのデフォルトベースURL
DEFAULT_OPENAI_API_VERSION = None  # Azure OpenAIの場合は必要

# Pydanticモデル
class SitemapEntry(BaseModel):
    """サイトマップのエントリ"""
    url: str = Field(description="ページのURL")
    title: str = Field(description="ページのタイトル")

class SitemapResult(BaseModel):
    """サイトマップ抽出結果"""
    sitemap: Dict[str, str] = Field(
        description="URLをキー、タイトルを値とするサイトマップ",
        default_factory=dict
    )

def detect_available_provider() -> Optional[str]:
    """
    利用可能なLLMプロバイダを検出する
    
    Returns:
        str: 利用可能なプロバイダ名、見つからない場合はNone
    """
    for provider in LLM_PROVIDERS:
        if os.environ.get(LLM_API_KEY_ENV[provider]):
            return provider
    return None

class LLMFactory:
    """LLMプロバイダのファクトリークラス"""
    
    @staticmethod
    def create_llm(
        provider: Literal["google", "openai", "anthropic", "azure"],
        model: str,
        api_key: str,
        **kwargs
    ) -> BaseChatModel:
        """
        LLMインスタンスを作成する
        
        Args:
            provider: LLMプロバイダ ("google", "openai", "anthropic", "azure")
            model: 使用するモデル名
            api_key: APIキー
            **kwargs: その他のオプション
                - api_base: OpenAI互換サービスのベースURL
                - api_version: Azure OpenAIのAPIバージョン
                - deployment_name: Azure OpenAIのデプロイメント名
                - temperature: 生成時の温度
                - max_tokens: 最大トークン数
            
        Returns:
            LLMインスタンス
        """
        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=kwargs.get("temperature", 0.2),
                max_output_tokens=kwargs.get("max_tokens"),
            )
        elif provider in ["openai", "azure"]:
            openai_kwargs = {
                "model": model if provider == "openai" else None,
                "deployment_name": kwargs.get("deployment_name") if provider == "azure" else None,
                "temperature": kwargs.get("temperature", 0.2),
                "max_tokens": kwargs.get("max_tokens"),
                "api_key": api_key,
                "api_version": kwargs.get("api_version") if provider == "azure" else None,
                "azure": provider == "azure",
            }
            
            # ベースURLの設定
            api_base = kwargs.get("api_base")
            if api_base:
                openai_kwargs["api_base"] = api_base
            elif provider == "azure":
                raise ValueError("Azure OpenAIを使用する場合は、api_baseの指定が必要です")
            
            return ChatOpenAI(**{k: v for k, v in openai_kwargs.items() if v is not None})
            
        elif provider == "anthropic":
            return ChatAnthropic(
                model=model,
                api_key=api_key,
                temperature=kwargs.get("temperature", 0.2),
                max_tokens=kwargs.get("max_tokens"),
            )
        else:
            raise ValueError(f"不明なLLMプロバイダです: {provider}")

SUMMARY_TEMPLATE = """
以下はウェブページの内容です。

これらの情報を基に、ページの全体像を把握できる要約を日本語でマークダウン形式で作成してください。
出力にはマークダウンのみを含めてください。

1. ページの内容をよく読み、深く理解してください。
2. 理解した内容に基づいて、以下の点に注意して要約を作成してください：
   1. ページの全体像を視覚的にわかりやすく説明
   2. 本ページで説明されている主要な概念について、具体例を交えつつ見出しを活用して階層的に整理する

---

ページのタイトル: {title}
ページのURL: {url}

ページの内容:
{content}
"""

SECTION_SUMMARY_TEMPLATE = """
以下はドキュメントの1つのセクションに含まれるサブセクションの要約群です。

これらの情報を基に、セクションの全体像を把握できる要約を日本語でマークダウン形式で作成してください。
出力にはマークダウンのみを含めてください。

1. サブセクションの要約群をよく読み、深く理解してください。
2. 理解した内容に基づいて、以下の点に注意して要約を作成してください：
   1. セクションの全体像を視覚的にわかりやすく説明
   2. 各サブセクションの主要な概念について、具体例を交えつつ見出しを活用して階層的に整理する

---

セクション: {section_name}
サブセクションの要約群:
{summaries}
"""


FINAL_SUMMARY_TEMPLATE = """
以下は{site_name}のドキュメントから抽出した各セクションの要約です。

これらの情報を基に、最終的な調査レポートを日本語でマークダウン形式で作成してください。
出力にはマークダウンのみを含めてください。

1. 各セクションの要約をよく読み、深く理解してください。
2. 理解した内容に基づいて、以下の点に注意してレポートを作成してください：
   1. ドキュメントの全体像を視覚的にわかりやすく説明
   2. 各セクションの主要な概念について、具体例を交えつつ見出しを活用して階層的に整理する

---

各セクションの要約:
{section_summaries}
"""

class ShallowResearcher:
    """ウェブサイトを自動的にクロールして要約するクラス"""
    
    def __init__(
        self,
        url: str,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        concurrency: int = DEFAULT_CONCURRENCY,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        verbose: bool = False,
        force_rerun: bool = False,
        restrict_path: bool = PATH_RESTRICTION_ENABLED,  # パス制限オプション
        mcp_server_url: Optional[str] = None,  # MCPサーバーのURL
        **kwargs
    ):
        """
        初期化メソッド
        
        Args:
            url: 調査対象のルートURL
            output_dir: 出力ディレクトリ
            concurrency: 同時実行数
            rate_limit: リクエスト間の秒数
            llm_provider: LLMプロバイダ。未指定の場合は利用可能なプロバイダを自動検出
            llm_model: 使用するLLMモデル（指定がない場合はプロバイダのデフォルトモデル）
            api_key: LLM APIキー
            verbose: 詳細出力モード
            force_rerun: すべてのページを強制的に再実行
            restrict_path: ルートURLのパスに基づいてURLを制限する
            mcp_server_url: MCPサーバーのURL（Playwright MCPサーバー、デフォルトでheadless）
        """
        self.root_url = url
        self.output_dir = Path(output_dir)
        self.concurrency = concurrency
        self.rate_limit = rate_limit
        self.verbose = verbose
        self.force_rerun = force_rerun
        self.restrict_path = restrict_path  # パス制限設定を保存
        self.root_path = urlparse(url).path  # ルートURLのパスを保存
        # デフォルトのPlaywright MCPサーバー（ヘッドレスモード）
        self.mcp_server_url = mcp_server_url if mcp_server_url is not None else "npx @playwright/mcp@latest --headless --browser chromium"
        self.site_map = {}
        self.summaries = {}
        self.visited = set()
        self.semaphore = None  # asyncioで初期化
        self.mcp_client = None  # MCPクライアント
        
        # コンソール出力
        self.console = Console()
        
        # LLMプロバイダの決定（自動検出または指定）
        if llm_provider:
            self.llm_provider = llm_provider.lower()
            if self.llm_provider not in LLM_PROVIDERS:
                raise ValueError(f"不明なLLMプロバイダです: {self.llm_provider}")
        else:
            self.llm_provider = detect_available_provider()
            if not self.llm_provider:
                raise ValueError("利用可能なLLMプロバイダが見つかりません。環境変数でAPIキーを設定するか、--providerオプションで指定してください。")
            if self.verbose:
                self.console.print(f"[cyan]LLMプロバイダを自動検出: {self.llm_provider}[/]")
        
        # モデルの設定
        self.llm_model = llm_model or DEFAULT_LLM_MODELS.get(self.llm_provider)
        if not self.llm_model:
            raise ValueError(f"モデルが指定されておらず、プロバイダ{self.llm_provider}のデフォルトモデルも見つかりません。")
        
        # APIキーの取得
        api_key = api_key or os.environ.get(LLM_API_KEY_ENV[self.llm_provider])
        if not api_key:
            raise ValueError(f"{LLM_API_KEY_ENV[self.llm_provider]}が必要です。環境変数で設定するか、--api-keyオプションで指定してください。")
        
        # OpenAI互換サービスのオプションを環境変数から取得
        for key, env_var in OPENAI_ENV_VARS.items():
            if key not in kwargs and os.environ.get(env_var):
                kwargs[key] = os.environ[env_var]
                if self.verbose:
                    self.console.print(f"[cyan]OpenAI互換オプション {key} を環境変数から設定: {kwargs[key]}[/]")
        
        # LLMの初期化
        try:
            self.llm = LLMFactory.create_llm(
                provider=self.llm_provider,
                model=self.llm_model,
                api_key=api_key,
                temperature=0.2,
                max_tokens=8192,
                **kwargs
            )
        except Exception as e:
            raise Exception(f"LLMの初期化に失敗しました: {e}")
        
        # チェーンの設定
        self.summary_chain = (
            ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
            | self.llm
            | StrOutputParser()
        )
        
        self.final_summary_chain = (
            ChatPromptTemplate.from_template(FINAL_SUMMARY_TEMPLATE)
            | self.llm
            | StrOutputParser()
        )
        
        # Structured Output用のパーサー
        self.sitemap_parser = PydanticOutputParser(pydantic_object=SitemapResult)
        
        # MCP Agent（初期化時には設定せず、initialize()で設定）
        self.mcp_agent = None
    
    def _clean_tool_schemas_for_gemini(self, tools):
        """
        Gemini向けにツールスキーマをクリーンアップする
        additionalPropertiesと$schemaを削除して警告を回避する
        """
        def clean_schema(schema):
            if isinstance(schema, dict):
                # additionalPropertiesと$schemaを削除
                cleaned = {k: v for k, v in schema.items() 
                          if k not in ['additionalProperties', '$schema']}
                # 再帰的にクリーンアップ
                return {k: clean_schema(v) for k, v in cleaned.items()}
            elif isinstance(schema, list):
                return [clean_schema(item) for item in schema]
            else:
                return schema
        
        cleaned_tools = []
        for tool in tools:
            try:
                # ツールのargs_schemaをクリーンアップ
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    tool.args_schema = clean_schema(tool.args_schema)
                cleaned_tools.append(tool)
            except Exception as e:
                if self.verbose:
                    self.console.print(f"[yellow]ツール {tool.name} のスキーマクリーンアップ中にエラー: {e}[/]")
                # エラーが発生してもツールは含める
                cleaned_tools.append(tool)
        
        return cleaned_tools
    
    async def initialize(self):
        """非同期リソースの初期化"""
        self.semaphore = asyncio.Semaphore(self.concurrency)
        
        # MCPクライアントの初期化
        if self.mcp_server_url == "":
            raise ValueError("MCPサーバーURLが設定されていません。--mcp-server-urlオプションでMCPサーバーを指定してください。")
        
        # MultiServerMCPClientの設定
        server_config = {
            "playwright": {
                "command": self.mcp_server_url.split()[0],  # "npx"
                "args": self.mcp_server_url.split()[1:],   # ["@playwright/mcp@latest", "--headless"] (headlessの場合)
                "transport": "stdio"
            }
        }
        
        self.mcp_client = MultiServerMCPClient(server_config)
        if self.verbose:
            self.console.print(f"[cyan]MCPクライアントを設定しました: {self.mcp_server_url}[/]")
        
        # MCPツールを取得してエージェントを作成
        try:
            mcp_tools = await self.mcp_client.get_tools()
            if self.verbose:
                self.console.print("[green]MCPクライアントが正常に動作しています[/]")
        except Exception as e:
            if self.verbose:
                self.console.print(f"[red]MCPクライアント接続エラー: {e}[/]")
            raise RuntimeError(f"MCPクライアントの接続に失敗しました: {e}")
        if not mcp_tools:
            raise RuntimeError("MCPツールが見つかりませんでした。MCPサーバーが正しく動作しているか確認してください。")
        
        # Gemini使用時はスキーマクリーンアップを実行
        if self.llm_provider == "google":
            mcp_tools = self._clean_tool_schemas_for_gemini(mcp_tools)
            if self.verbose:
                self.console.print("[cyan]Gemini向けにツールスキーマをクリーンアップしました[/]")
        
        # LangGraphのReActエージェントを作成
        self.mcp_agent = create_react_agent(self.llm, mcp_tools)
        if self.verbose:
            self.console.print(f"[cyan]MCPエージェント（LangGraph ReAct対応）を作成しました（ツール数: {len(mcp_tools)}）[/]")
            # 利用可能なツールをリスト表示
            tool_names = [tool.name for tool in mcp_tools]
            self.console.print(f"[cyan]利用可能なMCPツール: {', '.join(tool_names)}[/]")
    
    def _should_include_url(self, url: str) -> bool:
        """
        URLが処理対象に含めるべきかどうかを判定する
        
        Args:
            url: チェックするURL
            
        Returns:
            bool: URLを含めるべきかどうか
        """
        if not self.restrict_path:
            return True
            
        parsed_url = urlparse(url)
        url_path = parsed_url.path
        
        # ルートURLのパスで始まるURLのみを含める
        return url_path.startswith(self.root_path)

    async def debug_mcp_tools(self) -> None:
        """
        MCPツールの動作をデバッグするためのメソッド
        """
        if not self.mcp_agent:
            raise RuntimeError("MCPエージェントが利用できません。")
        
        if self.verbose:
            self.console.print(f"[cyan]MCPツールのデバッグテストを開始: {self.root_url}[/]")
        
        # シンプルなテストメッセージ
        test_question = f"""Playwright MCPツールを使用してウェブサイト {self.root_url} をテストしてください。

手順:
1. ページに移動
2. ページタイトルを取得
3. 基本的なリンクを1つ探す

実行した内容を詳しく説明してください。"""
        
        try:
            result = await self.mcp_agent.ainvoke({
                "messages": [("human", test_question)]
            })
            
            messages = result.get("messages", [])
            if messages:
                agent_output = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
                self.console.print(f"[green]デバッグテスト結果:[/]")
                self.console.print(agent_output)
            else:
                self.console.print("[yellow]エージェントからの応答がありませんでした。[/]")
                
        except Exception as e:
            self.console.print(f"[red]デバッグテスト中にエラー: {e}[/]")
            raise

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        テキストからJSONを抽出する改良版メソッド
        
        Args:
            text: 解析対象のテキスト
            
        Returns:
            抽出されたJSONオブジェクトまたはNone
        """
        import json
        import re
        
        # 1. コードブロック内のJSON検索
        code_block_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?"sitemap".*?\})\s*```',
            r'`(\{.*?"sitemap".*?\})`'
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    result = json.loads(match)
                    if "sitemap" in result and isinstance(result["sitemap"], dict):
                        return result
                except json.JSONDecodeError:
                    continue
        
        # 2. 改良されたJSONパターン（バランスの取れた括弧）
        def find_balanced_json(text: str, start_keyword: str = "sitemap") -> List[str]:
            """バランスの取れたJSONオブジェクトを探す"""
            results = []
            start_pos = 0
            
            while True:
                # sitemapキーワードを含む開始位置を探す
                sitemap_pos = text.find(f'"{start_keyword}"', start_pos)
                if sitemap_pos == -1:
                    break
                
                # その前の開始ブレースを探す
                brace_pos = text.rfind('{', 0, sitemap_pos)
                if brace_pos == -1:
                    start_pos = sitemap_pos + 1
                    continue
                
                # バランスの取れた終了ブレースを探す
                brace_count = 0
                for i in range(brace_pos, len(text)):
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # バランスの取れたJSONを発見
                            json_str = text[brace_pos:i+1]
                            results.append(json_str)
                            start_pos = i + 1
                            break
                else:
                    # 終了ブレースが見つからない
                    start_pos = sitemap_pos + 1
                    continue
                    
            return results
        
        # バランスの取れたJSONを探して解析
        json_candidates = find_balanced_json(text)
        if self.verbose:
            self.console.print(f"[cyan]バランスの取れたJSON候補: {len(json_candidates)}個[/]")
        
        for candidate in json_candidates:
            try:
                result = json.loads(candidate)
                if "sitemap" in result and isinstance(result["sitemap"], dict):
                    return result
            except json.JSONDecodeError as e:
                if self.verbose:
                    self.console.print(f"[yellow]JSON解析エラー: {e} - 候補: {candidate[:100]}...[/]")
                continue
        
        # 3. より単純なパターンマッチング
        simple_patterns = [
            r'"sitemap"\s*:\s*(\{[^}]+\})',  # シンプルなsitemapオブジェクト
            r'"sitemap"\s*:\s*(\{.*?\})',    # より柔軟なマッチング
        ]
        
        for pattern in simple_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # sitemapオブジェクトを完全なJSONに包む
                    json_str = f'{{"sitemap": {match}}}'
                    result = json.loads(json_str)
                    if isinstance(result["sitemap"], dict):
                        return result
                except json.JSONDecodeError:
                    continue
        
        return None

    async def extract_sitemap_with_mcp(self) -> Dict[str, str]:
        """
        LLMエージェントとPlaywright MCPを使用してサイトマップを抽出する
        
        Returns:
            サイトマップ (URL -> タイトル のマッピング)
        """
        if not self.mcp_agent:
            raise RuntimeError("MCPエージェントが利用できません。MCPサーバーが正しく設定されているか確認してください。")
        
        # LLMエージェントにサイトマップ抽出を依頼
        if self.verbose:
            self.console.print(f"[cyan]MCPエージェントでサイトマップ抽出を開始: {self.root_url}[/]")
        
        # 改良されたプロンプト（より具体的で構造化された指示）
        question = f"""ウェブサイト {self.root_url} からナビゲーションリンクを抽出してサイトマップを作成してください。

具体的な手順:
1. playwright_goto_page ツールを使用してページに移動
2. playwright_get_element_info または playwright_evaluate を使用して以下のセレクタからリンクを抽出:
   - nav a[href] (ナビゲーション内のリンク)
   - .nav a[href], .navigation a[href] (ナビゲーションクラス内のリンク)
   - header a[href] (ヘッダー内のリンク)
   - .menu a[href], .main-menu a[href] (メニュー内のリンク)
3. 各リンクのhref属性とテキスト内容を取得
4. 同じドメイン内のHTTPSリンクのみを対象とし、以下を除外:
   - アンカーリンク (#で始まる)
   - JavaScriptリンク (javascript:で始まる)
   - ファイルリンク (.pdf, .jpg, .png等で終わる)
   - mailto:リンク

結果を以下の厳密なJSON形式で返してください（他のテキストは含めないでください）:

```json
{{
  "sitemap": {{
    "https://example.com/page1": "Page 1 Title",
    "https://example.com/page2": "Page 2 Title"
  }}
}}
```

重要: 応答にはJSON以外の説明やコメントを含めないでください。"""
        
        try:
            result = await self.mcp_agent.ainvoke({
                "messages": [("human", question)]
            })
            
            if self.verbose:
                self.console.print(f"[cyan]MCPエージェントの実行完了[/]")
            
            # LangGraphの結果からメッセージを取得
            messages = result.get("messages", [])
            if messages:
                # 最後のメッセージからcontent取得
                agent_output = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
            else:
                agent_output = ""
                
            if self.verbose:
                self.console.print(f"[cyan]エージェントの出力長: {len(agent_output)} 文字[/]")
                # 出力の最初の500文字を表示
                preview = agent_output[:500] + "..." if len(agent_output) > 500 else agent_output
                self.console.print(f"[cyan]エージェント出力プレビュー: {preview}[/]")
            
            # 改良されたJSON抽出を試行
            parsed_result = self._extract_json_from_text(agent_output)
            
            if parsed_result and "sitemap" in parsed_result:
                sitemap_data = parsed_result["sitemap"]
                if isinstance(sitemap_data, dict) and sitemap_data:
                    # URLフィルタリング
                    filtered_sitemap = {}
                    for url, title in sitemap_data.items():
                        if isinstance(url, str) and isinstance(title, str):
                            if self._should_include_url(url):
                                filtered_sitemap[url] = title
                    
                    if filtered_sitemap:
                        if self.verbose:
                            self.console.print(f"[green]MCPエージェント（JSON解析）で{len(filtered_sitemap)}個のリンクを抽出しました[/]")
                        return filtered_sitemap
            
            # フォールバック: URLパターンマッチング
            if self.verbose:
                self.console.print("[cyan]JSON解析が失敗したため、URLパターンマッチングを試行[/]")
            
            return self._extract_urls_from_text(agent_output)
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[red]MCPエージェント実行中にエラー: {e}[/]")
                self.console.print(f"[cyan]詳細なエラー情報: {type(e).__name__}: {str(e)}[/]")
            
            # エラー時は空のサイトマップを返す
            return {}

    def _extract_urls_from_text(self, text: str) -> Dict[str, str]:
        """
        テキストからURLを抽出してサイトマップを作成する
        
        Args:
            text: 解析対象のテキスト
            
        Returns:
            サイトマップ (URL -> タイトル のマッピング)
        """
        import re
        
        url_pattern = r'https?://[^\s"\'<>\)]*'
        urls = re.findall(url_pattern, text)
        
        if self.verbose:
            self.console.print(f"[cyan]出力から{len(urls)}個のURLを発見[/]")
        
        if urls:
            sitemap = {}
            base_netloc = urlparse(self.root_url).netloc
            valid_urls = []
            
            for url in urls:
                # URLをクリーンアップ
                url = url.rstrip('.,;!?')
                try:
                    parsed = urlparse(url)
                    if (parsed.netloc == base_netloc and 
                        self._should_include_url(url) and
                        not url.endswith(('.png', '.jpg', '.jpeg', '.gif', '.css', '.js', '.pdf'))):
                        valid_urls.append(url)
                except Exception:
                    continue
            
            if self.verbose:
                self.console.print(f"[cyan]有効なURL: {len(valid_urls)}個[/]")
            
            for url in set(valid_urls):  # 重複を除去
                # URLから簡単なタイトルを生成
                path = urlparse(url).path
                title = path.rstrip('/').split('/')[-1].replace('-', ' ').replace('_', ' ').title()
                if not title or title == "":
                    title = "Home"
                sitemap[url] = title
            
            if sitemap:
                if self.verbose:
                    self.console.print(f"[green]URLパターンマッチングで{len(sitemap)}個のリンクを抽出しました[/]")
                return sitemap
        
        # 結果が見つからない場合は空のサイトマップを返す
        if self.verbose:
            self.console.print("[yellow]MCPエージェントでリンクが見つかりませんでした。空のサイトマップを返します。[/]")
        return {}
    
    
    async def extract_content(self, page: Page) -> Dict[str, Any]:
        """
        ページからコンテンツを抽出する
        
        Args:
            page: Playwrightのページオブジェクト
            
        Returns:
            抽出したコンテンツの辞書
        """
        # メインコンテンツエリアのセレクタ
        main_selectors = [
            # 標準的なコンテンツエリア
            "main", 
            "article", 
            ".content", 
            "#content", 
            ".main-content",
            
            # ドキュメントサイト特有のセレクタ
            ".documentation",
            ".docs-content",
            ".markdown-body",
            ".mdx-content",
            ".prose",
            "[role=main]",
            ".main-docs-content",
            
            # MDXやGatsbyなどで一般的に使用されるクラス
            ".mdx-wrapper",
            ".gatsby-content",
            ".docs-wrapper",
            ".docs-container",
            
            # APIドキュメント特有のセレクタ
            ".api-content",
            ".api-documentation",
            ".reference-content",
            
            # フォールバック用の広範なセレクタ
            "[class*='content']",
            "[class*='documentation']",
            "[class*='markdown']"
        ]
        
        content = ""
        title = await page.title()
        url = page.url
        
        # メインコンテンツエリアの検出
        for selector in main_selectors:
            try:
                main_element = await page.query_selector(selector)
                if main_element:
                    content = await main_element.inner_text()
                    break
            except Exception:
                continue
        
        # メインコンテンツが見つからない場合、bodyから抽出
        if not content:
            try:
                body = await page.query_selector("body")
                if body:
                    content = await body.inner_text()
            except Exception as e:
                if self.verbose:
                    self.console.print(f"[yellow]警告: コンテンツ抽出中にエラーが発生しました: {e}[/]")
        
        return {
            "title": title,
            "url": url,
            "content": content,
        }
    async def summarize_page(self, url: str, task_id: Optional[TaskID] = None, progress: Optional[Progress] = None) -> Dict[str, Any]:
        """
        ページを要約する
        
        Args:
            url: ページのURL
            task_id: 進捗表示用のタスクID (オプション)
            progress: 進捗オブジェクト (オプション)
            
        Returns:
            要約情報
        """
        async with self.semaphore:
            # レート制限
            await asyncio.sleep(self.rate_limit)
            
            # 進捗表示の更新
            if progress and task_id:
                progress.update(task_id, description=f"処理中: {url}")
            
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    async with async_playwright() as p:
                        browser = await p.chromium.launch(headless=True)
                        page = await browser.new_page()
                        await page.goto(url, wait_until="networkidle")
                        
                        # コンテンツ抽出
                        page_data = await self.extract_content(page)
                        await browser.close()
                        
                        # 要約生成
                        if progress and task_id:
                            progress.update(task_id, description=f"要約生成中: {url}")
                        
                        # コンテンツが短すぎる場合はそのまま返す
                        if len(page_data["content"]) < 100:
                            summary = f"# {page_data['title']}\n\nこのページには十分なコンテンツがありません。"
                        else:
                            summary = await self.summary_chain.ainvoke({
                                "title": page_data["title"],
                                "url": url,
                                "content": page_data["content"]
                            })
                        
                        result = {
                            "title": page_data["title"],
                            "url": url,
                            "summary": summary
                        }
                        
                        if progress is not None and task_id is not None:
                            progress.update(task_id, description=f"完了: {url}", completed=True)
                        
                        return result
                
                except Exception as e:
                    retries += 1
                    if retries >= MAX_RETRIES:
                        if progress is not None and task_id is not None:
                            progress.update(task_id, description=f"エラー: {url}", completed=True)
                        return {
                            "title": f"エラー: {url}",
                            "url": url,
                            "summary": f"# エラー\n\nこのページの処理中にエラーが発生しました: {str(e)}"
                        }
                    else:
                        await asyncio.sleep(RETRY_DELAY)

    def _load_previous_sitemap(self) -> Optional[Dict[str, str]]:
        """前回のサイトマップを読み込む"""
        sitemap_path = self.output_dir / "sitemap.json"
        if sitemap_path.exists():
            try:
                with open(sitemap_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                if self.verbose:
                    self.console.print(f"[yellow]警告: 前回のサイトマップの読み込みに失敗しました: {e}[/]")
        return None

    def _should_rerun_all(self, previous_sitemap: Optional[Dict[str, str]]) -> bool:
        """すべてのページを再実行する必要があるかチェック"""
        if self.force_rerun:
            return True
        
        if not self.output_dir.exists():
            return True
            
        if previous_sitemap is None:
            return True
            
        # サイトマップの比較
        return previous_sitemap != self.site_map

    def _should_regenerate_final_summary(self) -> bool:
        """最終要約の再生成が必要かチェック"""
        final_md = self.output_dir / "final_summary.md"
        
        # ファイルが存在しない場合は再生成
        if not final_md.exists():
            if self.verbose:
                self.console.print("[yellow]最終要約ファイルが見つからないため、再生成します。[/]")
            return True
            
        # ページの要約ファイルが1つでもあれば再生成可能
        return any(self.output_dir.glob("page_*.md"))

    def _get_pending_pages(self, _: Optional[Dict[str, str]]) -> List[str]:
        """処理が必要なページのURLリストを取得"""
        pending_pages = []
        
        for url in self.site_map:
            page_exists = False
            # URLからファイル名のパターンを生成
            url_path = urlparse(url).path.replace('/', '_')
            
            # 対応するMarkdownファイルを探す
            for file in self.output_dir.glob(f"page_*{url_path}.md"):
                if file.exists():
                    page_exists = True
                    break
            
            if not page_exists:
                pending_pages.append(url)
        
        return pending_pages

    async def process_site(self):
        """サイト全体を処理する"""
        await self.initialize()
        
        # 出力ディレクトリの作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 最終要約の再生成が必要かチェック
        needs_final_summary = self._should_regenerate_final_summary()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:
            # サイトマップの取得
            sitemap_task = progress.add_task("[cyan]サイトマップを取得中...", total=1)
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                try:
                    await page.goto(self.root_url, wait_until="networkidle")
                    # ルートページのタイトルを取得
                    root_title = await page.title()
                    await browser.close()
                    
                    # MCPを使用してサイトマップを抽出（現在はフォールバック）
                    self.site_map = await self.extract_sitemap_with_mcp()
                    
                    # ルートページも追加
                    self.site_map[self.root_url] = root_title
                    
                    progress.update(sitemap_task, completed=True, description=f"サイトマップ取得完了: {len(self.site_map)}ページ")
                except Exception as e:
                    progress.update(sitemap_task, completed=True, description=f"[red]サイトマップ取得エラー: {e}[/]")
                    return
            
            # 前回のサイトマップを読み込み
            previous_sitemap = self._load_previous_sitemap()
            
            # 再実行が必要かチェック
            if self._should_rerun_all(previous_sitemap):
                pages_to_process = list(self.site_map.keys())
                if self.verbose:
                    reason = "強制再実行" if self.force_rerun else "サイトマップの変更" if previous_sitemap else "初回実行"
                    self.console.print(f"[yellow]すべてのページを処理します（理由: {reason}）[/]")
            else:
                pages_to_process = self._get_pending_pages(previous_sitemap)
                if self.verbose:
                    self.console.print(f"[yellow]未処理の{len(pages_to_process)}ページを処理します[/]")
            
            if not pages_to_process and not needs_final_summary:
                self.console.print("[green]すべてのページは既に処理済みです。[/]")
                return
            
            # ページの処理が必要な場合のみ
            if pages_to_process:
                # サイトマップを保存（新しいバージョンで更新）
                sitemap_path = self.output_dir / "sitemap.json"
                with open(sitemap_path, "w", encoding="utf-8") as f:
                    json.dump(self.site_map, f, ensure_ascii=False, indent=2)
                
                # 各ページの処理
                pages_task = progress.add_task("[cyan]ページを処理中...", total=len(pages_to_process))
                tasks = []
                
                for url in pages_to_process:
                    page_task = progress.add_task(f"待機中: {url}", total=1, visible=True)
                    task = asyncio.create_task(self.summarize_page(url, page_task, progress))
                    tasks.append((task, page_task))
                    progress.update(pages_task, advance=0)
                
                # すべてのタスクを実行
                for i, (task, page_task) in enumerate(tasks):
                    result = await task
                    
                    # 要約を保存
                    if result:
                        self.summaries[result["url"]] = result
                        
                        # ファイル名の生成（URLから安全なファイル名に変換）
                        file_name = f"page_{i+1}_{urlparse(result['url']).path.replace('/', '_')}.md"
                        file_name = file_name.replace(':', '_').replace('?', '_').replace('&', '_')
                        file_path = self.output_dir / file_name
                        
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(result["summary"])
                    
                    progress.update(pages_task, advance=1)
            
            # 必要な場合は最終要約を生成
            if pages_to_process or needs_final_summary:
                final_task = progress.add_task("[cyan]最終要約を生成中...", total=1)
                
                try:
                    await self.generate_final_summary()
                    progress.update(final_task, completed=True, description="最終要約の生成完了")
                except Exception as e:
                    progress.update(final_task, completed=True, description=f"[red]最終要約の生成エラー: {e}[/]")
                    raise
        
        # 完了
        self.console.print(f"[green]処理完了！[/] 出力ディレクトリ: {self.output_dir}")
        sitemap_path = self.output_dir / "sitemap.json"
        if sitemap_path.exists():
            self.console.print(f"サイトマップ: {sitemap_path}")
        self.console.print(f"最終要約: {self.output_dir}/final_summary.md")
    
    def _build_site_tree(self, sitemap: Dict[str, str]) -> Dict[str, Any]:
        """
        サイトマップからツリー構造を構築する
        
        Args:
            sitemap: URL -> タイトルのマッピング
            
        Returns:
            ツリー構造の辞書
        """
        tree = {}
        for url, title in sitemap.items():
            path_parts = urlparse(url).path.strip('/').split('/')
            current = tree
            
            # パスの各部分をツリーに追加
            for i, part in enumerate(path_parts):
                if not part:
                    continue
                if part not in current:
                    current[part] = {
                        'title': title if i == len(path_parts) - 1 else part.replace('-', ' ').title(),
                        'url': url if i == len(path_parts) - 1 else None,
                        'part': part,
                        'children': {},
                        'content': None,
                        'summary': None
                    }
                current = current[part]['children']
        
        return tree
    
    async def _summarize_tree_node(self, node: Dict[str, Any], parent_path: str = "") -> str:
        """
        ツリーノードを再帰的に要約する
        
        Args:
            node: ツリーノード
            parent_path: 親ノードまでのパス
            
        Returns:
            ノードの要約
        """
        # リーフノードの場合、コンテンツを読み込む
        if node['url'] and not node['children']:
            md_files = list(self.output_dir.glob(f"page_*{urlparse(node['url']).path.replace('/', '_')}.md"))
            if md_files:
                with open(md_files[0], "r", encoding="utf-8") as f:
                    node['content'] = f.read()
                    return node['content']
            return ""
        
        # 子ノードの要約を収集
        children_summaries = []
        for _, child_node in node['children'].items():
            child_path = f"{parent_path}/{node['part']}" if parent_path else node['part']
            child_summary = await self._summarize_tree_node(child_node, child_path)
            if child_summary:
                children_summaries.append(child_summary)
        
        if not children_summaries:
            return ""
        
        # このノードの要約を生成
        section_chain = ChatPromptTemplate.from_template(SECTION_SUMMARY_TEMPLATE) | self.llm | StrOutputParser()
        try:
            # サマリーをJSON形式に変換
            summaries_json = json.dumps({
                "summaries": children_summaries
            }, ensure_ascii=False)
            
            node['summary'] = await section_chain.ainvoke({
                "section_name": node['title'],
                "summaries": summaries_json
            })
            
            # ノードの要約をファイルに保存
            if parent_path:
                # 親パスがある場合はそれを含めたファイル名を生成
                file_name = f"node_{parent_path}_{node['title']}_summary.md".lower()
            else:
                file_name = f"node_{node['title']}_summary.md".lower()
                
            # ファイル名の特殊文字を置換
            file_name = file_name.replace('/', '_').replace('|', '_').replace(':', '_').replace('?', '_').replace('&', '_').replace(' ', '_')
            file_path = self.output_dir / file_name
            
            # 要約を保存
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(node['summary'])
            
            if self.verbose:
                self.console.print(f"[green]ノード要約を保存: {file_path}[/]")
            
            return node['summary']
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]警告: セクション {node['title']} の要約生成中にエラー: {e}[/]")
            return "\n\n".join(children_summaries)

    async def generate_final_summary(self):
        """最終要約のみを生成する"""
        await self.initialize()
        
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        site_name = urlparse(self.root_url).netloc
        
        # サイトマップを読み込み
        sitemap_path = self.output_dir / "sitemap.json"
        if not sitemap_path.exists():
            raise ValueError("サイトマップが見つかりません。先に全ページの要約を生成してください。")
        
        with open(sitemap_path, "r", encoding="utf-8") as f:
            sitemap = json.load(f)
        
        # サイトのツリー構造を構築
        site_tree = self._build_site_tree(sitemap)

        try:
            # ツリー構造に基づいて再帰的に要約を生成
            all_sections = []
            for section_name, section_node in site_tree.items():
                section_summary = await self._summarize_tree_node(section_node)
                if section_summary:
                    # セクション要約をファイルに保存
                    section_filename = f"section_{section_name.lower()}_summary.md"
                    section_path = self.output_dir / section_filename
                    with open(section_path, "w", encoding="utf-8") as f:
                        f.write(section_summary)
                    
                    all_sections.append(f"## {section_node['title']}\n{section_summary}")
            
            # 最終要約を生成（セクションサマリーをJSONで渡す）
            section_summaries_json = json.dumps({
                "sections": all_sections
            }, ensure_ascii=False)
            
            final_summary = await self.final_summary_chain.ainvoke({
                "site_name": site_name,
                "section_summaries": section_summaries_json
            })
            
            # 最終要約を保存
            final_path = self.output_dir / "final_summary.md"
            with open(final_path, "w", encoding="utf-8") as f:
                f.write(final_summary)
            
            self.console.print("[green]最終要約の生成完了！[/]")
            self.console.print(f"最終要約: {final_path}")
            
        except Exception as e:
            raise Exception(f"最終要約の生成中にエラーが発生しました: {e}")

    def run(self):
        """メインの実行メソッド"""
        asyncio.run(self.process_site())

    def run_final_summary(self):
        """最終要約のみを生成するメソッド"""
        asyncio.run(self.generate_final_summary())

    def run_mcp_debug(self):
        """MCPツールのデバッグテストを実行するメソッド"""
        asyncio.run(self._run_mcp_debug_async())
    
    async def _run_mcp_debug_async(self):
        """MCPデバッグテストの非同期実行"""
        await self.initialize()
        await self.debug_mcp_tools()


def main():
    """コマンドラインインターフェース"""
    parser = argparse.ArgumentParser(description="Shallow Research - ウェブサイトの調査と要約を自動化")
    parser.add_argument("url", help="調査対象のURL")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_DIR, help=f"出力ディレクトリ (デフォルト: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("-c", "--concurrency", type=int, default=DEFAULT_CONCURRENCY, help=f"同時実行数 (デフォルト: {DEFAULT_CONCURRENCY})")
    parser.add_argument("-r", "--rate-limit", type=float, default=DEFAULT_RATE_LIMIT, help=f"リクエスト間の秒数 (デフォルト: {DEFAULT_RATE_LIMIT})")
    parser.add_argument("-m", "--model", default=None, help="使用するLLMモデル（指定がない場合はプロバイダのデフォルトモデル）")
    parser.add_argument("-p", "--provider", default=None, help="LLMプロバイダ")
    parser.add_argument("-k", "--api-key", help="API Key (環境変数で設定可能)")
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細出力モード")
    parser.add_argument("-f", "--force", action="store_true", help="すべてのページを強制的に再実行")
    parser.add_argument("--final-only", action="store_true", help="最終要約のみを生成")
    parser.add_argument("--mcp-debug", action="store_true", help="MCPツールのデバッグテストのみを実行")
    parser.add_argument("--restrict-path", action="store_true", default=True, help="ルートURLのパスに基づいてURLを制限する")
    
    # MCP関連のオプション
    parser.add_argument("--mcp-server-url", help="MCPサーバーのコマンド（デフォルト: npx @playwright/mcp@latest --headless）")
    
    # OpenAI互換サービスのオプション
    parser.add_argument("--api-base", help="OpenAI互換サービスのベースURL")
    parser.add_argument("--api-version", help="Azure OpenAIのAPIバージョン")
    parser.add_argument("--deployment-name", help="Azure OpenAIのデプロイメント名")

    args = parser.parse_args()
    
    # OpenAI互換サービスの追加設定
    kwargs = {}
    if args.api_base:
        kwargs["api_base"] = args.api_base
    if args.api_version:
        kwargs["api_version"] = args.api_version
    if args.deployment_name:
        kwargs["deployment_name"] = args.deployment_name
    
    # 研究オブジェクトの作成
    researcher = ShallowResearcher(
        url=args.url,
        output_dir=args.output,
        concurrency=args.concurrency,
        rate_limit=args.rate_limit,
        llm_provider=args.provider,
        llm_model=args.model,
        api_key=args.api_key,
        verbose=args.verbose,
        force_rerun=args.force,
        restrict_path=args.restrict_path,
        mcp_server_url=args.mcp_server_url,
        **kwargs  # OpenAI互換サービスの設定を追加
    )
    
    try:
        if args.final_only:
            researcher.run_final_summary()
        elif args.mcp_debug:
            researcher.run_mcp_debug()
        else:
            researcher.run()
    except KeyboardInterrupt:
        print("\n処理を中断しました。")
        sys.exit(1)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
