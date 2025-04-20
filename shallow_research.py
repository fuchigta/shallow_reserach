#!/usr/bin/env python3
"""
Shallow Research - ウェブサイトの構造化されたリサーチを自動化するツール
"""

import os
import sys
import argparse
import time
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from urllib.parse import urljoin, urlparse

import markdown
from playwright.async_api import async_playwright, Page
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
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
    "google": "gemini-2.0-flash",
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
        """
        self.root_url = url
        self.output_dir = Path(output_dir)
        self.concurrency = concurrency
        self.rate_limit = rate_limit
        self.verbose = verbose
        self.force_rerun = force_rerun
        self.restrict_path = restrict_path  # パス制限設定を保存
        self.root_path = urlparse(url).path  # ルートURLのパスを保存
        self.site_map = {}
        self.summaries = {}
        self.visited = set()
        self.semaphore = None  # asyncioで初期化
        
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
    
    async def initialize(self):
        """非同期リソースの初期化"""
        self.semaphore = asyncio.Semaphore(self.concurrency)
    
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

    async def extract_sitemap(self, page: Page) -> Dict[str, str]:
        """
        ページからサイトマップを抽出する
        
        Args:
            page: Playwrightのページオブジェクト
            
        Returns:
            サイトマップ (URL -> タイトル のマッピング)
        """
        # サイドバーなどのナビゲーション要素を探す
        nav_selectors = [
            # 標準的なナビゲーション要素
            "nav", 
            ".sidebar",
            ".navigation",
            ".menu",
            "ul.nav",
            "[role=navigation]",
            "#sidebar",
            ".toc",
            
            # ドキュメントサイト特有のセレクタ
            ".docs-sidebar",
            ".docs-navigation",
            ".documentation-nav",
            ".table-of-contents",
            "[role=doc-toc]",
            "[role=complementary]",
            ".docusaurus-sidebar",
            ".gatsby-sidebar",
            ".nextjs-sidebar",
            
            # MDXやGatsbyなどで一般的に使用されるクラス
            ".mdx-sidebar",
            ".mdx-nav",
            ".gatsby-nav",
            
            # 一般的なサイドバーの構造を持つ要素
            "aside",
            "aside a",
            ".left-sidebar",
            ".right-sidebar",
            ".side-nav",
            ".side-menu",
            
            # より広範な検索のためのフォールバック
            "[class*='sidebar']",
            "[class*='navigation']",
            "[class*='menu']",
            "[class*='toc']",
            "main nav",
            "header nav"
        ]
        
        sitemap = {}
        base_url = self.root_url
        
        for selector in nav_selectors:
            try:
                nav_elements = await page.query_selector_all(f"{selector} a")
                if nav_elements:
                    for elem in nav_elements:
                        href = await elem.get_attribute("href")
                        if href and not href.startswith("#") and not href.startswith("javascript:"):
                            # 相対URLを絶対URLに変換
                            abs_url = urljoin(base_url, href)
                            # 同じドメイン内のURLかつパス制限に合致するURLのみ対象とする
                            if (urlparse(abs_url).netloc == urlparse(base_url).netloc and
                                self._should_include_url(abs_url)):
                                title = await elem.text_content()
                                title = title.strip() if title else "No Title"
                                sitemap[abs_url] = title
            except Exception as e:
                if self.verbose:
                    self.console.print(f"[yellow]警告: ナビゲーション要素 {selector} の抽出中にエラーが発生しました: {e}[/]")
        
        # サイトマップが空の場合、ページ内のすべてのリンクを取得するフォールバック
        if not sitemap:
            try:
                # まずヘッダーやメインコンテンツ内のリンクを探す
                main_link_selectors = [
                    "header a",
                    "main a",
                    "article a",
                    ".content a",
                    "#content a",
                    ".main-content a",
                    ".markdown-body a",
                    ".prose a",
                    "[role=main] a"
                ]
                
                for selector in main_link_selectors:
                    links = await page.query_selector_all(selector)
                    for link in links:
                        href = await link.get_attribute("href")
                        if href and not href.startswith("#") and not href.startswith("javascript:"):
                            abs_url = urljoin(base_url, href)
                            if urlparse(abs_url).netloc == urlparse(base_url).netloc:
                                title = await link.text_content()
                                title = title.strip() if title else "No Title"
                                sitemap[abs_url] = title
                
                # まだリンクが見つからない場合は、すべてのリンクを対象にする
                if not sitemap:
                    all_links = await page.query_selector_all("a")
                    for link in all_links:
                        href = await link.get_attribute("href")
                        if href and not href.startswith("#") and not href.startswith("javascript:"):
                            abs_url = urljoin(base_url, href)
                            if urlparse(abs_url).netloc == urlparse(base_url).netloc:
                                # タイトルの抽出を改善
                                title = await link.evaluate("el => el.getAttribute('title') || el.textContent")
                                title = title.strip() if title else "No Title"
                                
                                # URLのパスからタイトルを推測（タイトルが空の場合）
                                if title == "No Title":
                                    path = urlparse(abs_url).path
                                    if path:
                                        # パスの最後の部分を取得し、ハイフンやアンダースコアを空白に変換
                                        path_title = path.rstrip('/').split('/')[-1]
                                        path_title = path_title.replace('-', ' ').replace('_', ' ')
                                        title = path_title
                                
                                sitemap[abs_url] = title
            except Exception as e:
                if self.verbose:
                    self.console.print(f"[yellow]警告: リンク抽出中にエラーが発生しました: {e}[/]")
        
        return sitemap
    
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
        final_html = self.output_dir / "final_summary.html"
        
        # いずれかのファイルが存在しない場合は再生成
        if not final_md.exists() or not final_html.exists():
            if self.verbose:
                self.console.print("[yellow]最終要約ファイルが見つからないため、再生成します。[/]")
            return True
            
        # ページの要約ファイルが1つでもあれば再生成可能
        has_page_summaries = False
        for md_file in self.output_dir.glob("page_*.md"):
            has_page_summaries = True
            break
            
        return has_page_summaries

    def _get_pending_pages(self, previous_sitemap: Optional[Dict[str, str]]) -> List[str]:
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
                    self.site_map = await self.extract_sitemap(page)
                    
                    # ルートページも追加
                    root_title = await page.title()
                    self.site_map[self.root_url] = root_title
                    
                    progress.update(sitemap_task, completed=True, description=f"サイトマップ取得完了: {len(self.site_map)}ページ")
                except Exception as e:
                    progress.update(sitemap_task, completed=True, description=f"[red]サイトマップ取得エラー: {e}[/]")
                    await browser.close()
                    return
                
                await browser.close()
            
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
        self.console.print(f"HTML版: {self.output_dir}/final_summary.html")
    
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
        for child_name, child_node in node['children'].items():
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
            
            # HTMLバージョンを生成
            html_content = markdown.markdown(final_summary)
            html_path = self.output_dir / "final_summary.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{site_name} - 調査レポート</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; padding: 1em; max-width: 900px; margin: 0 auto; color: #333; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        code {{ background-color: #f5f5f5; padding: 0.2em 0.4em; border-radius: 3px; font-family: monospace; }}
        pre {{ background-color: #f5f5f5; padding: 1em; border-radius: 5px; overflow-x: auto; }}
        a {{ color: #3498db; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>{site_name} - 調査レポート</h1>
    {html_content}
    <hr>
    <footer>
        <p>生成日時: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>調査対象URL: <a href="{self.root_url}">{self.root_url}</a></p>
    </footer>
</body>
</html>""")
            
            self.console.print("[green]最終要約の生成完了！[/]")
            self.console.print(f"最終要約: {final_path}")
            self.console.print(f"HTML版: {html_path}")
            
        except Exception as e:
            raise Exception(f"最終要約の生成中にエラーが発生しました: {e}")

    def run(self):
        """メインの実行メソッド"""
        asyncio.run(self.process_site())

    def run_final_summary(self):
        """最終要約のみを生成するメソッド"""
        asyncio.run(self.generate_final_summary())


def main():
    """コマンドラインインターフェース"""
    parser = argparse.ArgumentParser(description="Shallow Research - ウェブサイトの調査と要約を自動化")
    parser.add_argument("url", help="調査対象のURL")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_DIR, help=f"出力ディレクトリ (デフォルト: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("-c", "--concurrency", type=int, default=DEFAULT_CONCURRENCY, help=f"同時実行数 (デフォルト: {DEFAULT_CONCURRENCY})")
    parser.add_argument("-r", "--rate-limit", type=float, default=DEFAULT_RATE_LIMIT, help=f"リクエスト間の秒数 (デフォルト: {DEFAULT_RATE_LIMIT})")
    parser.add_argument("-m", "--model", default=None, help="使用するLLMモデル（指定がない場合はプロバイダのデフォルトモデル）")
    parser.add_argument("-p", "--provider", default=LLM_PROVIDERS[0], help=f"LLMプロバイダ (デフォルト: {LLM_PROVIDERS[0]})")
    parser.add_argument("-k", "--api-key", help="API Key (環境変数で設定可能)")
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細出力モード")
    parser.add_argument("-f", "--force", action="store_true", help="すべてのページを強制的に再実行")
    parser.add_argument("--final-only", action="store_true", help="最終要約のみを生成")
    parser.add_argument("--restrict-path", action="store_true", default=True, help="ルートURLのパスに基づいてURLを制限する")
    
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
        **kwargs  # OpenAI互換サービスの設定を追加
    )
    
    try:
        if args.final_only:
            researcher.run_final_summary()
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
