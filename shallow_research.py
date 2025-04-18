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
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse

import markdown
from playwright.async_api import async_playwright, Page, Browser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.markdown import Markdown

from dotenv import load_dotenv
load_dotenv()

# 定数
DEFAULT_OUTPUT_DIR = "research_output"
DEFAULT_CONCURRENCY = 3
DEFAULT_RATE_LIMIT = 1  # リクエスト間の秒数
MAX_RETRIES = 3
RETRY_DELAY = 2
SUMMARY_TEMPLATE = """
以下のウェブページの内容を要約してください。重要なポイント、主要な概念、コード例があれば含めてください。
マークダウン形式で返してください。

ページのタイトル: {title}
ページのURL: {url}

ページの内容:
{content}

要約 (マークダウン形式):
"""

FINAL_SUMMARY_TEMPLATE = """
以下は{site_name}のドキュメントから抽出した要約です。これらの情報を基に、全体的な概要をマークダウン形式で作成してください。
主要な概念、重要なポイント、関連性を強調し、読者が{site_name}の基本を理解できるようにしてください。

{summaries}

全体要約 (マークダウン形式):
"""


class ShallowResearcher:
    """ウェブサイトを自動的にクロールして要約するクラス"""
    
    def __init__(
        self,
        url: str,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        concurrency: int = DEFAULT_CONCURRENCY,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        llm_model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        verbose: bool = False
    ):
        """
        初期化メソッド
        
        Args:
            url: 調査対象のルートURL
            output_dir: 出力ディレクトリ
            concurrency: 同時実行数
            rate_limit: リクエスト間の秒数
            llm_model: 使用するLLMモデル
            api_key: LLM APIキー
            verbose: 詳細出力モード
        """
        self.root_url = url
        self.output_dir = Path(output_dir)
        self.concurrency = concurrency
        self.rate_limit = rate_limit
        self.verbose = verbose
        self.site_map = {}
        self.summaries = {}
        self.visited = set()
        self.semaphore = None  # asyncioで初期化
        
        # コンソール出力
        self.console = Console()
        
        # LLMの設定
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google APIキーが必要です。環境変数GOOGLE_API_KEYを設定するか、--api-keyオプションで指定してください。")
        
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=api_key,
            temperature=0.2,
            max_output_tokens=4096,
        )
        
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
            "nav", 
            ".sidebar", 
            ".navigation", 
            ".menu", 
            "ul.nav", 
            "[role=navigation]",
            "#sidebar",
            ".toc"
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
                            # 同じドメイン内のURLのみ対象とする
                            if urlparse(abs_url).netloc == urlparse(base_url).netloc:
                                title = await elem.text_content()
                                title = title.strip() if title else "No Title"
                                sitemap[abs_url] = title
            except Exception as e:
                if self.verbose:
                    self.console.print(f"[yellow]警告: ナビゲーション要素 {selector} の抽出中にエラーが発生しました: {e}[/]")
        
        # サイトマップが空の場合、ページ内のすべてのリンクを取得するフォールバック
        if not sitemap:
            try:
                all_links = await page.query_selector_all("a")
                for link in all_links:
                    href = await link.get_attribute("href")
                    if href and not href.startswith("#") and not href.startswith("javascript:"):
                        abs_url = urljoin(base_url, href)
                        if urlparse(abs_url).netloc == urlparse(base_url).netloc:
                            title = await link.text_content()
                            title = title.strip() if title else "No Title"
                            sitemap[abs_url] = title
            except Exception as e:
                if self.verbose:
                    self.console.print(f"[yellow]警告: すべてのリンク抽出中にエラーが発生しました: {e}[/]")
        
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
            "main", 
            "article", 
            ".content", 
            "#content", 
            ".main-content",
            ".documentation",
            ".docs-content",
            ".markdown-body"
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
        
        # コードブロックを探す (可能であれば)
        code_blocks = []
        try:
            code_elements = await page.query_selector_all("pre code, pre, .code-block")
            for code_elem in code_elements:
                code_text = await code_elem.inner_text()
                if code_text:
                    code_blocks.append(code_text)
        except Exception:
            pass
        
        return {
            "title": title,
            "url": url,
            "content": content,
            "code_blocks": code_blocks
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
                        
                        # コードブロックがあれば追加
                        if page_data["code_blocks"]:
                            code_section = "\n\n## コード例\n\n"
                            for i, code in enumerate(page_data["code_blocks"][:3]):  # 最大3つまで
                                code_section += f"```\n{code[:500]}...\n```\n\n"  # 長すぎる場合は切り詰める
                            summary += code_section
                        
                        result = {
                            "title": page_data["title"],
                            "url": url,
                            "summary": summary
                        }
                        
                        if progress and task_id:
                            progress.update(task_id, description=f"完了: {url}", completed=True)
                        
                        return result
                
                except Exception as e:
                    retries += 1
                    if retries >= MAX_RETRIES:
                        if progress and task_id:
                            progress.update(task_id, description=f"エラー: {url}", completed=True)
                        return {
                            "title": f"エラー: {url}",
                            "url": url,
                            "summary": f"# エラー\n\nこのページの処理中にエラーが発生しました: {str(e)}"
                        }
                    else:
                        await asyncio.sleep(RETRY_DELAY)
    
    async def process_site(self):
        """サイト全体を処理する"""
        await self.initialize()
        
        # 出力ディレクトリの作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # サイト名の取得（URLのドメイン部分）
        site_name = urlparse(self.root_url).netloc
        
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
            
            # サイトマップが空の場合
            if not self.site_map:
                self.console.print("[red]エラー: サイトマップを抽出できませんでした。URLを確認して再試行してください。[/]")
                return
            
            # サイトマップを保存
            sitemap_path = self.output_dir / "sitemap.json"
            with open(sitemap_path, "w", encoding="utf-8") as f:
                json.dump(self.site_map, f, ensure_ascii=False, indent=2)
            
            # 各ページの処理
            pages_task = progress.add_task("[cyan]ページを処理中...", total=len(self.site_map))
            tasks = []
            
            for url in self.site_map:
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
            
            # 最終要約の作成
            final_task = progress.add_task("[cyan]最終要約を作成中...", total=1)
            
            # すべての要約を連結
            all_summaries = ""
            for url, summary_data in self.summaries.items():
                title = summary_data["title"]
                url_summary = summary_data["summary"]
                all_summaries += f"## {title}\n\n{url_summary}\n\n---\n\n"
            
            try:
                final_summary = await self.final_summary_chain.ainvoke({
                    "site_name": site_name,
                    "summaries": all_summaries
                })
                
                # 最終要約を保存
                final_path = self.output_dir / "final_summary.md"
                with open(final_path, "w", encoding="utf-8") as f:
                    f.write(final_summary)
                
                progress.update(final_task, completed=True, description="最終要約の作成完了")
                
                # HTMLバージョンも作成
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
                
            except Exception as e:
                progress.update(final_task, completed=True, description=f"[red]最終要約の作成エラー: {e}[/]")
        
        # 完了
        self.console.print(f"[green]処理完了！[/] 出力ディレクトリ: {self.output_dir}")
        self.console.print(f"サイトマップ: {sitemap_path}")
        self.console.print(f"最終要約: {self.output_dir}/final_summary.md")
        self.console.print(f"HTML版: {self.output_dir}/final_summary.html")
    
    def run(self):
        """メインの実行メソッド"""
        asyncio.run(self.process_site())


def main():
    """コマンドラインインターフェース"""
    parser = argparse.ArgumentParser(description="Shallow Research - ウェブサイトの調査と要約を自動化")
    parser.add_argument("url", help="調査対象のURL")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_DIR, help=f"出力ディレクトリ (デフォルト: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("-c", "--concurrency", type=int, default=DEFAULT_CONCURRENCY, help=f"同時実行数 (デフォルト: {DEFAULT_CONCURRENCY})")
    parser.add_argument("-r", "--rate-limit", type=float, default=DEFAULT_RATE_LIMIT, help=f"リクエスト間の秒数 (デフォルト: {DEFAULT_RATE_LIMIT})")
    parser.add_argument("-m", "--model", default="gemini-1.5-flash", help="使用するLLMモデル (デフォルト: gemini-1.5-flash)")
    parser.add_argument("-k", "--api-key", help="Google API Key (環境変数 GOOGLE_API_KEY でも設定可能)")
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細出力モード")
    
    args = parser.parse_args()
    
    # 研究オブジェクトの作成と実行
    researcher = ShallowResearcher(
        url=args.url,
        output_dir=args.output,
        concurrency=args.concurrency,
        rate_limit=args.rate_limit,
        llm_model=args.model,
        api_key=args.api_key,
        verbose=args.verbose
    )
    
    try:
        researcher.run()
    except KeyboardInterrupt:
        print("\n処理を中断しました。")
        sys.exit(1)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
