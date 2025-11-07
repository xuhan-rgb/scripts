#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
博客管理系统
支持文章的创建、编辑、删除、查看、搜索等功能
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional
import argparse


class BlogManager:
    """博客管理核心类"""

    def __init__(self, data_file: str = "blog_data.json"):
        """初始化博客管理器

        Args:
            data_file: 数据存储文件路径
        """
        self.data_file = data_file
        self.data = self._load_data()

    def _load_data(self) -> Dict:
        """从文件加载数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"警告: {self.data_file} 文件损坏，将创建新数据")
                return self._init_data()
        return self._init_data()

    def _init_data(self) -> Dict:
        """初始化数据结构"""
        return {
            "posts": [],
            "categories": [],
            "tags": [],
            "config": {
                "author": "博主",
                "blog_name": "我的博客",
                "created_at": datetime.now().isoformat()
            }
        }

    def _save_data(self):
        """保存数据到文件"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"错误: 保存数据失败 - {e}")
            sys.exit(1)

    def _generate_id(self) -> int:
        """生成新的文章ID"""
        if not self.data["posts"]:
            return 1
        return max(post["id"] for post in self.data["posts"]) + 1

    def create_post(self, title: str, content: str, category: str = "",
                   tags: List[str] = None) -> Dict:
        """创建新文章

        Args:
            title: 文章标题
            content: 文章内容
            category: 文章分类
            tags: 文章标签列表

        Returns:
            创建的文章字典
        """
        if tags is None:
            tags = []

        post = {
            "id": self._generate_id(),
            "title": title,
            "content": content,
            "category": category,
            "tags": tags,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "published"
        }

        self.data["posts"].append(post)

        # 更新分类和标签
        if category and category not in self.data["categories"]:
            self.data["categories"].append(category)

        for tag in tags:
            if tag not in self.data["tags"]:
                self.data["tags"].append(tag)

        self._save_data()
        return post

    def get_post(self, post_id: int) -> Optional[Dict]:
        """根据ID获取文章

        Args:
            post_id: 文章ID

        Returns:
            文章字典，如果不存在返回None
        """
        for post in self.data["posts"]:
            if post["id"] == post_id:
                return post
        return None

    def update_post(self, post_id: int, title: str = None, content: str = None,
                   category: str = None, tags: List[str] = None) -> bool:
        """更新文章

        Args:
            post_id: 文章ID
            title: 新标题（可选）
            content: 新内容（可选）
            category: 新分类（可选）
            tags: 新标签列表（可选）

        Returns:
            是否更新成功
        """
        post = self.get_post(post_id)
        if not post:
            return False

        if title is not None:
            post["title"] = title
        if content is not None:
            post["content"] = content
        if category is not None:
            post["category"] = category
            if category and category not in self.data["categories"]:
                self.data["categories"].append(category)
        if tags is not None:
            post["tags"] = tags
            for tag in tags:
                if tag not in self.data["tags"]:
                    self.data["tags"].append(tag)

        post["updated_at"] = datetime.now().isoformat()
        self._save_data()
        return True

    def delete_post(self, post_id: int) -> bool:
        """删除文章

        Args:
            post_id: 文章ID

        Returns:
            是否删除成功
        """
        for i, post in enumerate(self.data["posts"]):
            if post["id"] == post_id:
                self.data["posts"].pop(i)
                self._save_data()
                return True
        return False

    def list_posts(self, category: str = None, tag: str = None,
                  limit: int = None) -> List[Dict]:
        """列出文章

        Args:
            category: 按分类筛选（可选）
            tag: 按标签筛选（可选）
            limit: 限制返回数量（可选）

        Returns:
            文章列表
        """
        posts = self.data["posts"]

        if category:
            posts = [p for p in posts if p["category"] == category]

        if tag:
            posts = [p for p in posts if tag in p["tags"]]

        # 按创建时间降序排序
        posts = sorted(posts, key=lambda x: x["created_at"], reverse=True)

        if limit:
            posts = posts[:limit]

        return posts

    def search_posts(self, keyword: str) -> List[Dict]:
        """搜索文章

        Args:
            keyword: 搜索关键词

        Returns:
            匹配的文章列表
        """
        keyword = keyword.lower()
        results = []

        for post in self.data["posts"]:
            if (keyword in post["title"].lower() or
                keyword in post["content"].lower() or
                keyword in post.get("category", "").lower() or
                any(keyword in tag.lower() for tag in post.get("tags", []))):
                results.append(post)

        return results

    def get_categories(self) -> List[str]:
        """获取所有分类"""
        return self.data["categories"]

    def get_tags(self) -> List[str]:
        """获取所有标签"""
        return self.data["tags"]

    def get_stats(self) -> Dict:
        """获取博客统计信息

        Returns:
            统计信息字典
        """
        return {
            "total_posts": len(self.data["posts"]),
            "total_categories": len(self.data["categories"]),
            "total_tags": len(self.data["tags"]),
            "blog_name": self.data["config"]["blog_name"],
            "author": self.data["config"]["author"]
        }


class BlogCLI:
    """博客命令行界面"""

    def __init__(self):
        self.blog = BlogManager()

    def run(self):
        """运行CLI"""
        parser = argparse.ArgumentParser(description="博客管理系统")
        subparsers = parser.add_subparsers(dest='command', help='可用命令')

        # 创建文章
        create_parser = subparsers.add_parser('create', help='创建新文章')
        create_parser.add_argument('title', help='文章标题')
        create_parser.add_argument('-c', '--content', help='文章内容', default='')
        create_parser.add_argument('-f', '--file', help='从文件读取内容')
        create_parser.add_argument('--category', help='文章分类', default='')
        create_parser.add_argument('--tags', help='文章标签（逗号分隔）', default='')

        # 列出文章
        list_parser = subparsers.add_parser('list', help='列出文章')
        list_parser.add_argument('--category', help='按分类筛选')
        list_parser.add_argument('--tag', help='按标签筛选')
        list_parser.add_argument('--limit', type=int, help='限制数量')

        # 查看文章
        view_parser = subparsers.add_parser('view', help='查看文章')
        view_parser.add_argument('id', type=int, help='文章ID')

        # 编辑文章
        edit_parser = subparsers.add_parser('edit', help='编辑文章')
        edit_parser.add_argument('id', type=int, help='文章ID')
        edit_parser.add_argument('--title', help='新标题')
        edit_parser.add_argument('--content', help='新内容')
        edit_parser.add_argument('-f', '--file', help='从文件读取新内容')
        edit_parser.add_argument('--category', help='新分类')
        edit_parser.add_argument('--tags', help='新标签（逗号分隔）')

        # 删除文章
        delete_parser = subparsers.add_parser('delete', help='删除文章')
        delete_parser.add_argument('id', type=int, help='文章ID')

        # 搜索文章
        search_parser = subparsers.add_parser('search', help='搜索文章')
        search_parser.add_argument('keyword', help='搜索关键词')

        # 统计信息
        subparsers.add_parser('stats', help='显示统计信息')

        # 导出文章
        export_parser = subparsers.add_parser('export', help='导出文章')
        export_parser.add_argument('id', type=int, help='文章ID')
        export_parser.add_argument('-o', '--output', help='输出文件', required=True)
        export_parser.add_argument('--format', choices=['md', 'html'],
                                  default='md', help='导出格式')

        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return

        # 执行命令
        if args.command == 'create':
            self._create_post(args)
        elif args.command == 'list':
            self._list_posts(args)
        elif args.command == 'view':
            self._view_post(args)
        elif args.command == 'edit':
            self._edit_post(args)
        elif args.command == 'delete':
            self._delete_post(args)
        elif args.command == 'search':
            self._search_posts(args)
        elif args.command == 'stats':
            self._show_stats()
        elif args.command == 'export':
            self._export_post(args)

    def _create_post(self, args):
        """创建文章命令处理"""
        content = args.content

        if args.file:
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"错误: 无法读取文件 {args.file} - {e}")
                return

        tags = [tag.strip() for tag in args.tags.split(',') if tag.strip()]

        post = self.blog.create_post(
            title=args.title,
            content=content,
            category=args.category,
            tags=tags
        )

        print(f"✓ 文章创建成功！")
        print(f"  ID: {post['id']}")
        print(f"  标题: {post['title']}")
        print(f"  分类: {post['category'] or '无'}")
        print(f"  标签: {', '.join(post['tags']) or '无'}")

    def _list_posts(self, args):
        """列出文章命令处理"""
        posts = self.blog.list_posts(
            category=args.category,
            tag=args.tag,
            limit=args.limit
        )

        if not posts:
            print("没有找到文章")
            return

        print(f"\n共找到 {len(posts)} 篇文章:\n")
        print("=" * 80)

        for post in posts:
            print(f"ID: {post['id']}")
            print(f"标题: {post['title']}")
            print(f"分类: {post['category'] or '无'}")
            print(f"标签: {', '.join(post['tags']) or '无'}")
            print(f"创建时间: {post['created_at'][:19]}")

            # 显示内容预览
            preview = post['content'][:100]
            if len(post['content']) > 100:
                preview += '...'
            print(f"预览: {preview}")
            print("-" * 80)

    def _view_post(self, args):
        """查看文章命令处理"""
        post = self.blog.get_post(args.id)

        if not post:
            print(f"错误: 找不到ID为 {args.id} 的文章")
            return

        print("\n" + "=" * 80)
        print(f"标题: {post['title']}")
        print(f"ID: {post['id']}")
        print(f"分类: {post['category'] or '无'}")
        print(f"标签: {', '.join(post['tags']) or '无'}")
        print(f"创建时间: {post['created_at'][:19]}")
        print(f"更新时间: {post['updated_at'][:19]}")
        print("=" * 80)
        print(f"\n{post['content']}\n")
        print("=" * 80)

    def _edit_post(self, args):
        """编辑文章命令处理"""
        content = args.content

        if args.file:
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"错误: 无法读取文件 {args.file} - {e}")
                return

        tags = None
        if args.tags:
            tags = [tag.strip() for tag in args.tags.split(',') if tag.strip()]

        success = self.blog.update_post(
            post_id=args.id,
            title=args.title,
            content=content,
            category=args.category,
            tags=tags
        )

        if success:
            print(f"✓ 文章 ID {args.id} 更新成功！")
        else:
            print(f"错误: 找不到ID为 {args.id} 的文章")

    def _delete_post(self, args):
        """删除文章命令处理"""
        post = self.blog.get_post(args.id)

        if not post:
            print(f"错误: 找不到ID为 {args.id} 的文章")
            return

        print(f"确认删除文章: {post['title']} (ID: {args.id}) ? [y/N] ", end='')
        confirm = input().lower()

        if confirm == 'y':
            if self.blog.delete_post(args.id):
                print(f"✓ 文章 ID {args.id} 已删除")
            else:
                print("删除失败")
        else:
            print("已取消")

    def _search_posts(self, args):
        """搜索文章命令处理"""
        posts = self.blog.search_posts(args.keyword)

        if not posts:
            print(f"没有找到包含 '{args.keyword}' 的文章")
            return

        print(f"\n搜索 '{args.keyword}' 找到 {len(posts)} 篇文章:\n")
        print("=" * 80)

        for post in posts:
            print(f"ID: {post['id']}")
            print(f"标题: {post['title']}")
            print(f"分类: {post['category'] or '无'}")
            print(f"标签: {', '.join(post['tags']) or '无'}")

            # 高亮显示匹配的内容
            preview = post['content'][:150]
            if len(post['content']) > 150:
                preview += '...'
            print(f"预览: {preview}")
            print("-" * 80)

    def _show_stats(self):
        """显示统计信息命令处理"""
        stats = self.blog.get_stats()

        print("\n博客统计信息")
        print("=" * 50)
        print(f"博客名称: {stats['blog_name']}")
        print(f"作者: {stats['author']}")
        print(f"文章总数: {stats['total_posts']}")
        print(f"分类总数: {stats['total_categories']}")
        print(f"标签总数: {stats['total_tags']}")
        print("=" * 50)

        if stats['total_categories'] > 0:
            print("\n分类列表:")
            for cat in self.blog.get_categories():
                count = len([p for p in self.blog.data["posts"]
                           if p["category"] == cat])
                print(f"  - {cat}: {count} 篇")

        if stats['total_tags'] > 0:
            print("\n标签列表:")
            for tag in self.blog.get_tags():
                count = len([p for p in self.blog.data["posts"]
                           if tag in p["tags"]])
                print(f"  - {tag}: {count} 篇")

    def _export_post(self, args):
        """导出文章命令处理"""
        post = self.blog.get_post(args.id)

        if not post:
            print(f"错误: 找不到ID为 {args.id} 的文章")
            return

        try:
            if args.format == 'md':
                content = self._export_markdown(post)
            else:  # html
                content = self._export_html(post)

            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✓ 文章已导出到 {args.output}")
        except Exception as e:
            print(f"错误: 导出失败 - {e}")

    def _export_markdown(self, post: Dict) -> str:
        """导出为Markdown格式"""
        lines = [
            f"# {post['title']}",
            "",
            f"**分类:** {post['category'] or '无'}",
            f"**标签:** {', '.join(post['tags']) or '无'}",
            f"**创建时间:** {post['created_at'][:19]}",
            f"**更新时间:** {post['updated_at'][:19]}",
            "",
            "---",
            "",
            post['content']
        ]
        return '\n'.join(lines)

    def _export_html(self, post: Dict) -> str:
        """导出为HTML格式"""
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{post['title']}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }}
        .meta {{
            color: #666;
            font-size: 0.9em;
            margin: 20px 0;
        }}
        .content {{
            margin-top: 30px;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <h1>{post['title']}</h1>
    <div class="meta">
        <p><strong>分类:</strong> {post['category'] or '无'}</p>
        <p><strong>标签:</strong> {', '.join(post['tags']) or '无'}</p>
        <p><strong>创建时间:</strong> {post['created_at'][:19]}</p>
        <p><strong>更新时间:</strong> {post['updated_at'][:19]}</p>
    </div>
    <hr>
    <div class="content">
        {post['content']}
    </div>
</body>
</html>
"""
        return html


def main():
    """主函数"""
    cli = BlogCLI()
    cli.run()


if __name__ == '__main__':
    main()
