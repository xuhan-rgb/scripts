#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式博客编辑器
提供友好的用户界面来管理博客
"""

import os
import sys
import tempfile
import subprocess
from blog import BlogManager


class InteractiveBlogEditor:
    """交互式博客编辑器"""

    def __init__(self):
        self.blog = BlogManager()
        self.running = True

    def clear_screen(self):
        """清屏"""
        os.system('clear' if os.name != 'nt' else 'cls')

    def print_header(self):
        """打印头部"""
        print("\n" + "=" * 60)
        print(" " * 20 + "博客管理系统")
        print("=" * 60)

    def print_menu(self):
        """打印主菜单"""
        self.clear_screen()
        self.print_header()

        stats = self.blog.get_stats()
        print(f"\n博客: {stats['blog_name']} | 作者: {stats['author']}")
        print(f"文章总数: {stats['total_posts']} | "
              f"分类: {stats['total_categories']} | "
              f"标签: {stats['total_tags']}\n")

        print("请选择操作:")
        print("  1. 创建新文章")
        print("  2. 查看所有文章")
        print("  3. 搜索文章")
        print("  4. 编辑文章")
        print("  5. 删除文章")
        print("  6. 查看分类和标签")
        print("  7. 统计信息")
        print("  8. 导出文章")
        print("  9. 设置")
        print("  0. 退出")
        print("\n" + "=" * 60)

    def get_input(self, prompt: str, required: bool = True) -> str:
        """获取用户输入"""
        while True:
            value = input(prompt).strip()
            if value or not required:
                return value
            print("此项为必填项，请输入有效值")

    def pause(self):
        """暂停等待用户确认"""
        input("\n按回车键继续...")

    def create_post_interactive(self):
        """交互式创建文章"""
        self.clear_screen()
        self.print_header()
        print("\n创建新文章\n")

        title = self.get_input("文章标题: ")
        category = self.get_input("文章分类 (可选): ", required=False)
        tags_input = self.get_input("文章标签 (逗号分隔，可选): ", required=False)
        tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()]

        print("\n选择输入方式:")
        print("  1. 直接输入内容")
        print("  2. 使用编辑器编辑")
        print("  3. 从文件导入")

        choice = self.get_input("请选择 (1-3): ")

        content = ""
        if choice == "1":
            print("\n请输入文章内容 (输入EOF或Ctrl+D结束):")
            lines = []
            try:
                while True:
                    line = input()
                    if line == "EOF":
                        break
                    lines.append(line)
            except EOFError:
                pass
            content = '\n'.join(lines)

        elif choice == "2":
            content = self.edit_in_editor("")

        elif choice == "3":
            filepath = self.get_input("文件路径: ")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"错误: 无法读取文件 - {e}")
                self.pause()
                return

        if content or input("\n内容为空，确认创建? (y/N): ").lower() == 'y':
            post = self.blog.create_post(title, content, category, tags)
            print(f"\n✓ 文章创建成功! (ID: {post['id']})")
        else:
            print("\n已取消创建")

        self.pause()

    def list_posts_interactive(self):
        """交互式列出文章"""
        self.clear_screen()
        self.print_header()
        print("\n文章列表\n")

        print("筛选选项:")
        print("  1. 显示所有文章")
        print("  2. 按分类筛选")
        print("  3. 按标签筛选")

        choice = self.get_input("请选择 (1-3): ")

        category = None
        tag = None

        if choice == "2":
            categories = self.blog.get_categories()
            if categories:
                print(f"\n可用分类: {', '.join(categories)}")
                category = self.get_input("请输入分类: ")
            else:
                print("\n没有可用的分类")
                self.pause()
                return

        elif choice == "3":
            tags = self.blog.get_tags()
            if tags:
                print(f"\n可用标签: {', '.join(tags)}")
                tag = self.get_input("请输入标签: ")
            else:
                print("\n没有可用的标签")
                self.pause()
                return

        posts = self.blog.list_posts(category=category, tag=tag)

        if not posts:
            print("\n没有找到文章")
            self.pause()
            return

        print(f"\n找到 {len(posts)} 篇文章:\n")
        print("-" * 60)

        for post in posts:
            print(f"\nID: {post['id']} | {post['title']}")
            print(f"分类: {post['category'] or '无'} | "
                  f"标签: {', '.join(post['tags']) or '无'}")
            print(f"创建: {post['created_at'][:19]}")

            preview = post['content'][:80].replace('\n', ' ')
            if len(post['content']) > 80:
                preview += '...'
            print(f"预览: {preview}")
            print("-" * 60)

        # 提供查看选项
        view_id = self.get_input("\n输入文章ID查看详情 (按回车跳过): ",
                                 required=False)
        if view_id and view_id.isdigit():
            self.view_post_interactive(int(view_id))
        else:
            self.pause()

    def view_post_interactive(self, post_id: int = None):
        """交互式查看文章"""
        if post_id is None:
            self.clear_screen()
            self.print_header()
            print("\n查看文章\n")
            post_id_str = self.get_input("请输入文章ID: ")
            try:
                post_id = int(post_id_str)
            except ValueError:
                print("错误: 无效的文章ID")
                self.pause()
                return

        post = self.blog.get_post(post_id)

        if not post:
            print(f"\n错误: 找不到ID为 {post_id} 的文章")
            self.pause()
            return

        self.clear_screen()
        self.print_header()
        print(f"\n标题: {post['title']}")
        print(f"ID: {post['id']}")
        print(f"分类: {post['category'] or '无'}")
        print(f"标签: {', '.join(post['tags']) or '无'}")
        print(f"创建时间: {post['created_at'][:19]}")
        print(f"更新时间: {post['updated_at'][:19]}")
        print("\n" + "-" * 60)
        print(post['content'])
        print("-" * 60)

        self.pause()

    def search_posts_interactive(self):
        """交互式搜索文章"""
        self.clear_screen()
        self.print_header()
        print("\n搜索文章\n")

        keyword = self.get_input("请输入搜索关键词: ")
        posts = self.blog.search_posts(keyword)

        if not posts:
            print(f"\n没有找到包含 '{keyword}' 的文章")
            self.pause()
            return

        print(f"\n找到 {len(posts)} 篇文章:\n")
        print("-" * 60)

        for post in posts:
            print(f"\nID: {post['id']} | {post['title']}")
            print(f"分类: {post['category'] or '无'} | "
                  f"标签: {', '.join(post['tags']) or '无'}")

            preview = post['content'][:100].replace('\n', ' ')
            if len(post['content']) > 100:
                preview += '...'
            print(f"预览: {preview}")
            print("-" * 60)

        view_id = self.get_input("\n输入文章ID查看详情 (按回车跳过): ",
                                 required=False)
        if view_id and view_id.isdigit():
            self.view_post_interactive(int(view_id))
        else:
            self.pause()

    def edit_in_editor(self, content: str) -> str:
        """使用系统编辑器编辑内容"""
        editor = os.environ.get('EDITOR', 'nano')

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.md',
                                        delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_file = f.name

        try:
            subprocess.call([editor, temp_file])
            with open(temp_file, 'r', encoding='utf-8') as f:
                return f.read()
        finally:
            os.unlink(temp_file)

    def edit_post_interactive(self):
        """交互式编辑文章"""
        self.clear_screen()
        self.print_header()
        print("\n编辑文章\n")

        post_id_str = self.get_input("请输入文章ID: ")
        try:
            post_id = int(post_id_str)
        except ValueError:
            print("错误: 无效的文章ID")
            self.pause()
            return

        post = self.blog.get_post(post_id)
        if not post:
            print(f"错误: 找不到ID为 {post_id} 的文章")
            self.pause()
            return

        print(f"\n当前文章: {post['title']}")
        print("\n选择要编辑的内容:")
        print("  1. 标题")
        print("  2. 内容")
        print("  3. 分类")
        print("  4. 标签")
        print("  5. 全部编辑")

        choice = self.get_input("请选择 (1-5): ")

        title = None
        content = None
        category = None
        tags = None

        if choice in ["1", "5"]:
            new_title = self.get_input(f"新标题 (当前: {post['title']}): ",
                                      required=False)
            if new_title:
                title = new_title

        if choice in ["2", "5"]:
            print("\n编辑内容:")
            print("  1. 使用编辑器")
            print("  2. 直接输入")
            edit_choice = self.get_input("请选择 (1-2): ")

            if edit_choice == "1":
                content = self.edit_in_editor(post['content'])
            elif edit_choice == "2":
                print("\n请输入新内容 (输入EOF结束):")
                lines = []
                try:
                    while True:
                        line = input()
                        if line == "EOF":
                            break
                        lines.append(line)
                except EOFError:
                    pass
                content = '\n'.join(lines)

        if choice in ["3", "5"]:
            new_category = self.get_input(
                f"新分类 (当前: {post['category'] or '无'}): ",
                required=False
            )
            if new_category:
                category = new_category

        if choice in ["4", "5"]:
            current_tags = ', '.join(post['tags'])
            new_tags = self.get_input(
                f"新标签 (当前: {current_tags or '无'}，逗号分隔): ",
                required=False
            )
            if new_tags:
                tags = [tag.strip() for tag in new_tags.split(',') if tag.strip()]

        if any([title, content, category, tags is not None]):
            success = self.blog.update_post(post_id, title, content,
                                           category, tags)
            if success:
                print(f"\n✓ 文章 ID {post_id} 更新成功!")
            else:
                print(f"\n错误: 更新失败")
        else:
            print("\n没有进行任何修改")

        self.pause()

    def delete_post_interactive(self):
        """交互式删除文章"""
        self.clear_screen()
        self.print_header()
        print("\n删除文章\n")

        post_id_str = self.get_input("请输入要删除的文章ID: ")
        try:
            post_id = int(post_id_str)
        except ValueError:
            print("错误: 无效的文章ID")
            self.pause()
            return

        post = self.blog.get_post(post_id)
        if not post:
            print(f"错误: 找不到ID为 {post_id} 的文章")
            self.pause()
            return

        print(f"\n文章信息:")
        print(f"  标题: {post['title']}")
        print(f"  分类: {post['category'] or '无'}")
        print(f"  创建时间: {post['created_at'][:19]}")

        confirm = self.get_input(f"\n确认删除? (y/N): ")
        if confirm.lower() == 'y':
            if self.blog.delete_post(post_id):
                print(f"\n✓ 文章 ID {post_id} 已删除")
            else:
                print("\n错误: 删除失败")
        else:
            print("\n已取消删除")

        self.pause()

    def show_categories_tags(self):
        """显示分类和标签"""
        self.clear_screen()
        self.print_header()
        print("\n分类和标签\n")

        categories = self.blog.get_categories()
        tags = self.blog.get_tags()

        if categories:
            print("分类:")
            for cat in categories:
                count = len([p for p in self.blog.data["posts"]
                           if p["category"] == cat])
                print(f"  - {cat}: {count} 篇文章")
        else:
            print("分类: 无")

        print()

        if tags:
            print("标签:")
            for tag in tags:
                count = len([p for p in self.blog.data["posts"]
                           if tag in p["tags"]])
                print(f"  - {tag}: {count} 篇文章")
        else:
            print("标签: 无")

        self.pause()

    def show_stats(self):
        """显示统计信息"""
        self.clear_screen()
        self.print_header()
        print("\n统计信息\n")

        stats = self.blog.get_stats()

        print(f"博客名称: {stats['blog_name']}")
        print(f"作者: {stats['author']}")
        print(f"文章总数: {stats['total_posts']}")
        print(f"分类总数: {stats['total_categories']}")
        print(f"标签总数: {stats['total_tags']}")

        if stats['total_posts'] > 0:
            posts = self.blog.data["posts"]
            total_chars = sum(len(p['content']) for p in posts)
            avg_chars = total_chars // stats['total_posts']
            print(f"平均文章长度: {avg_chars} 字符")

            # 最近更新的文章
            latest = sorted(posts, key=lambda x: x['updated_at'], reverse=True)[:5]
            print("\n最近更新的文章:")
            for post in latest:
                print(f"  - [{post['id']}] {post['title']} "
                      f"({post['updated_at'][:19]})")

        self.pause()

    def export_post_interactive(self):
        """交互式导出文章"""
        self.clear_screen()
        self.print_header()
        print("\n导出文章\n")

        post_id_str = self.get_input("请输入文章ID: ")
        try:
            post_id = int(post_id_str)
        except ValueError:
            print("错误: 无效的文章ID")
            self.pause()
            return

        post = self.blog.get_post(post_id)
        if not post:
            print(f"错误: 找不到ID为 {post_id} 的文章")
            self.pause()
            return

        print(f"\n文章: {post['title']}")
        print("\n选择导出格式:")
        print("  1. Markdown (.md)")
        print("  2. HTML (.html)")

        choice = self.get_input("请选择 (1-2): ")
        format_ext = 'md' if choice == "1" else 'html'

        default_filename = f"{post['title'].replace(' ', '_')}.{format_ext}"
        filename = self.get_input(f"输出文件名 (默认: {default_filename}): ",
                                  required=False)
        if not filename:
            filename = default_filename

        try:
            from blog import BlogCLI
            cli = BlogCLI()

            if format_ext == 'md':
                content = cli._export_markdown(post)
            else:
                content = cli._export_html(post)

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"\n✓ 文章已导出到 {filename}")
        except Exception as e:
            print(f"\n错误: 导出失败 - {e}")

        self.pause()

    def settings(self):
        """设置"""
        self.clear_screen()
        self.print_header()
        print("\n设置\n")

        print("1. 修改博客名称")
        print("2. 修改作者名称")
        print("3. 返回主菜单")

        choice = self.get_input("\n请选择 (1-3): ")

        if choice == "1":
            current_name = self.blog.data["config"]["blog_name"]
            new_name = self.get_input(f"新博客名称 (当前: {current_name}): ",
                                     required=False)
            if new_name:
                self.blog.data["config"]["blog_name"] = new_name
                self.blog._save_data()
                print("\n✓ 博客名称已更新")
                self.pause()

        elif choice == "2":
            current_author = self.blog.data["config"]["author"]
            new_author = self.get_input(f"新作者名称 (当前: {current_author}): ",
                                       required=False)
            if new_author:
                self.blog.data["config"]["author"] = new_author
                self.blog._save_data()
                print("\n✓ 作者名称已更新")
                self.pause()

    def run(self):
        """运行交互式界面"""
        while self.running:
            self.print_menu()
            choice = self.get_input("\n请选择操作 (0-9): ")

            if choice == "1":
                self.create_post_interactive()
            elif choice == "2":
                self.list_posts_interactive()
            elif choice == "3":
                self.search_posts_interactive()
            elif choice == "4":
                self.edit_post_interactive()
            elif choice == "5":
                self.delete_post_interactive()
            elif choice == "6":
                self.show_categories_tags()
            elif choice == "7":
                self.show_stats()
            elif choice == "8":
                self.export_post_interactive()
            elif choice == "9":
                self.settings()
            elif choice == "0":
                self.clear_screen()
                print("\n感谢使用博客管理系统！\n")
                self.running = False
            else:
                print("\n无效的选择，请重新输入")
                self.pause()


def main():
    """主函数"""
    try:
        editor = InteractiveBlogEditor()
        editor.run()
    except KeyboardInterrupt:
        print("\n\n程序已中断\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
