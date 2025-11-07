#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
博客系统测试用例
"""

import os
import json
import unittest
import tempfile
from blog import BlogManager


class TestBlogManager(unittest.TestCase):
    """博客管理器测试"""

    def setUp(self):
        """测试前准备"""
        # 创建临时数据文件
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                                     suffix='.json')
        self.temp_file.close()
        self.blog = BlogManager(self.temp_file.name)

    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_create_post(self):
        """测试创建文章"""
        post = self.blog.create_post(
            title="测试文章",
            content="这是一篇测试文章",
            category="测试",
            tags=["tag1", "tag2"]
        )

        self.assertEqual(post['id'], 1)
        self.assertEqual(post['title'], "测试文章")
        self.assertEqual(post['content'], "这是一篇测试文章")
        self.assertEqual(post['category'], "测试")
        self.assertEqual(post['tags'], ["tag1", "tag2"])
        self.assertIn('created_at', post)
        self.assertIn('updated_at', post)

    def test_get_post(self):
        """测试获取文章"""
        # 创建文章
        created_post = self.blog.create_post("测试", "内容")

        # 获取文章
        post = self.blog.get_post(created_post['id'])
        self.assertIsNotNone(post)
        self.assertEqual(post['title'], "测试")

        # 获取不存在的文章
        post = self.blog.get_post(999)
        self.assertIsNone(post)

    def test_update_post(self):
        """测试更新文章"""
        # 创建文章
        post = self.blog.create_post("原标题", "原内容")
        post_id = post['id']

        # 更新标题
        success = self.blog.update_post(post_id, title="新标题")
        self.assertTrue(success)

        updated_post = self.blog.get_post(post_id)
        self.assertEqual(updated_post['title'], "新标题")
        self.assertEqual(updated_post['content'], "原内容")

        # 更新内容
        success = self.blog.update_post(post_id, content="新内容")
        self.assertTrue(success)

        updated_post = self.blog.get_post(post_id)
        self.assertEqual(updated_post['content'], "新内容")

        # 更新不存在的文章
        success = self.blog.update_post(999, title="新标题")
        self.assertFalse(success)

    def test_delete_post(self):
        """测试删除文章"""
        # 创建文章
        post = self.blog.create_post("测试", "内容")
        post_id = post['id']

        # 删除文章
        success = self.blog.delete_post(post_id)
        self.assertTrue(success)

        # 验证已删除
        post = self.blog.get_post(post_id)
        self.assertIsNone(post)

        # 删除不存在的文章
        success = self.blog.delete_post(999)
        self.assertFalse(success)

    def test_list_posts(self):
        """测试列出文章"""
        # 创建多篇文章
        self.blog.create_post("文章1", "内容1", category="分类1", tags=["标签1"])
        self.blog.create_post("文章2", "内容2", category="分类2", tags=["标签2"])
        self.blog.create_post("文章3", "内容3", category="分类1", tags=["标签1", "标签2"])

        # 列出所有文章
        posts = self.blog.list_posts()
        self.assertEqual(len(posts), 3)

        # 按分类筛选
        posts = self.blog.list_posts(category="分类1")
        self.assertEqual(len(posts), 2)

        # 按标签筛选
        posts = self.blog.list_posts(tag="标签2")
        self.assertEqual(len(posts), 2)

        # 限制数量
        posts = self.blog.list_posts(limit=2)
        self.assertEqual(len(posts), 2)

    def test_search_posts(self):
        """测试搜索文章"""
        # 创建文章
        self.blog.create_post("Python教程", "这是一个Python编程教程",
                             category="编程", tags=["Python", "教程"])
        self.blog.create_post("Java教程", "这是一个Java编程教程",
                             category="编程", tags=["Java", "教程"])
        self.blog.create_post("美食推荐", "今天推荐一道美味的菜",
                             category="生活", tags=["美食"])

        # 搜索标题
        posts = self.blog.search_posts("Python")
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0]['title'], "Python教程")

        # 搜索内容
        posts = self.blog.search_posts("编程")
        self.assertEqual(len(posts), 2)

        # 搜索分类
        posts = self.blog.search_posts("生活")
        self.assertEqual(len(posts), 1)

        # 搜索标签
        posts = self.blog.search_posts("教程")
        self.assertEqual(len(posts), 2)

        # 搜索不存在的内容
        posts = self.blog.search_posts("不存在的关键词")
        self.assertEqual(len(posts), 0)

    def test_categories_and_tags(self):
        """测试分类和标签管理"""
        # 创建文章
        self.blog.create_post("文章1", "内容1", category="分类1", tags=["标签1", "标签2"])
        self.blog.create_post("文章2", "内容2", category="分类2", tags=["标签2", "标签3"])

        # 获取分类
        categories = self.blog.get_categories()
        self.assertEqual(len(categories), 2)
        self.assertIn("分类1", categories)
        self.assertIn("分类2", categories)

        # 获取标签
        tags = self.blog.get_tags()
        self.assertEqual(len(tags), 3)
        self.assertIn("标签1", tags)
        self.assertIn("标签2", tags)
        self.assertIn("标签3", tags)

    def test_stats(self):
        """测试统计信息"""
        # 创建文章
        self.blog.create_post("文章1", "内容1", category="分类1", tags=["标签1"])
        self.blog.create_post("文章2", "内容2", category="分类2", tags=["标签2"])

        stats = self.blog.get_stats()

        self.assertEqual(stats['total_posts'], 2)
        self.assertEqual(stats['total_categories'], 2)
        self.assertEqual(stats['total_tags'], 2)
        self.assertIn('blog_name', stats)
        self.assertIn('author', stats)

    def test_data_persistence(self):
        """测试数据持久化"""
        # 创建文章
        post = self.blog.create_post("持久化测试", "测试数据持久化")

        # 创建新的实例，应该能读取之前的数据
        new_blog = BlogManager(self.temp_file.name)
        loaded_post = new_blog.get_post(post['id'])

        self.assertIsNotNone(loaded_post)
        self.assertEqual(loaded_post['title'], "持久化测试")
        self.assertEqual(loaded_post['content'], "测试数据持久化")

    def test_generate_id(self):
        """测试ID生成"""
        # 第一篇文章ID应该是1
        post1 = self.blog.create_post("文章1", "内容1")
        self.assertEqual(post1['id'], 1)

        # 第二篇文章ID应该是2
        post2 = self.blog.create_post("文章2", "内容2")
        self.assertEqual(post2['id'], 2)

        # 删除第一篇后，新文章ID应该是3
        self.blog.delete_post(1)
        post3 = self.blog.create_post("文章3", "内容3")
        self.assertEqual(post3['id'], 3)

    def test_empty_blog(self):
        """测试空博客"""
        posts = self.blog.list_posts()
        self.assertEqual(len(posts), 0)

        categories = self.blog.get_categories()
        self.assertEqual(len(categories), 0)

        tags = self.blog.get_tags()
        self.assertEqual(len(tags), 0)

        stats = self.blog.get_stats()
        self.assertEqual(stats['total_posts'], 0)

    def test_post_with_empty_category_tags(self):
        """测试创建没有分类和标签的文章"""
        post = self.blog.create_post("无分类标签", "内容")

        self.assertEqual(post['category'], "")
        self.assertEqual(post['tags'], [])


class TestBlogIntegration(unittest.TestCase):
    """博客系统集成测试"""

    def setUp(self):
        """测试前准备"""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                                     suffix='.json')
        self.temp_file.close()
        self.blog = BlogManager(self.temp_file.name)

    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_full_workflow(self):
        """测试完整工作流程"""
        # 1. 创建文章
        post = self.blog.create_post(
            title="我的第一篇博客",
            content="这是我的第一篇博客文章，讲述了我的编程之旅。",
            category="个人成长",
            tags=["编程", "生活"]
        )
        self.assertEqual(post['id'], 1)

        # 2. 查看文章
        retrieved_post = self.blog.get_post(1)
        self.assertEqual(retrieved_post['title'], "我的第一篇博客")

        # 3. 创建更多文章
        self.blog.create_post("Python学习笔记", "今天学习了Python基础",
                             category="技术", tags=["Python", "学习"])
        self.blog.create_post("美食日记", "今天做了一道好菜",
                             category="生活", tags=["美食", "生活"])

        # 4. 列出所有文章
        all_posts = self.blog.list_posts()
        self.assertEqual(len(all_posts), 3)

        # 5. 搜索文章
        search_results = self.blog.search_posts("Python")
        self.assertEqual(len(search_results), 1)

        # 6. 更新文章
        success = self.blog.update_post(1, title="我的第一篇博客（已更新）")
        self.assertTrue(success)

        # 7. 验证更新
        updated_post = self.blog.get_post(1)
        self.assertEqual(updated_post['title'], "我的第一篇博客（已更新）")

        # 8. 获取统计信息
        stats = self.blog.get_stats()
        self.assertEqual(stats['total_posts'], 3)
        self.assertEqual(stats['total_categories'], 3)

        # 9. 删除文章
        success = self.blog.delete_post(2)
        self.assertTrue(success)

        # 10. 验证删除
        all_posts = self.blog.list_posts()
        self.assertEqual(len(all_posts), 2)

        deleted_post = self.blog.get_post(2)
        self.assertIsNone(deleted_post)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回测试结果
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
