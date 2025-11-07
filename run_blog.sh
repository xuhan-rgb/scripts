#!/bin/bash
# 博客管理系统启动脚本

echo "========================================"
echo "      博客管理系统启动器"
echo "========================================"
echo ""
echo "请选择运行模式:"
echo "  1. 交互式模式（推荐）"
echo "  2. 命令行模式帮助"
echo "  3. 运行测试"
echo "  4. 查看统计信息"
echo "  5. 查看文档"
echo ""
read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        echo ""
        echo "启动交互式界面..."
        python3 blog_interactive.py
        ;;
    2)
        echo ""
        python3 blog.py --help
        ;;
    3)
        echo ""
        echo "运行测试套件..."
        python3 test_blog.py
        ;;
    4)
        echo ""
        python3 blog.py stats
        ;;
    5)
        echo ""
        if [ -f "BLOG_README.md" ]; then
            cat BLOG_README.md
        else
            echo "文档文件不存在"
        fi
        ;;
    *)
        echo ""
        echo "无效选项"
        exit 1
        ;;
esac
