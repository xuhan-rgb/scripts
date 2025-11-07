# 博客管理系统

一个功能完整的命令行博客管理系统，支持文章创建、编辑、删除、搜索等功能。

## 特性

- ✅ **完整的文章管理**：创建、编辑、删除、查看文章
- ✅ **分类和标签**：支持文章分类和多标签
- ✅ **强大的搜索**：按标题、内容、分类、标签搜索
- ✅ **Markdown支持**：原生支持Markdown格式
- ✅ **数据持久化**：JSON格式存储，易于备份和迁移
- ✅ **两种界面**：命令行参数模式和交互式模式
- ✅ **文章导出**：支持导出为Markdown或HTML格式
- ✅ **统计信息**：查看博客统计数据

## 安装

无需安装额外依赖，系统自带Python 3即可运行。

```bash
# 克隆或下载项目文件
chmod +x blog.py blog_interactive.py
```

## 使用方法

### 1. 命令行模式

#### 创建文章

```bash
# 直接创建
python3 blog.py create "我的第一篇博客" -c "这是文章内容" --category "技术" --tags "Python,编程"

# 从文件导入内容
python3 blog.py create "Python教程" -f article.md --category "技术" --tags "Python,教程"
```

#### 列出文章

```bash
# 列出所有文章
python3 blog.py list

# 按分类筛选
python3 blog.py list --category "技术"

# 按标签筛选
python3 blog.py list --tag "Python"

# 限制显示数量
python3 blog.py list --limit 5
```

#### 查看文章

```bash
python3 blog.py view 1
```

#### 编辑文章

```bash
# 更新标题
python3 blog.py edit 1 --title "新标题"

# 更新内容
python3 blog.py edit 1 --content "新内容"

# 从文件更新
python3 blog.py edit 1 -f new_content.md

# 更新分类和标签
python3 blog.py edit 1 --category "新分类" --tags "标签1,标签2"
```

#### 删除文章

```bash
python3 blog.py delete 1
```

#### 搜索文章

```bash
python3 blog.py search "Python"
```

#### 查看统计

```bash
python3 blog.py stats
```

#### 导出文章

```bash
# 导出为Markdown
python3 blog.py export 1 -o article.md --format md

# 导出为HTML
python3 blog.py export 1 -o article.html --format html
```

### 2. 交互式模式

交互式模式提供更友好的用户界面：

```bash
python3 blog_interactive.py
```

功能包括：
- 📝 创建新文章（支持编辑器）
- 📚 查看所有文章（支持筛选）
- 🔍 搜索文章
- ✏️ 编辑文章
- 🗑️ 删除文章
- 🏷️ 查看分类和标签
- 📊 统计信息
- 💾 导出文章
- ⚙️ 设置

## 数据存储

所有数据存储在 `blog_data.json` 文件中，格式如下：

```json
{
  "posts": [
    {
      "id": 1,
      "title": "文章标题",
      "content": "文章内容",
      "category": "分类",
      "tags": ["标签1", "标签2"],
      "created_at": "2025-01-01T12:00:00",
      "updated_at": "2025-01-01T12:00:00",
      "status": "published"
    }
  ],
  "categories": ["分类1", "分类2"],
  "tags": ["标签1", "标签2"],
  "config": {
    "author": "博主",
    "blog_name": "我的博客",
    "created_at": "2025-01-01T12:00:00"
  }
}
```

## 配置文件

可以通过 `blog_config.ini` 配置博客设置：

```ini
[Blog]
blog_name = 我的博客
author = 博主
data_file = blog_data.json
editor =

[Display]
preview_length = 100
posts_per_page = 10

[Export]
default_format = md
export_dir = ./exports
```

## 使用示例

### 完整工作流示例

```bash
# 1. 创建第一篇文章
python3 blog.py create "Python入门教程" \
  -c "Python是一门简单易学的编程语言..." \
  --category "编程教程" \
  --tags "Python,入门,教程"

# 2. 创建第二篇文章
python3 blog.py create "Flask Web开发" \
  -f flask_tutorial.md \
  --category "编程教程" \
  --tags "Python,Flask,Web"

# 3. 列出所有文章
python3 blog.py list

# 4. 查看第一篇文章
python3 blog.py view 1

# 5. 搜索包含"Python"的文章
python3 blog.py search "Python"

# 6. 编辑文章
python3 blog.py edit 1 --title "Python入门教程（2025版）"

# 7. 导出文章
python3 blog.py export 1 -o python_tutorial.md

# 8. 查看统计信息
python3 blog.py stats

# 9. 删除文章
python3 blog.py delete 2
```

### 交互式模式示例

```bash
# 启动交互式界面
python3 blog_interactive.py

# 在界面中：
# 1. 选择 "1" 创建新文章
# 2. 输入标题、分类、标签
# 3. 选择使用编辑器编辑内容
# 4. 保存后返回主菜单
# 5. 选择 "2" 查看所有文章
# 6. 选择 "7" 查看统计信息
```

## 高级功能

### 使用外部编辑器

在交互式模式下，可以使用系统编辑器（如vim、nano、code等）编辑文章：

```bash
# 设置默认编辑器
export EDITOR=vim
# 或
export EDITOR=nano
# 或
export EDITOR=code

# 然后启动交互式界面
python3 blog_interactive.py
```

### 批量导入文章

可以编写脚本批量导入文章：

```python
from blog import BlogManager

blog = BlogManager()

articles = [
    {"title": "文章1", "content": "内容1", "category": "分类1"},
    {"title": "文章2", "content": "内容2", "category": "分类2"},
]

for article in articles:
    blog.create_post(**article)
```

### 数据备份

```bash
# 备份数据
cp blog_data.json blog_data_backup_$(date +%Y%m%d).json

# 恢复数据
cp blog_data_backup_20250101.json blog_data.json
```

## 测试

运行测试套件：

```bash
python3 test_blog.py
```

测试覆盖：
- ✅ 文章创建、读取、更新、删除
- ✅ 分类和标签管理
- ✅ 搜索功能
- ✅ 数据持久化
- ✅ 统计信息
- ✅ 完整工作流

## 技术架构

```
blog.py                   # 核心博客管理类和CLI
├── BlogManager          # 博客管理核心类
│   ├── create_post()   # 创建文章
│   ├── get_post()      # 获取文章
│   ├── update_post()   # 更新文章
│   ├── delete_post()   # 删除文章
│   ├── list_posts()    # 列出文章
│   ├── search_posts()  # 搜索文章
│   └── get_stats()     # 获取统计
└── BlogCLI              # 命令行界面

blog_interactive.py       # 交互式界面
└── InteractiveBlogEditor # 交互式编辑器

test_blog.py             # 测试套件
├── TestBlogManager      # 单元测试
└── TestBlogIntegration  # 集成测试

blog_config.ini          # 配置文件
blog_data.json           # 数据文件
```

## 常见问题

### Q: 如何修改博客名称和作者？

A: 有两种方式：
1. 使用交互式界面的"设置"菜单
2. 直接编辑 `blog_data.json` 中的 `config` 部分

### Q: 文章ID会重复吗？

A: 不会。系统使用自增ID，即使删除文章，新文章的ID也会继续递增。

### Q: 如何迁移数据？

A: 只需复制 `blog_data.json` 文件到新位置即可。

### Q: 支持Markdown渲染吗？

A: 系统存储原始Markdown文本。导出为HTML时会保留格式，但不进行Markdown到HTML的转换。可以使用外部工具（如pandoc）进行转换。

### Q: 如何批量删除文章？

A: 可以编写Python脚本调用BlogManager的delete_post方法。

## 未来计划

- [ ] Markdown到HTML的渲染
- [ ] 文章草稿功能
- [ ] 文章修订历史
- [ ] 全文搜索优化
- [ ] Web界面
- [ ] 数据库支持（SQLite）
- [ ] 图片管理
- [ ] 评论系统
- [ ] RSS生成

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 作者

博客管理系统 v1.0

## 更新日志

### v1.0 (2025-11-07)
- ✅ 初始版本发布
- ✅ 完整的文章CRUD功能
- ✅ 分类和标签系统
- ✅ 搜索功能
- ✅ 命令行和交互式两种界面
- ✅ 文章导出功能
- ✅ 完整的测试覆盖
