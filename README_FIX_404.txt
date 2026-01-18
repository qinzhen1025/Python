使用说明（修复 https://qinzhen1025.github.io/Python/ 404）
====================================================

你现在仓库缺少站点入口与 Jekyll 配置。把本压缩包里的内容“合并/复制”到你的仓库根目录即可。

你需要做：
1) 解压后，把以下文件/文件夹直接放到仓库根目录（与 _portfolio/、images/ 同级）：
   - _config.yml
   - index.html
   - portfolio.md
   - _layouts/

2) GitHub 仓库 Settings -> Pages：
   - Deploy from a branch
   - Branch: main（或你的默认分支）
   - Folder: /(root)

3) 推送后访问：
   - 首页：https://qinzhen1025.github.io/Python/
   - 列表页：https://qinzhen1025.github.io/Python/portfolio/

注意（图片链接）：
- 如果你的项目 md 里用的是绝对路径： ![](/images/xxx.png)
  建议改成： ![]({{ "/images/xxx.png" | relative_url }})
  这样在 /Python/ 子路径下也能正确显示。
