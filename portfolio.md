---
title: "Portfolio"
permalink: /portfolio/
---

# Portfolio

下面是我的项目列表：

{% assign items = site.portfolio | sort: "date" | reverse %}
{% for p in items %}
<div style="border:1px solid #eee; border-radius:12px; padding:16px; margin:16px 0;">
  <div style="display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap;">
    {% if p.header.teaser %}
      <img src="{{ p.header.teaser | relative_url }}" alt="teaser" style="width:240px; max-width:100%; border-radius:10px; border:1px solid #eee;">
    {% endif %}
    <div style="flex:1; min-width:260px;">
      <h2 style="margin:0 0 6px 0;">
        <a href="{{ p.url | relative_url }}">{{ p.title }}</a>
      </h2>
      {% if p.excerpt %}
        <p style="margin:8px 0 10px 0; color:#333;">{{ p.excerpt }}</p>
      {% endif %}
      <p style="margin:0; color:#666; font-size:14px;">
        {% if p.date %}📅 {{ p.date | date: "%Y-%m-%d" }}{% endif %}
        {% if p.tags %}
          · 🏷️ {% for t in p.tags %}<span style="display:inline-block; margin-right:6px;">{{ t }}</span>{% endfor %}
        {% endif %}
      </p>
    </div>
  </div>
</div>
{% endfor %}

{% if items.size == 0 %}
> 还没有检测到作品集条目。请确认你的文件在 `_portfolio/` 下，并且 Front Matter 中包含：  
> `collection: portfolio`
{% endif %}
