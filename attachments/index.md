---

title: 附件

---

[liquid cheetsheet](https://cloudcannon.com/community/jekyll-cheat-sheet/)
[liquid tutorial](https://cloudcannon.com/community/learn/jekyll-tutorial/)
[liquid playground](https://geekplayers.com/run-liquild-online.html)
[liquid official doc](https://shopify.github.io/liquid/basics/introduction/)
[how to use liquid in jekyll](https://blog.webjeda.com/jekyll-liquid/)

[ruby online playground](https://try.ruby-lang.org/)
[ruby in twenty minutes](https://www.ruby-lang.org/zh_cn/documentation/quickstart/)

[liquid operators](https://learn.microsoft.com/en-us/power-apps/maker/portals/liquid/liquid-operators)


{% raw %}

{% assign image_files = site.static_files %}

{% for myimage in image_files %}
  {% if myimage.path contains 'attachments' %}
    [{{ myimage.path }}]({{ myimage.path }})
  {% endif %}
{% endfor %}

{% endraw %}