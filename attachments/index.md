---

title: 附件
tags: ['jykell', 'cmake', 'android']

---

 * [liquid cheetsheet](https://cloudcannon.com/community/jekyll-cheat-sheet/)

 * [liquid tutorial](https://cloudcannon.com/community/learn/jekyll-tutorial/)

 * [liquid playground](https://geekplayers.com/run-liquild-online.html)

 * [liquid official doc](https://shopify.github.io/liquid/basics/introduction/)

 * [how to use liquid in jekyll](https://blog.webjeda.com/jekyll-liquid/)

 * [ruby online playground](https://try.ruby-lang.org/)

 * [ruby in twenty minutes](https://www.ruby-lang.org/zh_cn/documentation/quickstart/)

 * [liquid operators](https://learn.microsoft.com/en-us/power-apps/maker/portals/liquid/liquid-operators)


{% assign image_files = site.static_files %}

{% for myimage in image_files %}
  {% if myimage.path contains 'pdf' %}
    <a href="{{ myimage.path }}">{{ myimage.path }}</a>
  {% endif %}
{% endfor %}


 {% if page.tags.size > 0 %} 
   {% for tagName in page.tags %} 
      {% capture tags_content %}
        {{ tags_content }} 
        <a href='/tags?tagName={{ tagName }}'><i class='glyphicon glyphicon-tag'></i>{{ tagName }}</a>
      {% endcapture %} 
      
  {% endfor %} 
  {% else %} 
    {% assign tags_content = '' %} 
  {% endif %} 
  
  {{ tags_content }} 


 {% for tag in site.tags %} 
 {% assign tagName = tag | first | downcase %} 
 {% assign postsCount = tag | last | size %} 
 <li>
 <a href='/tags?tagName={{ tagName }}'><i class='glyphicon glyphicon-tag'></i>{{ tagName }}</a>
 ({{ postsCount }})
 </li> 
 
 {% endfor %} 


 <h2>Tags</h2>
<ul>
  {% assign sorted_tags = site.tags | sort %}
  {% for tag in sorted_tags %}
    {% assign t = tag | first %}
    {% assign posts = tag | last %}
    <li>
      <a href="/tags/# {{ "{{ t | downcase | replace:' ','-'" }}}}">
        {{ t | downcase | replace:' ','-' " }}
        <span>({{ posts | size }})</span>
      </a>
    </li>
  {% endfor %}
</ul>