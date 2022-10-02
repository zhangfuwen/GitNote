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
<a href='/tags?tagName={{ tagName }}'><i class='glyphicon glyphicon-tag'></i>{{ tagName }}</a>
  {% endfor %} 
 {% else %} 
    {% assign tags_content = '' %} 
 {% endif %} 
  
----

<h2>Post Tags</h2>
<ul id="postTags">
{% assign rawtags = "" %}
{% for post in site.posts %}
  {% assign ttags = post.tags | join:'|' | append:'|' %}
  {% assign rawtags = rawtags | append:ttags %}
<li class="post">
<a href ="{{ post.url }}"> {{ post.title }} </a>
  {% for tagName in post.tags %}
<a href='/tags?tagName={{ tagName }}'><i class='glyphicon glyphicon-tag'></i>{{ tagName }}</a>
  {% endfor %}
</li>
{% endfor %}
{% for post in site.pages %}
  {% assign ttags = post.tags | join:'|' | append:'|' %}
  {% assign rawtags = rawtags | append:ttags %}
<li class="page"> 
<a href ="{{ post.url }}"> {{ post.title }} </a>
  {% for tagName in post.tags %}
<a href='/tags?tagName={{ tagName }}'><i class='glyphicon glyphicon-tag'></i>{{ tagName }}</a>
  {% endfor %}
</li>
{% endfor %}
{% assign rawtags = rawtags | split:'|' | sort %}
</ul>

---

<!-- {% if site.tags != "" %} -->
<!-- {% assign site.tags = "" %} -->
{% for tag in rawtags %}
  {% if tag != "" %}
    {% if tags == "" %}
      {% assign tags = tag | split:'|' %}
    {% endif %}
    {% unless tags contains tag %}
      {% assign tags = tags | join:'|' | append:'|' | append:tag | split:'|' %}
    {% endunless %}
  {% endif %}
{% endfor %}
<!-- {% endif %} -->

<h2>Site Tags</h2>
<ul id="site_tags">
{% for tagName in tags %}
<a href='/tags?tagName={{ tagName }}'><i class='glyphicon glyphicon-tag'></i>{{ tagName }}</a>
{% endfor %}
</ul>

 {% for tag in site.tags %} 
 <!-- {% assign tagName = tag | first %}  -->
 <!-- {% assign tagName = tag | first | downcase %}  -->
 <!-- {% assign postsCount = tag t | size %}  -->
 <li>
 <a href='/tags?tagName={{ tagName }}'><i class='glyphicon glyphicon-tag'></i>{{ tagName }}</a>
 <!-- ({{ postsCount }}) -->
 </li> 
 
 {% endfor %} 


 <h2>Tags</h2>
<ul>
{% assign sorted_tags = site.tags | sort %}
{% for tag in sorted_tags %}
  {% assign t = tag | first %}
  {% assign posts = tag | last %}
<li>
  <a href="/tags/# "{{ t | downcase | replace:' ','-' }}">
    {{ t | downcase | replace:' ','-' " }}
    <span>({{ posts | size }})</span>
  </a>
</li>
{% endfor %}
</ul>
