{% assign image_files = site.static_files | where: "attachment", true %}
{% for myimage in image_files %}
  [{{ myimage.path }}]({{ myimage.path }})
{% endfor %}