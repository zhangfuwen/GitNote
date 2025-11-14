<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#VK_KHR_image_format_list">VK_KHR_image_format_list</a></li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/extensions/VK_KHR_image_format_list.html
layout: default
---</p>
</div>
<h1 id="VK_KHR_image_format_list" class="sect0">VK_KHR_image_format_list</h1>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Promoted to core in Vulkan 1.2</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>On some implementations, setting the <code>VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT</code> on <code>VkImage</code> creation can cause access to that <code>VkImage</code> to perform worse than an equivalent <code>VkImage</code> created without <code>VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT</code> because the implementation does not know what <code>VkImageView</code> formats will be paired with the <code>VkImage</code>. This may force the implementation to disable (<code>VkImageView</code>) format-specific optimizations such as lossless image compression. If the <code>VkImageFormatListCreateInfo</code> struct used to explicitly list the <code>VkImageView</code> formats the <code>VkImage</code> may be paired with, the implementation may be able to enable format-specific optimization in additional cases.</p>
</div>
<div class="paragraph">
<p>If the application is not using the <code>VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT</code> to create images, then there is no need to be concerned with this extension.</p>
</div>