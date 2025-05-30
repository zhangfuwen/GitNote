<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#VK_EXT_inline_uniform_block">VK_EXT_inline_uniform_block</a>
<ul class="sectlevel1">
<li><a href="#_suggestions">1. Suggestions</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/extensions/VK_EXT_inline_uniform_block.html
layout: default
---</p>
</div>
<h1 id="VK_EXT_inline_uniform_block" class="sect0">VK_EXT_inline_uniform_block</h1>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Promoted to core in Vulkan 1.3</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>For a common implementation, descriptors are just a table to indirectly point to the data that was bound to it during the recording of the command buffer. The issue is that not all descriptors are created equally, for example, one descriptor might only be a few DWORDS in size.</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/VK_EXT_inline_uniform_block_before.png" alt="VK_EXT_inline_uniform_block_before.png">
</div>
</div>
<div class="paragraph">
<p>Using <code>VK_EXT_inline_uniform_block</code> gives an implementation the opportunity to reduce the number of indirections an implementation takes to access uniform values, when only a few values are used. Unlike push constants, this data can be reused across multiple disjoint sets of draws/dispatches.</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/VK_EXT_inline_uniform_block_after.png" alt="VK_EXT_inline_uniform_block_after.png">
</div>
</div>
<div class="sect1">
<h2 id="_suggestions">1. Suggestions</h2>
<div class="sectionbody">
<div class="ulist">
<ul>
<li>
<p>Make sure to check the <code>VkPhysicalDeviceInlineUniformBlockPropertiesEXT</code> struct for the limitation for the implementation&#8217;s usage of inline uniform blocks.</p>
</li>
<li>
<p>Don&#8217;t overdo the usage of inlining, otherwise the driver may need to repack them into a buffer, adding CPU overhead and losing the indirection benefit - aim for no more than a few dwords.</p>
</li>
</ul>
</div>
</div>
</div>