<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#VK_KHR_draw_indirect_count">VK_KHR_draw_indirect_count</a></li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/extensions/VK_KHR_draw_indirect_count.html
layout: default
---</p>
</div>
<h1 id="VK_KHR_draw_indirect_count" class="sect0">VK_KHR_draw_indirect_count</h1>
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
<p>Every call to <code>vkCmdDraw</code> consumes a set of parameters describing the draw call. To batch draw calls together the same parameters are stored in a <code>VkBuffer</code> in blocks of <code>VkDrawIndirectCommand</code>. Using <code>vkCmdDrawIndirect</code> allows you to invoke a <code>drawCount</code> number of draws, but the <code>drawCount</code> is needed at record time. The new <code>vkCmdDrawIndirectCount</code> call allows the <code>drawCount</code> to also be in a <code>VkBuffer</code>. This allows the value of <code>drawCount</code> to be dynamic and decided when the draw call is executed.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>The <code>vkCmdDrawIndirectCount</code> and <code>vkCmdDrawIndexedIndirectCount</code> function can be used if the extension is supported or if the <code>VkPhysicalDeviceVulkan12Features::drawIndirectCount</code> feature bit is <code>true</code>.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>The following diagram is to visualize the difference between <code>vkCmdDraw</code>, <code>vkCmdDrawIndirect</code>, and <code>vkCmdDrawIndirectCount</code>.</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/VK_KHR_draw_indirect_count_example.png" alt="VK_KHR_draw_indirect_count example">
</div>
</div>