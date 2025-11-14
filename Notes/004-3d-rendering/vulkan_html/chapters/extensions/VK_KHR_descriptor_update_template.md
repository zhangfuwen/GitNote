<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#VK_KHR_descriptor_update_template">VK_KHR_descriptor_update_template</a></li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/extensions/VK_KHR_descriptor_update_template.html
layout: default
---</p>
</div>
<h1 id="VK_KHR_descriptor_update_template" class="sect0">VK_KHR_descriptor_update_template</h1>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Promoted to core in Vulkan 1.1</p>
</div>
<div class="paragraph">
<p><a href="https://www.khronos.org/assets/uploads/developers/library/2018-vulkan-devday/11-DescriptorUpdateTemplates.pdf">Presentation from Montreal Developer Day</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension is designed around how some applications create and update many <code>VkDescriptorSets</code> during the initialization phase. It&#8217;s not unlikely that a lot of updates end up having the same <code>VkDescriptorLayout</code> and the same bindings are being updated so therefore descriptor update templates are designed to only pass the update information once.</p>
</div>
<div class="paragraph">
<p>The descriptors themselves are not specified in the <code>VkDescriptorUpdateTemplate</code>, rather, offsets into an application provided a pointer to host memory are specified, which are combined with a pointer passed to <code>vkUpdateDescriptorSetWithTemplate</code> or <code>vkCmdPushDescriptorSetWithTemplateKHR</code>. This allows large batches of updates to be executed without having to convert application data structures into a strictly-defined Vulkan data structure.</p>
</div>