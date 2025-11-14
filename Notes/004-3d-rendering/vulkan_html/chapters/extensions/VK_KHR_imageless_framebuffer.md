<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#VK_KHR_imageless_framebuffer">VK_KHR_imageless_framebuffer</a></li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/extensions/VK_KHR_imageless_framebuffer.html
layout: default
---</p>
</div>
<h1 id="VK_KHR_imageless_framebuffer" class="sect0">VK_KHR_imageless_framebuffer</h1>
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
<p>When creating a <code>VkFramebuffer</code> you normally need to pass the <code>VkImageViews</code> being used in <code>VkFramebufferCreateInfo::pAttachments</code>.</p>
</div>
<div class="paragraph">
<p>To use an imageless <code>VkFramebuffer</code></p>
</div>
<div class="ulist">
<ul>
<li>
<p>Make sure the implementation has support for it by querying <code>VkPhysicalDeviceImagelessFramebufferFeatures::imagelessFramebuffer</code> or <code>VkPhysicalDeviceVulkan12Features::imagelessFramebuffer</code></p>
</li>
<li>
<p>Set the <code>VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT</code> in <code>VkFramebufferCreateInfo::flags</code></p>
</li>
<li>
<p>Include a <code>VkFramebufferAttachmentsCreateInfo</code> struct in the <code>VkFramebufferCreateInfo::pNext</code></p>
</li>
<li>
<p>When beginning the render pass, pass in a <code>VkRenderPassAttachmentBeginInfo</code> structure into <code>VkRenderPassBeginInfo::pNext</code> with the compatible attachments</p>
</li>
</ul>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// Fill information about attachment
VkFramebufferAttachmentImageInfo attachments_image_info = {};
// ...

VkFramebufferAttachmentsCreateInfo attachments_create_info = {};
// ...
attachments_create_info.attachmentImageInfoCount = 1;
attachments_create_info.pAttachmentImageInfos = &amp;attachments_image_info;

// Create FrameBuffer as imageless
VkFramebufferCreateInfo framebuffer_info = {};
framebuffer_info.pNext = &amp;attachments_create_info;
framebuffer_info.flags |= VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT;
// ...
framebffer_info.pAttachments = NULL; // pAttachments is ignored here now

vkCreateFramebuffer(device, &amp;framebuffer_info, NULL, &amp;framebuffer_object);

// ...

// Start recording a command buffer
VkRenderPassAttachmentBeginInfo attachment_begin_info = {};
// attachment_begin_info.pAttachments contains VkImageView objects

VkRenderPassBeginInfo begin_info = {};
begin_info.pNext = &amp;attachment_begin_info;
// ...

vkCmdBeginRenderPass(command_buffer, &amp;begin_info, VK_SUBPASS_CONTENTS_INLINE);</code></pre>
</div>
</div>