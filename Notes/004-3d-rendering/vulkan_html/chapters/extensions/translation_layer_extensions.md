<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#translation-layer-extensions">Translation Layer Extensions</a>
<ul class="sectlevel1">
<li><a href="#VK_EXT_custom_border_color">1. VK_EXT_custom_border_color</a></li>
<li><a href="#VK_EXT_border_color_swizzle">2. VK_EXT_border_color_swizzle</a></li>
<li><a href="#VK_EXT_depth_clip_enable">3. VK_EXT_depth_clip_enable</a></li>
<li><a href="#VK_EXT_depth_clip_control">4. VK_EXT_depth_clip_control</a></li>
<li><a href="#VK_EXT_provoking_vertex">5. VK_EXT_provoking_vertex</a></li>
<li><a href="#VK_EXT_transform_feedback">6. VK_EXT_transform_feedback</a></li>
<li><a href="#VK_EXT_image_view_min_lod">7. VK_EXT_image_view_min_lod</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/extensions/translation_layer_extensions.html
layout: default
---</p>
</div>
<h1 id="translation-layer-extensions" class="sect0">Translation Layer Extensions</h1>
<div class="paragraph">
<p>There is a class of extensions that were only created to allow efficient ways for <a href="../portability_initiative.html#translation-layer">translation layers</a> to map to Vulkan.</p>
</div>
<div class="paragraph">
<p>This includes replicating legacy behavior that is challenging for drivers to implement efficiently. This functionality is <strong>not</strong> considered forward looking, and is <strong>not</strong> expected to be promoted to a KHR extension or to core Vulkan.</p>
</div>
<div class="paragraph">
<p>Unless this is needed for translation, it is <strong>highly recommended</strong> that developers use alternative techniques of using the GPU to achieve the same functionality.</p>
</div>
<div class="sect1">
<h2 id="VK_EXT_custom_border_color">1. VK_EXT_custom_border_color</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Vulkan provides a transparent black, opaque black, and opaque white <code>VkBorderColor</code> for <code>VkSampler</code> objects in the core spec. Both OpenGL and D3D have the option to set the sampler border to be a custom color.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_border_color_swizzle">2. VK_EXT_border_color_swizzle</h2>
<div class="sectionbody">
<div class="paragraph">
<p>After the publication of <code>VK_EXT_custom_border_color</code>, it was discovered that some implementations had undefined behavior when combining a sampler that uses a custom border color with image views whose component mapping is not the identity mapping.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_depth_clip_enable">3. VK_EXT_depth_clip_enable</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The depth clip enable functionality is specified differently from D3D11 and Vulkan. Instead of <code>VkPipelineRasterizationStateCreateInfo::depthClampEnable</code>, D3D11 has <a href="https://docs.microsoft.com/en-us/windows/win32/api/d3d11/ns-d3d11-d3d11_rasterizer_desc">DepthClipEnable (D3D12_RASTERIZER_DESC)</a>, which only affects the viewport clip of depth values before rasterization and does not affect the depth clamp that always occurs in the output merger stage of the D3D11 graphics pipeline.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_depth_clip_control">4. VK_EXT_depth_clip_control</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The depth clip control functionality allows the application to use the OpenGL depth range in NDC. In OpenGL it is <code>[-1, 1]</code> as opposed to Vulkan’s default of <code>[0, 1]</code>. Support for clip control was supported in OpenGL via the <a href="https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_clip_control.txt">ARB_clip_control</a> extension.</p>
</div>
<div class="paragraph">
<p>More info in the <a href="../depth.html#user-defined-clipping-and-culling">depth chapter</a></p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_provoking_vertex">5. VK_EXT_provoking_vertex</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Vulkan&#8217;s defaults convention for provoking vertex is &#8220;first vertex&#8221; while OpenGL&#8217;s defaults convention is &#8220;last vertex&#8221;.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_transform_feedback">6. VK_EXT_transform_feedback</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Everything needed for transform feedback can be done via a compute shader in Vulkan. There is also a great <a href="https://www.jlekstrand.net/jason/blog/2018/10/transform-feedback-is-terrible-so-why/">blog by Jason Ekstrand</a> on why transform feedback is terrible and should be avoided.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_image_view_min_lod">7. VK_EXT_image_view_min_lod</h2>
<div class="sectionbody">
<div class="paragraph">
<p>This extension provides an API-side version of the <code>MinLod</code> SPIR-V qualifier.
The new value is associated with the image view, and is
intended to match D3D12&#8217;s SRV ResourceMinLODClamp parameter.
Using MinLod and similar functionality is primarily intended for sparse texturing since higher resolution mip levels can be paged in and out on demand.
There are many ways to achieve a similar clamp in Vulkan. A <code>VkImageView</code> can clamp the base level, but a <code>MinLod</code> can also clamp to a fractional LOD
and does not have to modify the base texture dimension, which might simplify some algorithms. <code>VkSampler</code>&#8203;s can also clamp to fractional LOD, but
using many unique samplers for this purpose might not be practical.</p>
</div>
</div>
</div>