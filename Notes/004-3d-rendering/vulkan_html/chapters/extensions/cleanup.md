<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#cleanup">Cleanup Extensions</a>
<ul class="sectlevel1">
<li><a href="#VK_KHR_driver_properties">1. VK_KHR_driver_properties</a></li>
<li><a href="#VK_EXT_host_query_reset">2. VK_EXT_host_query_reset</a></li>
<li><a href="#VK_KHR_separate_depth_stencil_layouts">3. VK_KHR_separate_depth_stencil_layouts</a></li>
<li><a href="#VK_KHR_depth_stencil_resolve">4. VK_KHR_depth_stencil_resolve</a></li>
<li><a href="#VK_EXT_separate_stencil_usage">5. VK_EXT_separate_stencil_usage</a></li>
<li><a href="#VK_KHR_dedicated_allocation">6. VK_KHR_dedicated_allocation</a></li>
<li><a href="#VK_EXT_sampler_filter_minmax">7. VK_EXT_sampler_filter_minmax</a></li>
<li><a href="#VK_KHR_sampler_mirror_clamp_to_edge">8. VK_KHR_sampler_mirror_clamp_to_edge</a></li>
<li><a href="#VK_EXT_4444_formats-and-VK_EXT_ycbcr_2plane_444_formats">9. VK_EXT_4444_formats and VK_EXT_ycbcr_2plane_444_formats</a></li>
<li><a href="#VK_KHR_format_feature_flags2">10. VK_KHR_format_feature_flags2</a></li>
<li><a href="#VK_EXT_rgba10x6_formats">11. VK_EXT_rgba10x6_formats</a></li>
<li><a href="#maintenance-extensions">12. Maintenance Extensions</a></li>
<li><a href="#pnext-expansions">13. pNext Expansions</a>
<ul class="sectlevel2">
<li><a href="#_example">13.1. Example</a></li>
<li><a href="#_it_is_fine_to_not_use_these">13.2. It is fine to not use these</a></li>
</ul>
</li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/extensions/cleanup.html
layout: default
---</p>
</div>
<h1 id="cleanup" class="sect0">Cleanup Extensions</h1>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>These are extensions that are unofficially called &#8220;cleanup extension&#8221;. The Vulkan Guide defines them as cleanup extensions due to their nature of only adding a small bit of functionality or being very simple, self-explanatory extensions in terms of their purpose.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="sect1">
<h2 id="VK_KHR_driver_properties">1. VK_KHR_driver_properties</h2>
<div class="sectionbody">
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
<p>This extension adds more information to query about each implementation. The <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VkDriverId">VkDriverId</a> will be a registered vendor&#8217;s ID for the implementation. The <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VkConformanceVersion">VkConformanceVersion</a> displays which version of <a href="../vulkan_cts.html#vulkan-cts">the Vulkan Conformance Test Suite</a> the implementation passed.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_host_query_reset">2. VK_EXT_host_query_reset</h2>
<div class="sectionbody">
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
<p>This extension allows an application to call <code>vkResetQueryPool</code> from the host instead of needing to setup logic to submit <code>vkCmdResetQueryPool</code> since this is mainly just a quick write to memory for most implementations.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_separate_depth_stencil_layouts">3. VK_KHR_separate_depth_stencil_layouts</h2>
<div class="sectionbody">
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
<p>This extension allows an application when using a depth/stencil format to do an image translation on each the depth and stencil separately. Starting in Vulkan 1.2 this functionality is required for all implementations.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_depth_stencil_resolve">4. VK_KHR_depth_stencil_resolve</h2>
<div class="sectionbody">
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
<p>This extension adds support for automatically resolving multisampled depth/stencil attachments in a subpass in a similar manner as for color attachments.</p>
</div>
<div class="paragraph">
<p>For more information please check out the GDC presentation. (<a href="https://www.khronos.org/assets/uploads/developers/presentations/Vulkan-Depth-Stencil-Resolve-GDC-Mar19.pdf">slides</a> and <a href="https://www.youtube.com/watch?v=GnnEmJFFC7Q&amp;t=1980s">video</a>)</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_separate_stencil_usage">5. VK_EXT_separate_stencil_usage</h2>
<div class="sectionbody">
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
<p>There are formats that express both the usage of depth and stencil, but there was no way to list a different usage for them. The <code>VkImageStencilUsageCreateInfo</code> now lets an application pass in a separate <code>VkImageUsageFlags</code> for the stencil usage of an image. The depth usage is the original usage passed into <code>VkImageCreateInfo::usage</code> and without using <code>VkImageStencilUsageCreateInfo</code> the stencil usage will be the same as well.</p>
</div>
<div class="paragraph">
<p>A good use case of this is when using the <a href="../VK_KHR_image_format_list.html#VK_KHR_image_format_list">VK_KHR_image_format_list</a> extension. This provides a way for the application to more explicitly describe the possible image views of their <code>VkImage</code> at creation time. This allows some implementations to possibly do implementation dependent optimization depending on the usages set.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_dedicated_allocation">6. VK_KHR_dedicated_allocation</h2>
<div class="sectionbody">
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
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>Normally applications allocate large chunks for <code>VkDeviceMemory</code> and then suballocate to various buffers and images. There are times where it might be better to have a dedicated allocation for <code>VkImage</code> or <code>VkBuffer</code>. An application can pass <code>VkMemoryDedicatedRequirements</code> into <code>vkGetBufferMemoryRequirements2</code> or <code>vkGetImageMemoryRequirements2</code> to find out if a dedicated allocation is preferred or required. When dealing with external memory it will often require a dedicated allocation.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_sampler_filter_minmax">7. VK_EXT_sampler_filter_minmax</h2>
<div class="sectionbody">
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
<p>By default, Vulkan samplers using linear filtering return a filtered texel value produced by computing a weighted average of a collection of texels in the neighborhood of the texture coordinate provided. This extension provides a new sampler parameter which allows applications to produce a filtered texel value by computing a component-wise minimum (<code>VK_SAMPLER_REDUCTION_MODE_MIN</code>) or maximum (<code>VK_SAMPLER_REDUCTION_MODE_MAX</code>) of the texels that would normally be averaged. This is similar to <a href="https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_filter_minmax.txt">GL EXT_texture_filter_minmax</a>.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_sampler_mirror_clamp_to_edge">8. VK_KHR_sampler_mirror_clamp_to_edge</h2>
<div class="sectionbody">
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
<p>This extension adds a new sampler address mode (<code>VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE</code>) that effectively uses a texture map twice as large as the original image in which the additional half of the new image is a mirror image of the original image. This new mode relaxes the need to generate images whose opposite edges match by using the original image to generate a matching &#8220;mirror image&#8221;. This mode allows the texture to be mirrored only once in the negative <code>s</code>, <code>t</code>, and <code>r</code> directions.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_4444_formats-and-VK_EXT_ycbcr_2plane_444_formats">9. VK_EXT_4444_formats and VK_EXT_ycbcr_2plane_444_formats</h2>
<div class="sectionbody">
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
<p>These extensions add new <code>VkFormat</code> that were not originally in the spec</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_format_feature_flags2">10. VK_KHR_format_feature_flags2</h2>
<div class="sectionbody">
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
<p>This extension adds a new <code>VkFormatFeatureFlagBits2KHR</code> 64bits format feature flag type to extend the existing <code>VkFormatFeatureFlagBits</code> which is limited to 31 flags.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_rgba10x6_formats">11. VK_EXT_rgba10x6_formats</h2>
<div class="sectionbody">
<div class="paragraph">
<p>This extension adds an exception for <code>VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16</code> in the <a href="https://github.com/KhronosGroup/Vulkan-ValidationLayers/pull/3397">validation layers</a> to allow being able to render to the format.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="maintenance-extensions">12. Maintenance Extensions</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The maintenance extensions add a collection of minor features that were intentionally left out or overlooked from the original Vulkan 1.0 release.</p>
</div>
<div class="paragraph">
<p>Currently, there are 4 maintenance extensions. The first 3 were bundled in Vulkan 1.1 as core. All the details for each are well defined in the extension appendix page.</p>
</div>
<div class="ulist">
<ul>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VK_KHR_maintenance1">VK_KHR_maintenance1</a> - core in Vulkan 1.1</p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VK_KHR_maintenance2">VK_KHR_maintenance2</a> - core in Vulkan 1.1</p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VK_KHR_maintenance3">VK_KHR_maintenance3</a> - core in Vulkan 1.1</p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VK_KHR_maintenance4">VK_KHR_maintenance4</a> - core in Vulkan 1.3</p>
</li>
</ul>
</div>
</div>
</div>
<div class="sect1">
<h2 id="pnext-expansions">13. pNext Expansions</h2>
<div class="sectionbody">
<div class="paragraph">
<p>There have been a few times where the Vulkan Working Group realized that some structs in the original 1.0 Vulkan spec were missing the ability to be extended properly due to missing <code>sType</code>/<code>pNext</code>.</p>
</div>
<div class="paragraph">
<p>Keeping backward compatibility between versions is very important, so the best solution was to create an extension to amend the mistake. These extensions are mainly new structs, but also need to create new function entry points to make use of the new structs.</p>
</div>
<div class="paragraph">
<p>The current list of extensions that fit this category are:</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>VK_KHR_get_memory_requirements2</code></p>
<div class="ulist">
<ul>
<li>
<p>Added to core in Vulkan 1.1</p>
</li>
</ul>
</div>
</li>
<li>
<p><code>VK_KHR_get_physical_device_properties2</code></p>
<div class="ulist">
<ul>
<li>
<p>Added to core in Vulkan 1.1</p>
</li>
</ul>
</div>
</li>
<li>
<p><code>VK_KHR_bind_memory2</code></p>
<div class="ulist">
<ul>
<li>
<p>Added to core in Vulkan 1.1</p>
</li>
</ul>
</div>
</li>
<li>
<p><code>VK_KHR_create_renderpass2</code></p>
<div class="ulist">
<ul>
<li>
<p>Added to core in Vulkan 1.2</p>
</li>
</ul>
</div>
</li>
<li>
<p><code>VK_KHR_copy_commands2</code></p>
<div class="ulist">
<ul>
<li>
<p>Added to core in Vulkan 1.3</p>
</li>
</ul>
</div>
</li>
</ul>
</div>
<div class="paragraph">
<p>All of these are very simple extensions and were promoted to core in their respective versions to make it easier to use without having to query for their support.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p><code>VK_KHR_get_physical_device_properties2</code> has additional functionality as it adds the ability to query feature support for extensions and newer Vulkan versions. It has become a requirement for most other Vulkan extensions because of this.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="sect2">
<h3 id="_example">13.1. Example</h3>
<div class="paragraph">
<p>Using <code>VK_KHR_bind_memory2</code> as an example, instead of using the standard <code>vkBindImageMemory</code></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// VkImage images[3]
// VkDeviceMemory memories[2];

vkBindImageMemory(myDevice, images[0], memories[0], 0);
vkBindImageMemory(myDevice, images[1], memories[0], 64);
vkBindImageMemory(myDevice, images[2], memories[1], 0);</code></pre>
</div>
</div>
<div class="paragraph">
<p>They can now be batched together</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// VkImage images[3];
// VkDeviceMemory memories[2];

VkBindImageMemoryInfo infos[3];
infos[0] = {VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, NULL, images[0], memories[0], 0};
infos[1] = {VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, NULL, images[1], memories[0], 64};
infos[2] = {VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, NULL, images[2], memories[1], 0};

vkBindImageMemory2(myDevice, 3, infos);</code></pre>
</div>
</div>
<div class="paragraph">
<p>Some extensions such as <code>VK_KHR_sampler_ycbcr_conversion</code> expose structs that can be passed into the <code>pNext</code></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkBindImagePlaneMemoryInfo plane_info[2];
plane_info[0] = {VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO, NULL, VK_IMAGE_ASPECT_PLANE_0_BIT};
plane_info[1] = {VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO, NULL, VK_IMAGE_ASPECT_PLANE_1_BIT};

// Can now pass other extensions structs into the pNext missing from vkBindImagemMemory()
VkBindImageMemoryInfo infos[2];
infos[0] = {VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, &amp;plane_info[0], image, memories[0], 0};
infos[1] = {VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, &amp;plane_info[1], image, memories[1], 0};

vkBindImageMemory2(myDevice, 2, infos);</code></pre>
</div>
</div>
</div>
<div class="sect2">
<h3 id="_it_is_fine_to_not_use_these">13.2. It is fine to not use these</h3>
<div class="paragraph">
<p>Unless an application need to make use of one of the extensions that rely on the above extensions, it is normally ok to use the original function/structs still.</p>
</div>
<div class="paragraph">
<p>One possible way to handle this is as followed:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">void HandleVkBindImageMemoryInfo(const VkBindImageMemoryInfo* info) {
    // ...
}

//
// Entry points into tool/implementation
//
void vkBindImageMemory(VkDevice device,
                       VkImage image,
                       VkDeviceMemory memory,
                       VkDeviceSize memoryOffset)
{
    VkBindImageMemoryInfo info;
    // original call doesn't have a pNext or sType
    info.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO;
    info.pNext = nullptr;

    // Match the rest of struct the same
    info.image = image;
    info.memory = memory;
    info.memoryOffset = memoryOffset;

    HandleVkBindImageMemoryInfo(&amp;info);
}

void vkBindImageMemory2(VkDevice device,
                        uint32_t bindInfoCount, const
                        VkBindImageMemoryInfo* pBindInfos)
{
    for (uint32_t i = 0; i &lt; bindInfoCount; i++) {
        HandleVkBindImageMemoryInfo(pBindInfos[i]);
    }
}</code></pre>
</div>
</div>
</div>
</div>
</div>