<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#formats">Formats</a>
<ul class="sectlevel1">
<li><a href="#feature-support">1. Feature Support</a>
<ul class="sectlevel2">
<li><a href="#_format_feature_query_example">1.1. Format Feature Query Example</a></li>
</ul>
</li>
<li><a href="#_variations_of_formats">2. Variations of Formats</a>
<ul class="sectlevel2">
<li><a href="#_color">2.1. Color</a></li>
<li><a href="#_depth_and_stencil">2.2. Depth and Stencil</a></li>
<li><a href="#_compressed">2.3. Compressed</a></li>
<li><a href="#_planar">2.4. Planar</a></li>
<li><a href="#_packed">2.5. Packed</a></li>
<li><a href="#_external">2.6. External</a></li>
</ul>
</li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink: /Notes/004-3d-rendering/vulkan/chapters/formats.html
---</p>
</div>
<h1 id="formats" class="sect0">Formats</h1>
<div class="paragraph">
<p>Vulkan formats are used to describe how memory is laid out. This chapter aims to give a high-level overview of the variations of formats in Vulkan and some logistical information for how to use them. All details are already well specified in both the <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#formats">Vulkan Spec format chapter</a> and the <a href="https://www.khronos.org/registry/DataFormat/specs/1.3/dataformat.1.3.html">Khronos Data Format Specification</a>.</p>
</div>
<div class="paragraph">
<p>The most common use case for a <code>VkFormat</code> is when creating a <code>VkImage</code>. Because the <code>VkFormat</code>&#8203;s are well defined, they are also used when describing the memory layout for things such as a <code>VkBufferView</code>, <a href="vertex_input_data_processing.html#input-attribute-format">vertex input attribute</a>, <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#spirvenv-image-formats">mapping SPIR-V image formats</a>, creating <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkAccelerationStructureGeometryTrianglesDataKHR.html">triangle geometry in a bottom-level acceleration structure</a>, etc.</p>
</div>
<div class="sect1">
<h2 id="feature-support">1. Feature Support</h2>
<div class="sectionbody">
<div class="paragraph">
<p>It is important to understand that "format support" is not a single binary value per format, but rather each format has a set of <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkFormatFeatureFlagBits.html">VkFormatFeatureFlagBits</a> that each describes with features are supported for a format.</p>
</div>
<div class="paragraph">
<p>The supported formats may vary across implementations, but a <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#features-required-format-support">minimum set of format features are guaranteed</a>. An application can <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#formats-properties">query</a> for the supported format properties.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Both <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_get_physical_device_properties2.html">VK_KHR_get_physical_device_properties2</a> and <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_format_feature_flags2.html">VK_KHR_format_feature_flags2</a> expose another way to query for format features.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="sect2">
<h3 id="_format_feature_query_example">1.1. Format Feature Query Example</h3>
<div class="paragraph">
<p>In this example, the code will check if the <code>VK_FORMAT_R8_UNORM</code> format supports being sampled from a <code>VkImage</code> created with <code>VK_IMAGE_TILING_LINEAR</code> for <code>VkImageCreateInfo::tiling</code>. To do this, the code will query the <code>linearTilingFeatures</code> flags for <code>VK_FORMAT_R8_UNORM</code> to see if the <code>VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT</code> is supported by the implementation.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// Using core Vulkan 1.0
VkFormatProperties formatProperties;
vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R8_UNORM, &amp;formatProperties);
if ((formatProperties.linearTilingFeatures &amp; VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0) {
    // supported
} else {
    // not supported
}</code></pre>
</div>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// Using core Vulkan 1.1 or VK_KHR_get_physical_device_properties2
VkFormatProperties2 formatProperties2;
formatProperties2.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
formatProperties2.pNext = nullptr; // used for possible extensions

vkGetPhysicalDeviceFormatProperties2(physicalDevice, VK_FORMAT_R8_UNORM, &amp;formatProperties2);
if ((formatProperties2.formatProperties.linearTilingFeatures &amp; VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0) {
    // supported
} else {
    // not supported
}</code></pre>
</div>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// Using VK_KHR_format_feature_flags2
VkFormatProperties3KHR formatProperties3;
formatProperties2.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3_KHR;
formatProperties2.pNext = nullptr;

VkFormatProperties2 formatProperties2;
formatProperties2.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
formatProperties2.pNext = &amp;formatProperties3;

vkGetPhysicalDeviceFormatProperties2(physicalDevice, VK_FORMAT_R8_UNORM, &amp;formatProperties2);
if ((formatProperties3.linearTilingFeatures &amp; VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT_KHR) != 0) {
    // supported
} else {
    // not supported
}</code></pre>
</div>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_variations_of_formats">2. Variations of Formats</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Formats come in many variations, most can be grouped by the <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#_identification_of_formats">name of the format</a>. When dealing with images, the  <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkImageAspectFlagBits.html">VkImageAspectFlagBits</a> values are used to represent which part of the data is being accessed for operations such as clears and copies.</p>
</div>
<div class="sect2">
<h3 id="_color">2.1. Color</h3>
<div class="paragraph">
<p>Format with a <code>R</code>, <code>G</code>, <code>B</code> or <code>A</code> component and accessed with the <code>VK_IMAGE_ASPECT_COLOR_BIT</code></p>
</div>
</div>
<div class="sect2">
<h3 id="_depth_and_stencil">2.2. Depth and Stencil</h3>
<div class="paragraph">
<p>Formats with a <code>D</code> or <code>S</code> component. These formats are <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#formats-depth-stencil">considered opaque</a> and have special rules when it comes to <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VkBufferImageCopy">copy to and from</a> depth/stencil images.</p>
</div>
<div class="paragraph">
<p>Some formats have both a depth and stencil component and can be accessed separately with <code>VK_IMAGE_ASPECT_DEPTH_BIT</code> and <code>VK_IMAGE_ASPECT_STENCIL_BIT</code>.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_separate_depth_stencil_layouts.html">VK_KHR_separate_depth_stencil_layouts</a> and <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_EXT_separate_stencil_usage.html">VK_EXT_separate_stencil_usage</a>, which are both promoted to Vulkan 1.2, can be used to have finer control between the depth and stencil components.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>More information about depth format can also be found in the <a href="depth.html#depth-formats">depth chapter</a>.</p>
</div>
</div>
<div class="sect2">
<h3 id="_compressed">2.3. Compressed</h3>
<div class="paragraph">
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#compressed_image_formats">Compressed image formats</a>
representation of multiple pixels encoded interdependently within a region.</p>
</div>
<table class="tableblock frame-all grid-all stretch">
<caption class="title">Table 1. Vulkan Compressed Image Formats</caption>
<colgroup>
<col style="width: 50%;">
<col style="width: 50%;">
</colgroup>
<thead>
<tr>
<th class="tableblock halign-left valign-top">Format</th>
<th class="tableblock halign-left valign-top">How to enable</th>
</tr>
</thead>
<tbody>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#appendix-compressedtex-bc">BC (Block-Compressed)</a></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>VkPhysicalDeviceFeatures::textureCompressionBC</code></p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#appendix-compressedtex-etc2">ETC2 and EAC</a></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>VkPhysicalDeviceFeatures::textureCompressionETC2</code></p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#appendix-compressedtex-astc">ASTC LDR</a></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock"><code>VkPhysicalDeviceFeatures::textureCompressionASTC_LDR</code></p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#appendix-compressedtex-astc">ASTC HDR</a></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock"><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_EXT_texture_compression_astc_hdr.html">VK_EXT_texture_compression_astc_hdr</a></p></td>
</tr>
<tr>
<td class="tableblock halign-left valign-top"><p class="tableblock"><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#appendix-compressedtex-pvrtc">PVRTC</a></p></td>
<td class="tableblock halign-left valign-top"><p class="tableblock"><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_IMG_format_pvrtc.html">VK_IMG_format_pvrtc</a></p></td>
</tr>
</tbody>
</table>
</div>
<div class="sect2">
<h3 id="_planar">2.4. Planar</h3>
<div class="paragraph">
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_sampler_ycbcr_conversion.html">VK_KHR_sampler_ycbcr_conversion</a> and <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_EXT_ycbcr_2plane_444_formats.html">VK_EXT_ycbcr_2plane_444_formats</a> add <a href="VK_KHR_sampler_ycbcr_conversion.html#multi-planar-formats">multi-planar formats</a> to Vulkan. The planes can be accessed separately with <code>VK_IMAGE_ASPECT_PLANE_0_BIT</code>, <code>VK_IMAGE_ASPECT_PLANE_1_BIT</code>, and <code>VK_IMAGE_ASPECT_PLANE_2_BIT</code>.</p>
</div>
</div>
<div class="sect2">
<h3 id="_packed">2.5. Packed</h3>
<div class="paragraph">
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#formats-packed">Packed formats</a> are for the purposes of address alignment. As an example, <code>VK_FORMAT_A8B8G8R8_UNORM_PACK32</code> and <code>VK_FORMAT_R8G8B8A8_UNORM</code> might seem very similar, but when using the formula from the <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#fxvertex-input-extraction">Vertex Input Extraction section of the spec</a></p>
</div>
<div class="quoteblock">
<blockquote>
<div class="paragraph">
<p>attribAddress = bufferBindingAddress + vertexOffset + attribDesc.offset;</p>
</div>
</blockquote>
</div>
<div class="paragraph">
<p>For <code>VK_FORMAT_R8G8B8A8_UNORM</code> the <code>attribAddress</code> has to be a multiple of the component size (8 bits) while <code>VK_FORMAT_A8B8G8R8_UNORM_PACK32</code> has to be a multiple of the packed size (32 bits).</p>
</div>
</div>
<div class="sect2">
<h3 id="_external">2.6. External</h3>
<div class="paragraph">
<p>Currently only supported with the <code>VK_ANDROID_external_memory_android_hardware_buffer</code> extension. This extension allows Android applications to import implementation-defined external formats to be used with a <a href="VK_KHR_sampler_ycbcr_conversion.html">VkSamplerYcbcrConversion</a>. There are many restrictions what are allowed with these external formats which are <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#memory-external-android-hardware-buffer-external-formats">documented in the spec</a>.</p>
</div>
</div>
</div>
</div>