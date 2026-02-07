<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#subgroups">Subgroups</a>
<ul class="sectlevel1">
<li><a href="#_resources">1. Resources</a></li>
<li><a href="#_subgroup_size">2. Subgroup size</a>
<ul class="sectlevel2">
<li><a href="#VK_EXT_subgroup_size_control">2.1. VK_EXT_subgroup_size_control</a></li>
</ul>
</li>
<li><a href="#_checking_for_support">3. Checking for support</a>
<ul class="sectlevel2">
<li><a href="#_guaranteed_support">3.1. Guaranteed support</a></li>
</ul>
</li>
<li><a href="#VK_KHR_shader_subgroup_extended_types">4. VK_KHR_shader_subgroup_extended_types</a></li>
<li><a href="#VK_EXT_shader_subgroup_ballot-and-VK_EXT_shader_subgroup_vote">5. VK_EXT_shader_subgroup_ballot and VK_EXT_shader_subgroup_vote</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/subgroups.html
layout: default
---</p>
</div>
<h1 id="subgroups" class="sect0">Subgroups</h1>
<div class="paragraph">
<p>The Vulkan Spec defines subgroups as:</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>A set of shader invocations that can synchronize and share data with each other efficiently. In compute shaders, the local workgroup is a superset of the subgroup.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>For many implementations, a subgroup is the groups of invocations that run the same instruction at once. Subgroups allow for a shader writer to work at a finer granularity than a single workgroup.</p>
</div>
<div class="sect1">
<h2 id="_resources">1. Resources</h2>
<div class="sectionbody">
<div class="paragraph">
<p>For more detailed information about subgroups there is a great <a href="https://www.khronos.org/blog/vulkan-subgroup-tutorial">Khronos blog post</a> as well as a presentation from Vulkan Developer Day 2018 (<a href="https://www.khronos.org/assets/uploads/developers/library/2018-vulkan-devday/06-subgroups.pdf">slides</a> and <a href="https://www.youtube.com/watch?v=8MyqQLu_tW0">video</a>). GLSL support can be found in the <a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/khr/GL_KHR_shader_subgroup.txt">GL_KHR_shader_subgroup</a> extension.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_subgroup_size">2. Subgroup size</h2>
<div class="sectionbody">
<div class="paragraph">
<p>It is important to also realize the size of a subgroup can be dynamic for an implementation. Some implementations may dispatch shaders with a varying subgroup size for different subgroups. As a result, they could implicitly split a large subgroup into smaller subgroups or represent a small subgroup as a larger subgroup, some of whose invocations were inactive on launch.</p>
</div>
<div class="sect2">
<h3 id="VK_EXT_subgroup_size_control">2.1. VK_EXT_subgroup_size_control</h3>
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
<p>This extension was created due to some implementation having more than one subgroup size and Vulkan originally only exposing a single subgroup size.</p>
</div>
<div class="paragraph">
<p>For example, if an implementation has both support for subgroups of size <code>4</code> and <code>16</code> before they would have had to expose only one size, but now can expose both. This allows applications to potentially control the hardware at a finer granularity for implementations that expose multiple subgroup sizes. If an device does not support this extension, it most likely means there is only one supported subgroup size to expose.</p>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_checking_for_support">3. Checking for support</h2>
<div class="sectionbody">
<div class="paragraph">
<p>With Vulkan 1.1, all the information for subgroups is found in <code>VkPhysicalDeviceSubgroupProperties</code></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkPhysicalDeviceSubgroupProperties subgroupProperties;

VkPhysicalDeviceProperties2KHR deviceProperties2;
deviceProperties2.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
deviceProperties2.pNext      = &amp;subgroupProperties;
vkGetPhysicalDeviceProperties2(physicalDevice, &amp;deviceProperties2);

// Example of checking if supported in fragment shader
if ((subgroupProperties.supportedStages &amp; VK_SHADER_STAGE_FRAGMENT_BIT) != 0) {
    // fragment shaders supported
}

// Example of checking if ballot is supported
if ((subgroupProperties.supportedOperations &amp; VK_SUBGROUP_FEATURE_BALLOT_BIT) != 0) {
    // ballot subgroup operations supported
}</code></pre>
</div>
</div>
<div class="sect2">
<h3 id="_guaranteed_support">3.1. Guaranteed support</h3>
<div class="paragraph">
<p>For supported stages, the Vulkan Spec guarantees the following support:</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p><strong>supportedStages</strong> will have the <strong>VK_SHADER_STAGE_COMPUTE_BIT</strong> bit set if any of the physical device&#8217;s queues support <strong>VK_QUEUE_COMPUTE_BIT</strong>.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>For supported operations, the Vulkan Spec guarantees the following support:</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p><strong>supportedOperations</strong> will have the <strong>VK_SUBGROUP_FEATURE_BASIC_BIT</strong> bit set if any of the physical device&#8217;s queues support <strong>VK_QUEUE_GRAPHICS_BIT</strong> or <strong>VK_QUEUE_COMPUTE_BIT</strong>.</p>
</div>
</td>
</tr>
</table>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_KHR_shader_subgroup_extended_types">4. VK_KHR_shader_subgroup_extended_types</h2>
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
<div class="paragraph">
<p><a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GLSL_EXT_shader_subgroup_extended_types.txt">GLSL_EXT_shader_subgroup_extended_types</a></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension allows subgroup operations to use 8-bit integer, 16-bit integer, 64-bit integer, 16-bit floating-point, and vectors of these types in group operations with subgroup scope if the implementation supports the types already.</p>
</div>
<div class="paragraph">
<p>For example, if an implementation supports 8-bit integers an application can now use the GLSL <code>genI8Type subgroupAdd(genI8Type value);</code> call which will get mapped to <code>OpGroupNonUniformFAdd</code> in SPIR-V.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="VK_EXT_shader_subgroup_ballot-and-VK_EXT_shader_subgroup_vote">5. VK_EXT_shader_subgroup_ballot and VK_EXT_shader_subgroup_vote</h2>
<div class="sectionbody">
<div class="paragraph">
<p><code>VK_EXT_shader_subgroup_ballot</code> and <code>VK_EXT_shader_subgroup_vote</code> were the original efforts to expose subgroups in Vulkan. If an application is using Vulkan 1.1 or greater, there is no need to use these extensions and should instead use the core API to query for subgroup support.</p>
</div>
</div>
</div>