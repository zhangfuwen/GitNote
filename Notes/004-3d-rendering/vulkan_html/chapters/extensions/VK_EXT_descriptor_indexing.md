<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#VK_EXT_descriptor_indexing">VK_EXT_descriptor_indexing</a>
<ul class="sectlevel1">
<li><a href="#_update_after_bind">1. Update after Bind</a></li>
<li><a href="#_partially_bound">2. Partially bound</a></li>
<li><a href="#_dynamic_indexing">3. Dynamic Indexing</a></li>
<li><a href="#_dynamic_non_uniform_indexing">4. Dynamic Non-Uniform Indexing</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/extensions/VK_EXT_descriptor_indexing.html
layout: default
---</p>
</div>
<h1 id="VK_EXT_descriptor_indexing" class="sect0">VK_EXT_descriptor_indexing</h1>
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
<p><a href="https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/EXT/SPV_EXT_descriptor_indexing.html">SPV_EXT_descriptor_indexing</a></p>
</div>
<div class="paragraph">
<p><a href="https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_nonuniform_qualifier.txt">GLSL - GL_EXT_nonuniform_qualifier</a></p>
</div>
<div class="paragraph">
<p>Presentation from Montreal Developer Day (<a href="https://www.youtube.com/watch?v=tXipcoeuNh4">video</a> and <a href="https://www.khronos.org/assets/uploads/developers/library/2018-vulkan-devday/11-DescriptorUpdateTemplates.pdf">slides</a>)</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>This extension was designed to be broken down into a few different, smaller features to allow implementations to add support for the each feature when possible.</p>
</div>
<div class="sect1">
<h2 id="_update_after_bind">1. Update after Bind</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Without this extension, descriptors in an application are not allowed to update between recording the command buffer and the execution of the command buffers. With this extension an application can querying for <code>descriptorBinding*UpdateAfterBind</code> support for the type of descriptor being used which allows an application to then update in between recording and execution.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="title">Example</div>
<div class="paragraph">
<p>If an application has a <code>StorageBuffer</code> descriptor, then it will query for <code>descriptorBindingStorageBufferUpdateAfterBind</code> support.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>After enabling the desired feature support for updating after bind, an application needs to setup the following in order to use a descriptor that can update after bind:</p>
</div>
<div class="ulist">
<ul>
<li>
<p>The <code>VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT</code> flag for any <code>VkDescriptorPool</code> the descriptor is allocated from.</p>
</li>
<li>
<p>The <code>VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT</code> flag for any <code>VkDescriptorSetLayout</code> the descriptor is from.</p>
</li>
<li>
<p>The <code>VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT</code> for each binding in the <code>VkDescriptorSetLayout</code> that the descriptor will use.</p>
</li>
</ul>
</div>
<div class="paragraph">
<p>The following code example gives an idea of the difference between enabling update after bind and without it:</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/VK_EXT_descriptor_indexing_update_after_bind.png" alt="VK_EXT_descriptor_indexing_update_after_bind.png">
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_partially_bound">2. Partially bound</h2>
<div class="sectionbody">
<div class="paragraph">
<p>With the <code>descriptorBindingPartiallyBound</code> feature and using <code>VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT</code> in the <code>VkDescriptorSetLayoutBindingFlagsCreateInfo::pBindingFlags</code> an application developer isn&#8217;t required to update all the descriptors at time of use.</p>
</div>
<div class="paragraph">
<p>An example would be if an application&#8217;s GLSL has</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">layout(set = 0, binding = 0) uniform sampler2D textureSampler[64];</code></pre>
</div>
</div>
<div class="paragraph">
<p>but only binds the first 32 slots in the array. This also relies on the the application knowing that it will not index into the unbound slots in the array.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_dynamic_indexing">3. Dynamic Indexing</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Normally when an application indexes into an array of bound descriptors the index needs to be known at compile time. With the <code>shader*ArrayDynamicIndexing</code> feature, a certain type of descriptor can be indexed by &#8220;dynamically uniform&#8221; integers. This was already supported as a <code>VkPhysicalDeviceFeatures</code> for most descriptors, but this extension adds <code>VkPhysicalDeviceDescriptorIndexingFeatures</code> struct that lets implementations expose support for dynamic uniform indexing of input attachments, uniform texel buffers, and storage texel buffers as well.</p>
</div>
<div class="paragraph">
<p>The key word here is &#8220;uniform&#8221; which means that all invocations in a SPIR-V Invocation Group need to all use the same dynamic index. This translates to either all invocations in a single <code>vkCmdDraw*</code> call or a single workgroup of a <code>vkCmdDispatch*</code> call.</p>
</div>
<div class="paragraph">
<p>An example of dynamic uniform indexing in GLSL</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">layout(set = 0, binding = 0) uniform sampler2D mySampler[64];
layout(set = 0, binding = 1) uniform UniformBufferObject {
    int textureId;
} ubo;

// ...

void main() {
    // ...
    vec4 samplerColor = texture(mySampler[ubo.textureId], uvCoords);
    // ...
}</code></pre>
</div>
</div>
<div class="paragraph">
<p>This example is &#8220;dynamic&#8221; as it is will not be known until runtime what the value of <code>ubo.textureId</code> is. This is also &#8220;uniform&#8221; as all invocations will use <code>ubo.textureId</code> in this shader.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_dynamic_non_uniform_indexing">4. Dynamic Non-Uniform Indexing</h2>
<div class="sectionbody">
<div class="paragraph">
<p>To be dynamically <strong>non-uniform</strong> means that it is possible that invocations might index differently into an array of descriptors, but it won&#8217;t be known until runtime. This extension exposes in <code>VkPhysicalDeviceDescriptorIndexingFeatures</code> a set of <code>shader*ArrayNonUniformIndexing</code> feature bits to show which descriptor types an implementation supports dynamic non-uniform indexing for. The SPIR-V extension adds a <code>NonUniform</code> decoration which can be set in GLSL with the help of the <code>nonuniformEXT</code> keyword added.</p>
</div>
<div class="paragraph">
<p>An example of dynamic non-uniform indexing in GLSL</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-glsl" data-lang="glsl">#version450
#extension GL_EXT_nonuniform_qualifier : enable

layout(set = 0, binding = 0) uniform sampler2D mySampler[64];
layout(set = 0, binding = 1) uniform UniformBufferObject {
    int textureId;
} ubo;

// ...

void main() {
    // ...
    if (uvCoords.x &gt; runtimeThreshold) {
        index = 0;
    } else {
        index = 1;
    }
    vec4 samplerColor = texture(mySampler[nonuniformEXT(index)], uvCoords);
    // ...
}</code></pre>
</div>
</div>
<div class="paragraph">
<p>This example is non-uniform as some invocations index a <code>mySampler[0]</code> and some at <code>mySampler[1]</code>. The <code>nonuniformEXT()</code> is needed in this case.</p>
</div>
</div>
</div>