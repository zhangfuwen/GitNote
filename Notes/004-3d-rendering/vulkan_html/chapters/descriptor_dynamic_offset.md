<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#descriptor-dynamic-offset">Descriptor Dynamic Offset</a>
<ul class="sectlevel1">
<li><a href="#_example">1. Example</a></li>
<li><a href="#_example_with_vk_whole_size">2. Example with VK_WHOLE_SIZE</a></li>
<li><a href="#_limits">3. Limits</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink: /Notes/004-3d-rendering/vulkan/chapters/descriptor_dynamic_offset.html
---</p>
</div>
<h1 id="descriptor-dynamic-offset" class="sect0">Descriptor Dynamic Offset</h1>
<div class="paragraph">
<p>Vulkan offers two types of descriptors that allow adjusting the offset at bind time as <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#descriptorsets-binding-dynamicoffsets">defined in the spec</a>.</p>
</div>
<div class="ulist">
<ul>
<li>
<p>dynamic uniform buffer (<code>VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC</code>)</p>
</li>
<li>
<p>dynamic storage buffer (<code>VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC</code>)</p>
</li>
</ul>
</div>
<div class="sect1">
<h2 id="_example">1. Example</h2>
<div class="sectionbody">
<div class="paragraph">
<p>This example will have buffer of 32 bytes and 16 of the bytes will be set at <code>vkUpdateDescriptorSets</code> time. In this first example, we will not add any dynamic offset.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-c" data-lang="c">VkDescriptorSet descriptorSet; // allocated
VkBuffer buffer; // size of 32 bytes

VkDescriptorBufferInfo bufferInfo = {
    buffer,
    4,      // offset
    16      // range
};

VkWriteDescriptorSet writeInfo = {
    .dstSet = descriptorSet,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
    .pBufferInfo = bufferInfo
};

vkUpdateDescriptorSets(
    1,         // descriptorWriteCount,
    &amp;writeInfo // pDescriptorWrites,
);

// No dynamic offset
vkCmdBindDescriptorSets(
    1,              // descriptorSetCount,
    &amp;descriptorSet, // pDescriptorSets,
    0,              // dynamicOffsetCount
    NULL            // pDynamicOffsets
);</code></pre>
</div>
</div>
<div class="paragraph">
<p>Our buffer now currently looks like the following:</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/descriptor_dynamic_offset_example_a.png" alt="descriptor_dynamic_offset_example_a.png">
</div>
</div>
<div class="paragraph">
<p>Next, a 8 byte dynamic offset will applied at bind time.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-c" data-lang="c">uint32_t offsets[1] = { 8 };
vkCmdBindDescriptorSets(
    1,              // descriptorSetCount,
    &amp;descriptorSet, // pDescriptorSets,
    1,              // dynamicOffsetCount
    offsets         // pDynamicOffsets
);</code></pre>
</div>
</div>
<div class="paragraph">
<p>Our buffer currently looks like the following:</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/descriptor_dynamic_offset_example_b.png" alt="descriptor_dynamic_offset_example_b.png">
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_example_with_vk_whole_size">2. Example with VK_WHOLE_SIZE</h2>
<div class="sectionbody">
<div class="paragraph">
<p>This time the <code>VK_WHOLE_SIZE</code> value will be used for the range. Everything looks the same as the above example except the <code>VkDescriptorBufferInfo::range</code></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-c" data-lang="c">VkDescriptorSet descriptorSet; // allocated
VkBuffer buffer; // size of 32 bytes

VkDescriptorBufferInfo info = {
    buffer,
    4,             // offset
    VK_WHOLE_SIZE  // range
};

VkWriteDescriptorSet writeInfo = {
    .dstSet = descriptorSet,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
    .pBufferInfo = bufferInfo
};

vkUpdateDescriptorSets(
    1,         // descriptorWriteCount,
    &amp;writeInfo // pDescriptorWrites,
);

// No dynamic offset
vkCmdBindDescriptorSets(
    1,              // descriptorSetCount,
    &amp;descriptorSet, // pDescriptorSets,
    0,              // dynamicOffsetCount
    NULL            // pDynamicOffsets
);</code></pre>
</div>
</div>
<div class="paragraph">
<p>Our buffer currently looks like the following:</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/descriptor_dynamic_offset_example_c.png" alt="descriptor_dynamic_offset_example_c.png">
</div>
</div>
<div class="paragraph">
<p>This time, if we attempt to apply a dynamic offset it will be met with undefined behavior and the <a href="https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/2846">validation layers will give an error</a></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-c" data-lang="c">// Invalid
uint32_t offsets[1] = { 8 };
vkCmdBindDescriptorSets(
    1,              // descriptorSetCount,
    &amp;descriptorSet, // pDescriptorSets,
    1,              // dynamicOffsetCount
    offsets         // pDynamicOffsets
);</code></pre>
</div>
</div>
<div class="paragraph">
<p>This is what it looks like with the invalid dynamic offset</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/descriptor_dynamic_offset_example_d.png" alt="descriptor_dynamic_offset_example_d.png">
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_limits">3. Limits</h2>
<div class="sectionbody">
<div class="paragraph">
<p>It is important to also check the <code>minUniformBufferOffsetAlignment</code> and <code>minStorageBufferOffsetAlignment</code> as both the base offset and dynamic offset must be multiples of these limits.</p>
</div>
</div>
</div>