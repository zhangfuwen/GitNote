<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#VK_KHR_synchronization2">VK_KHR_synchronization2</a>
<ul class="sectlevel1">
<li><a href="#_rethinking_pipeline_stages_and_access_flags">1. Rethinking Pipeline Stages and Access Flags</a>
<ul class="sectlevel2">
<li><a href="#_adding_barriers_for_setting_events">1.1. Adding barriers for setting events</a></li>
</ul>
</li>
<li><a href="#_reusing_the_same_pipeline_stage_and_access_flag_names">2. Reusing the same pipeline stage and access flag names</a></li>
<li><a href="#_vksubpassdependency">3. VkSubpassDependency</a></li>
<li><a href="#_splitting_up_pipeline_stages_and_access_masks">4. Splitting up pipeline stages and access masks</a>
<ul class="sectlevel2">
<li><a href="#_splitting_up_vk_pipeline_stage_vertex_input_bit">4.1. Splitting up VK_PIPELINE_STAGE_VERTEX_INPUT_BIT</a></li>
<li><a href="#_splitting_up_vk_pipeline_stage_all_transfer_bit">4.2. Splitting up VK_PIPELINE_STAGE_ALL_TRANSFER_BIT</a></li>
<li><a href="#_splitting_up_vk_access_shader_read_bit">4.3. Splitting up VK_ACCESS_SHADER_READ_BIT</a></li>
<li><a href="#_combining_shader_stages_for_pre_rasterization">4.4. Combining shader stages for pre-rasterization</a></li>
</ul>
</li>
<li><a href="#_vk_access_shader_write_bit_alias">5. VK_ACCESS_SHADER_WRITE_BIT alias</a></li>
<li><a href="#_top_of_pipe_and_bottom_of_pipe_deprecation">6. TOP_OF_PIPE and BOTTOM_OF_PIPE deprecation</a></li>
<li><a href="#_making_use_of_new_image_layouts">7. Making use of new image layouts</a></li>
<li><a href="#_new_submission_flow">8. New submission flow</a></li>
<li><a href="#_emulation_layer">9. Emulation Layer</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/extensions/VK_KHR_synchronization2.html
layout: default
---</p>
</div>
<h1 id="VK_KHR_synchronization2" class="sect0">VK_KHR_synchronization2</h1>
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
<p>The <code>VK_KHR_synchronization2</code> extension provides improvements to pipeline barriers, events, image layout transitions and queue submission. This document shows the difference between the original Vulkan synchronization operations and those provided by the extension. There are also examples of how to update application code to make use of the extension.</p>
</div>
<div class="sect1">
<h2 id="_rethinking_pipeline_stages_and_access_flags">1. Rethinking Pipeline Stages and Access Flags</h2>
<div class="sectionbody">
<div class="paragraph">
<p>One main change with the extension is to have pipeline stages and access flags now specified together in memory barrier structures. This makes the connection between the two more obvious.</p>
</div>
<div class="paragraph">
<p>The only new type of structure needed is <code>VkDependencyInfoKHR</code>, which wraps all the barriers into a single location.</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/VK_KHR_synchronization2_stage_access.png" alt="VK_KHR_synchronization2_stage_access">
</div>
</div>
<div class="sect2">
<h3 id="_adding_barriers_for_setting_events">1.1. Adding barriers for setting events</h3>
<div class="paragraph">
<p>Note that with the introduction of <code>VkDependencyInfoKHR</code> that <code>vkCmdSetEvent2KHR</code>, unlike <code>vkCmdSetEvent</code>, has the ability to add barriers. This was added to allow the <code>VkEvent</code> to be more useful. Because the implementation of a synchronization2 <code>VkEvent</code> is likely to be substantially different from a Vulkan 1.2 <code>VkEvent</code>, you must not mix extension and core api calls for a single <code>VkEvent</code>. For example, you must not call <code>vkCmdSetEvent2KHR()</code> and then <code>vkCmdWaitEvents()</code>.</p>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_reusing_the_same_pipeline_stage_and_access_flag_names">2. Reusing the same pipeline stage and access flag names</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Due to running out of the 32 bits for <code>VkAccessFlag</code> the <code>VkAccessFlags2KHR</code> type was created with a 64-bit range. To prevent the same issue for <code>VkPipelineStageFlags</code>, the <code>VkPipelineStageFlags2KHR</code> type was also created with a 64-bit range.</p>
</div>
<div class="paragraph">
<p>64-bit enumeration types are not available in all C/C++ compilers, so the code for the new fields uses <code>static const</code> values instead of an enum. As a result of this, there are no equivalent types to <code>VkPipelineStageFlagBits</code> and <code>VkAccessFlagBits</code>. Some code, including Vulkan functions such as <code>vkCmdWriteTimestamp()</code>, used the <code>Bits</code> type to indicate that the caller could only pass in a single bit value, rather than a mask of multiple bits. These calls need to be converted to take the <code>Flags</code> type and enforce the &#8220;only 1-bit&#8221; limitation via Valid Usage or the appropriate coding convention for your own code, as was done for <code>vkCmdWriteTimestamp2KHR()</code>.</p>
</div>
<div class="paragraph">
<p>The new flags include identical bits to the original synchronization flags, with the same base name and identical values.
Old flags can be used directly in the new APIs, subject to any typecasting constraints of the coding environment.
The following 2 examples show the naming differences:</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT</code> to <code>VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR</code></p>
</li>
<li>
<p><code>VK_ACCESS_SHADER_READ_BIT</code> to <code>VK_ACCESS_2_SHADER_READ_BIT_KHR</code></p>
</li>
</ul>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_vksubpassdependency">3. VkSubpassDependency</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Updating the use of the pipeline stages and access flags in <code>VkSubpassDependency</code> requires simply using <code>VkSubpassDependency2</code> which can have a <code>VkMemoryBarrier2KHR</code> passed in the <code>pNext</code></p>
</div>
<div class="paragraph">
<p>Example would be taking</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// Without VK_KHR_synchronization2
VkSubpassDependency dependency = {
    .srcSubpass = 0,
    .dstSubpass = 1,
    .srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
    .dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
    .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT
};</code></pre>
</div>
</div>
<div class="paragraph">
<p>and turning it into</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// With VK_KHR_synchronization2
VkMemoryBarrier2KHR memoryBarrier = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
    .pNext = nullptr,
    .srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT_KHR |
                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT_KHR,
    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT_KHR,
    .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT_KHR,
    .dstAccessMask = VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT_KHR
}

// The 4 fields unset are ignored according to the spec
// When VkMemoryBarrier2KHR is passed into pNext
VkSubpassDependency2 dependency = {
    .sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
    .pNext = &amp;memoryBarrier,
    .srcSubpass = 0,
    .dstSubpass = 1,
    .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT
};</code></pre>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_splitting_up_pipeline_stages_and_access_masks">4. Splitting up pipeline stages and access masks</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Some <code>VkAccessFlags</code> and <code>VkPipelineStageFlags</code> had values that were ambiguous to what it was targeting in hardware. The new <code>VkAccessFlags2KHR</code> and <code>VkPipelineStageFlags2KHR</code> break these up for some cases while leaving the old value for maintability.</p>
</div>
<div class="sect2">
<h3 id="_splitting_up_vk_pipeline_stage_vertex_input_bit">4.1. Splitting up VK_PIPELINE_STAGE_VERTEX_INPUT_BIT</h3>
<div class="paragraph">
<p>The <code>VK_PIPELINE_STAGE_VERTEX_INPUT_BIT</code> (now <code>VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT_KHR</code>) was split into 2 new stage flags which specify a dedicated stage for both the index input and the vertex input instead of having them combined into a single pipeline stage flag.</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT_KHR</code></p>
</li>
<li>
<p><code>VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT_KHR</code></p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_splitting_up_vk_pipeline_stage_all_transfer_bit">4.2. Splitting up VK_PIPELINE_STAGE_ALL_TRANSFER_BIT</h3>
<div class="paragraph">
<p>The <code>VK_PIPELINE_STAGE_ALL_TRANSFER_BIT</code> (now <code>VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT_KHR</code>) was split into 4 new stage flags which specify a dedicated stage for the various staging commands instead of having them combined into a single pipeline stage flag.</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>VK_PIPELINE_STAGE_2_COPY_BIT_KHR</code></p>
</li>
<li>
<p><code>VK_PIPELINE_STAGE_2_RESOLVE_BIT_KHR</code></p>
</li>
<li>
<p><code>VK_PIPELINE_STAGE_2_BLIT_BIT_KHR</code></p>
</li>
<li>
<p><code>VK_PIPELINE_STAGE_2_CLEAR_BIT_KHR</code></p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_splitting_up_vk_access_shader_read_bit">4.3. Splitting up VK_ACCESS_SHADER_READ_BIT</h3>
<div class="paragraph">
<p>The <code>VK_ACCESS_SHADER_READ_BIT</code> (now <code>VK_ACCESS_2_SHADER_READ_BIT_KHR</code>) was split into 3 new access flags which specify a dedicated access for the various case instead of having them combined into a single access flag.</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>VK_ACCESS_2_UNIFORM_READ_BIT_KHR</code></p>
</li>
<li>
<p><code>VK_ACCESS_2_SHADER_SAMPLED_READ_BIT_KHR</code></p>
</li>
<li>
<p><code>VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR</code></p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_combining_shader_stages_for_pre_rasterization">4.4. Combining shader stages for pre-rasterization</h3>
<div class="paragraph">
<p>Besides splitting up flags, the <code>VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT_KHR</code> was added to combine shader stages that occurs before rasterization in a single, convenient flag.</p>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_vk_access_shader_write_bit_alias">5. VK_ACCESS_SHADER_WRITE_BIT alias</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The <code>VK_ACCESS_SHADER_WRITE_BIT</code> (now <code>VK_ACCESS_2_SHADER_WRITE_BIT_KHR</code>) was given an alias of <code>VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR</code> to better describe the scope of what resources in the shader are described by the access flag.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_top_of_pipe_and_bottom_of_pipe_deprecation">6. TOP_OF_PIPE and BOTTOM_OF_PIPE deprecation</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The use of <code>VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT</code> and <code>VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT</code> are now deprecated and updating is simple as following the following 4 case with the new equivalents.</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT</code> in first synchronization scope</p>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// From
  .srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

// To
  .srcStageMask = VK_PIPELINE_STAGE_2_NONE_KHR;
  .srcAccessMask = VK_ACCESS_2_NONE_KHR;</code></pre>
</div>
</div>
</li>
<li>
<p><code>VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT</code> in second synchronization scope</p>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// From
  .dstStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

// To
  .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR;
  .dstAccessMask = VK_ACCESS_2_NONE_KHR;</code></pre>
</div>
</div>
</li>
<li>
<p><code>VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT</code> in first synchronization scope</p>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// From
  .srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

// To
  .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR;
  .srcAccessMask = VK_ACCESS_2_NONE_KHR;</code></pre>
</div>
</div>
</li>
<li>
<p><code>VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT</code> in second synchronization scope</p>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// From
  .dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

// To
  .dstStageMask = VK_PIPELINE_STAGE_2_NONE_KHR;
  .dstAccessMask = VK_ACCESS_2_NONE_KHR;</code></pre>
</div>
</div>
</li>
</ul>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_making_use_of_new_image_layouts">7. Making use of new image layouts</h2>
<div class="sectionbody">
<div class="paragraph">
<p><code>VK_KHR_synchronization2</code> adds 2 new image layouts <code>VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR</code> and <code>VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR</code> to help with making layout transition easier.</p>
</div>
<div class="paragraph">
<p>The following uses the example of doing a draw thats writes to both a color attachment and depth/stencil attachment which then are both sampled in the next draw. Prior a developer needed to make sure they matched up the layouts and access mask correctly such as the following:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkImageMemoryBarrier colorImageMemoryBarrier = {
  .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
  .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
  .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
  .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
};

VkImageMemoryBarrier depthStencilImageMemoryBarrier = {
  .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,,
  .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
  .oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
  .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
};</code></pre>
</div>
</div>
<div class="paragraph">
<p>but with <code>VK_KHR_synchronization2</code> this is made simple</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkImageMemoryBarrier colorImageMemoryBarrier = {
  .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT_KHR,
  .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT_KHR,
  .oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR, // new layout from VK_KHR_synchronization2
  .newLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR   // new layout from VK_KHR_synchronization2
};

VkImageMemoryBarrier depthStencilImageMemoryBarrier = {
  .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT_KHR,,
  .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT_KHR,
  .oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR, // new layout from VK_KHR_synchronization2
  .newLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR   // new layout from VK_KHR_synchronization2
};</code></pre>
</div>
</div>
<div class="paragraph">
<p>In the new case <code>VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR</code> works by contextually appling itself based on the image format used. So as long as <code>colorImageMemoryBarrier</code> is used on a color format, <code>VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR</code> maps to <code>VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL</code></p>
</div>
<div class="paragraph">
<p>Additionally, with <code>VK_KHR_synchronization2</code>, if <code>oldLayout</code> is equal to <code>newLayout</code>, no layout transition is performed and the image contents are preserved.  The layout used does not even need to match the layout of an image, so the following barrier is valid:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkImageMemoryBarrier depthStencilImageMemoryBarrier = {
  // other fields omitted
  .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  .newLayout = VK_IMAGE_LAYOUT_UNDEFINED,
};</code></pre>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_new_submission_flow">8. New submission flow</h2>
<div class="sectionbody">
<div class="paragraph">
<p><code>VK_KHR_synchronization2</code> adds the <code>vkQueueSubmit2KHR</code> command which main goal is to clean up the syntax for the function to wrap command buffers and semaphores in extensible structures, which incorporate changes from Vulkan 1.1, <code>VK_KHR_device_group</code>, and <code>VK_KHR_timeline_semaphore</code>.</p>
</div>
<div class="paragraph">
<p>Taking the following example of a normal queue submission call</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkSemaphore waitSemaphore;
VkSemaphore signalSemaphore;
VkCommandBuffer commandBuffers[8];

// Possible pNext from VK_KHR_timeline_semaphore
VkTimelineSemaphoreSubmitInfo timelineSemaphoreSubmitInfo = {
    // ...
    .pNext = nullptr
};

// Possible pNext from VK_KHR_device_group
VkDeviceGroupSubmitInfo deviceGroupSubmitInfo = {
    // ...
    .pNext = &amp;timelineSemaphoreSubmitInfo
};

// Possible pNext from Vulkan 1.1
VkProtectedSubmitInfo = protectedSubmitInfo {
    // ...
    .pNext = &amp;deviceGroupSubmitInfo
};

VkSubmitInfo submitInfo = {
    .pNext = &amp;protectedSubmitInfo, // Chains all 3 extensible structures
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &amp;waitSemaphore,
    .pWaitDstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
    .commandBufferCount = 8,
    .pCommandBuffers = commandBuffers,
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = signalSemaphore
};

vkQueueSubmit(queue, 1, submitInfo, fence);</code></pre>
</div>
</div>
<div class="paragraph">
<p>this can now be transformed to <code>vkQueueSubmit2KHR</code> as</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// Uses same semaphore and command buffer handles
VkSemaphore waitSemaphore;
VkSemaphore signalSemaphore;
VkCommandBuffer commandBuffers[8];

VkSemaphoreSubmitInfoKHR waitSemaphoreSubmitInfo = {
    .semaphore = waitSemaphore,
    .value = 1, // replaces VkTimelineSemaphoreSubmitInfo
    .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,
    .deviceIndex = 0, // replaces VkDeviceGroupSubmitInfo
};

// Note this is allowing a stage to set the signal operation
VkSemaphoreSubmitInfoKHR signalSemaphoreSubmitInfo = {
    .semaphore = waitSemaphore,
    .value = 2, // replaces VkTimelineSemaphoreSubmitInfo
    .stageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT_KHR, // when to signal semaphore
    .deviceIndex = 0, // replaces VkDeviceGroupSubmitInfo
};

// Need one for each VkCommandBuffer
VkCommandBufferSubmitInfoKHR = commandBufferSubmitInfos[8] {
    // ...
    {
        .commandBuffer = commandBuffers[i],
        .deviceMask = 0 // replaces VkDeviceGroupSubmitInfo
    },
};

VkSubmitInfo2KHR submitInfo = {
    .pNext = nullptr, // All 3 struct above are built into VkSubmitInfo2KHR
    .flags = VK_SUBMIT_PROTECTED_BIT_KHR, // also can be zero, replaces VkProtectedSubmitInfo
    .waitSemaphoreInfoCount = 1,
    .pWaitSemaphoreInfos = waitSemaphoreSubmitInfo,
    .commandBufferInfoCount = 8,
    .pCommandBufferInfos = commandBufferSubmitInfos,
    .signalSemaphoreInfoCount = 1,
    .pSignalSemaphoreInfos = signalSemaphoreSubmitInfo
}

vkQueueSubmit2KHR(queue, 1, submitInfo, fence);</code></pre>
</div>
</div>
<div class="paragraph">
<p>The difference between the two examples code snippets above is that the <code>vkQueueSubmit2KHR</code> will signal <code>VkSemaphore signalSemaphore</code> when the vertex shader stage is complete compared to the <code>vkQueueSubmit</code> call which will wait until the end of the submission.</p>
</div>
<div class="paragraph">
<p>To emulate the same behavior of semaphore signaling from <code>vkQueueSubmit</code> in <code>vkQueueSubmit2KHR</code> the <code>stageMask</code> can be set to <code>VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT</code></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// Waits until everything is done
VkSemaphoreSubmitInfoKHR signalSemaphoreSubmitInfo = {
    // ...
    .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    // ...
};</code></pre>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_emulation_layer">9. Emulation Layer</h2>
<div class="sectionbody">
<div class="paragraph">
<p>For devices that do not natively support this extension, there is a portable implementation in the <a href="https://github.com/KhronosGroup/Vulkan-ExtensionLayer">Vulkan-Extensionlayer</a> repository.   This layer should work with any Vulkan device. For more information see the <a href="https://github.com/KhronosGroup/Vulkan-ExtensionLayer/blob/master/docs/synchronization2_layer.md">layer documentation</a> and the <a href="https://github.com/KhronosGroup/Vulkan-ExtensionLayer/blob/bd8a72b14c67d011561cd795d777fb838c926e0f/tests/synchronization2_tests.cpp#L1243">Sync2Compat.Vulkan10</a> test case.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>The <code>VK_KHR_synchronization2</code> specification lists <code>VK_KHR_create_renderpass2</code> and <code>VK_KHR_get_phyiscal_device_properties2</code> as requirements. As a result, using synchronization2 without these extensions may result in validation errors. The extension requirements are being reevaluated and validation will be adjusted once this is complete.</p>
</div>
</td>
</tr>
</table>
</div>
</div>
</div>