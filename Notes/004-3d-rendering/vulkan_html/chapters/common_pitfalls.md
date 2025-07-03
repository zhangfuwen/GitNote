<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#common-pitfalls">Common Pitfalls for New Vulkan Developers</a>
<ul class="sectlevel1">
<li><a href="#_validation_layers">1. Validation Layers</a></li>
<li><a href="#_vulkan_is_a_box_of_tools">2. Vulkan Is a Box of Tools</a></li>
<li><a href="#_recording_command_buffers">3. Recording Command Buffers</a></li>
<li><a href="#_multiple_pipelines">4. Multiple Pipelines</a></li>
<li><a href="#_resource_duplication_per_swapchain_image">5. Resource Duplication per Swapchain Image</a></li>
<li><a href="#_multiple_queues_per_queue_family">6. Multiple Queues per Queue Family</a></li>
<li><a href="#_descriptor_sets">7. Descriptor Sets</a></li>
<li><a href="#_correct_api_usage_practices">8. Correct API usage practices</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink: /Notes/004-3d-rendering/vulkan/chapters/common_pitfalls.html
---</p>
</div>
<h1 id="common-pitfalls" class="sect0">Common Pitfalls for New Vulkan Developers</h1>
<div class="paragraph">
<p>This is a short list of assumptions, traps, and anti-patterns in the Vulkan API. It is not a list of &#8220;best practices&#8221;, rather it covers the common mistakes that developers new to Vulkan could easily make.</p>
</div>
<div class="sect1">
<h2 id="_validation_layers">1. Validation Layers</h2>
<div class="sectionbody">
<div class="paragraph">
<p>During development, ensure that the Validation Layers are enabled. They are an invaluable tool for catching mistakes while using the Vulkan API. Parameter checking, object lifetimes, and threading violations all are part of the provided error checks. A way to reassure that they are enabled is to verify if the text &#8220;Debug Messenger Added&#8221; is in the output stream. More info can be found in the <a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/layer_configuration.html">Vulkan SDK</a> layer documentation.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_vulkan_is_a_box_of_tools">2. Vulkan Is a Box of Tools</h2>
<div class="sectionbody">
<div class="paragraph">
<p>In Vulkan, most problems can be tackled with multiple methods, each with their own benefits and drawbacks. There is rarely a &#8220;perfect&#8221; solution and obsessing over finding one is often a fruitless effort. When faced with a problem, try to create an adequate solution that meets the current needs and isn&#8217;t overly convoluted. While the specification for Vulkan can be useful, it isn&#8217;t the best source for how to use Vulkan in practice. Instead, reference external sources, like this guide, hardware best practice guides, tutorials, and other articles for more in-depth information. Finally, profiling various solutions is an important part of discovering which solution to use.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_recording_command_buffers">3. Recording Command Buffers</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Many early Vulkan tutorials and documents recommended writing a command buffer once and re-using it wherever possible. In practice however re-use rarely has the advertized performance benefit while incurring a non-trivial development burden due to the complexity of implementation. While it may appear counterintuitive, as re-using computed data is a common optimization, managing a scene with objects being added and removed as well as techniques such as frustum culling which vary the draw calls issued on a per frame basis make reusing command buffers a serious design challenge. It requires a caching scheme to manage command buffers and maintaining state for determining if and when re-recording becomes necessary. Instead, prefer to re-record fresh command buffers every frame. If performance is a problem, recording can be multithreaded as well as using secondary command buffers for non-variable draw calls, like post processing.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_multiple_pipelines">4. Multiple Pipelines</h2>
<div class="sectionbody">
<div class="paragraph">
<p>A graphics <code>VkPipeline</code> contains the combination of state needed to perform a draw call. Rendering a scene with different shaders, blending modes, vertex layouts, etc, will require a pipeline for each possibility. Because pipeline creation and swapping them between draw calls have an associated cost, it is a good practice to create and swap pipelines only as needed. However, by using various techniques and features to further reduce creation and swapping beyond the simple cases can be counterproductive, as it adds complexity with no guarantee of benefit. For large engines this may be necessary, but otherwise it is unlikely to be a bottleneck. Using the pipeline cache can further reduce the costs without resorting to more complex schemes.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_resource_duplication_per_swapchain_image">5. Resource Duplication per Swapchain Image</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Pipelining frames is a common way to improve performance. By having multiple frames rendering at the same time, each using their own copy of the required resources, it reduces latency by removing resource contention. A simple implementation of this will duplicate the resources needed by each image in the swapchain. The issue is that this leads to assuming rendering resources must be duplicated once for each swapchain image. While practical for some resources, like the command buffers and semaphores used for each frame, the one-to-one duplication with swapchain images isn&#8217;t often necessary. Vulkan offers a large amount of flexibility, letting the developer choose what level of duplication is right for their situation. Many resources may only need two copies, for example, uniform buffers or data which is updated once per frame, and others may not need any duplication at all.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_multiple_queues_per_queue_family">6. Multiple Queues per Queue Family</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Several hardware platforms have more than one <code>VkQueue</code> per queue family. This can be useful by being able to submit work to the same queue family from separate queues. While there can be advantages, it isn&#8217;t necessarily better to create or use the extra queues. For specific performance recommendations, refer to hardware vendors' best practices guides.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_descriptor_sets">7. Descriptor Sets</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Descriptor Sets are designed to facilitate grouping data used in shaders by usage and update frequency. The Vulkan Spec mandates that hardware supports using at least 4 Descriptor Sets at a time, with most hardware supporting at least 8. Therefore there is very little reason not to use more than one where it is sensible.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_correct_api_usage_practices">8. Correct API usage practices</h2>
<div class="sectionbody">
<div class="paragraph">
<p>While the Validation Layers can catch many types of errors, they are not perfect. Below is a short list of good habits and possible sources of error when encountering odd behavior.</p>
</div>
<div class="ulist">
<ul>
<li>
<p>Initialize all variables and structs.</p>
</li>
<li>
<p>Use the correct <code>sType</code> for each structure.</p>
</li>
<li>
<p>Verify correct <code>pNext</code> chain usage, nulling it out when not needed.</p>
</li>
<li>
<p>There are no default values in Vulkan.</p>
</li>
<li>
<p>Use correct enum, <code>VkFlag</code>, and bitmask values.</p>
</li>
<li>
<p>Consider using a type-safe Vulkan wrapper, eg. <a href="https://github.com/KhronosGroup/Vulkan-Hpp">Vulkan.hpp</a> for C++</p>
</li>
<li>
<p>Check function return values, eg <code>VkResult</code>.</p>
</li>
<li>
<p>Call cleanup functions where appropriate.</p>
</li>
</ul>
</div>
</div>
</div>