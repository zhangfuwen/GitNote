<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#protected">Protected Memory</a>
<ul class="sectlevel1">
<li><a href="#_checking_for_support">1. Checking for support</a></li>
<li><a href="#_protected_queues">2. Protected queues</a></li>
<li><a href="#_protected_resources">3. Protected resources</a></li>
<li><a href="#_protected_swapchain">4. Protected swapchain</a></li>
<li><a href="#_protected_command_buffer">5. Protected command buffer</a></li>
<li><a href="#_submitting_protected_work">6. Submitting protected work</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/protected.html
layout: default
---</p>
</div>
<h1 id="protected" class="sect0">Protected Memory</h1>
<div class="paragraph">
<p>Protected memory divides device memory into &#8220;protected device memory&#8221; and &#8220;unprotected device memory&#8221;.</p>
</div>
<div class="paragraph">
<p>In general, most OS don&#8217;t allow one application to access another application&#8217;s GPU memory unless explicitly shared (e.g. via <a href="extensions/external.html#external-memory">external memory</a>). A common example of protected memory is for containing DRM content, which a process might be allowed to modify (e.g. for image filtering, or compositing playback controls and closed captions) but shouldn' be able to extract into unprotected memory. The data comes in encrypted and remains encrypted until it reaches the pixels on the display.</p>
</div>
<div class="paragraph">
<p>The Vulkan Spec <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#memory-protected-memory">explains in detail</a> what &#8220;protected device memory&#8221; enforces. The following is a breakdown of what is required in order to properly enable a protected submission using protected memory.</p>
</div>
<div class="sect1">
<h2 id="_checking_for_support">1. Checking for support</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Protected memory was added in Vulkan 1.1 and there was no extension prior. This means any Vulkan 1.0 device will not be capable of supporting protected memory. To check for support, an application must <a href="enabling_features.html#enabling-features">query and enable</a> the <code>VkPhysicalDeviceProtectedMemoryFeatures::protectedMemory</code> field.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_protected_queues">2. Protected queues</h2>
<div class="sectionbody">
<div class="paragraph">
<p>A protected queue can read both protected and unprotected memory, but can only write to protected memory. If a queue can write to unprotected memory, then it can&#8217;t also read from protected memory.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Often performance counters and other timing measurement systems are disabled or less accurate for protected queues to prevent side-channel attacks.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>Using <code>vkGetPhysicalDeviceQueueFamilyProperties</code> to get the <code>VkQueueFlags</code> of each queue, an application can find a queue family with <code>VK_QUEUE_PROTECTED_BIT</code> flag exposed. This does <strong>not</strong> mean the queues from the family are always protected, but rather the queues <strong>can be</strong> a protected queue.</p>
</div>
<div class="paragraph">
<p>To tell the driver to make the <code>VkQueue</code> protected, the <code>VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT</code> is needed in <code>VkDeviceQueueCreateInfo</code> during <code>vkCreateDevice</code>.</p>
</div>
<div class="paragraph">
<p>The following pseudo code is how an application could request for 2 protected <code>VkQueue</code> objects to be created from the same queue family:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkDeviceQueueCreateInfo queueCreateInfo[1];
queueCreateInfo[0].flags             = VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT;
queueCreateInfo[0].queueFamilyIndex  = queueFamilyFound;
queueCreateInfo[0].queueCount        = 2; // assuming 2 queues are in the queue family

VkDeviceCreateInfo deviceCreateInfo   = {};
deviceCreateInfo.pQueueCreateInfos    = queueCreateInfo;
deviceCreateInfo.queueCreateInfoCount = 1;
vkCreateDevice(physicalDevice, &amp;deviceCreateInfo, nullptr, &amp;deviceHandle);</code></pre>
</div>
</div>
<div class="paragraph">
<p>It is also possible to split the queues in a queue family so some are protected and some are not. The following pseudo code is how an application could request for 1 protected <code>VkQueue</code> and 1 unprotected <code>VkQueue</code> objects to be created from the same queue family:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkDeviceQueueCreateInfo queueCreateInfo[2];
queueCreateInfo[0].flags             = VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT;
queueCreateInfo[0].queueFamilyIndex  = queueFamilyFound;
queueCreateInfo[0].queueCount        = 1;

queueCreateInfo[1].flags             = 0; // unprotected because the protected flag is not set
queueCreateInfo[1].queueFamilyIndex  = queueFamilyFound;
queueCreateInfo[1].queueCount        = 1;

VkDeviceCreateInfo deviceCreateInfo   = {};
deviceCreateInfo.pQueueCreateInfos    = queueCreateInfo;
deviceCreateInfo.queueCreateInfoCount = 2;
vkCreateDevice(physicalDevice, &amp;deviceCreateInfo, nullptr, &amp;deviceHandle);</code></pre>
</div>
</div>
<div class="paragraph">
<p>Now instead of using <code>vkGetDeviceQueue</code> an application has to use <code>vkGetDeviceQueue2</code> in order to pass the <code>VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT</code> flag when getting the <code>VkQueue</code> handle.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkDeviceQueueInfo2 info = {};
info.queueFamilyIndex = queueFamilyFound;
info.queueIndex       = 0;
info.flags            = VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT;
vkGetDeviceQueue2(deviceHandle, &amp;info, &amp;protectedQueue);</code></pre>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_protected_resources">3. Protected resources</h2>
<div class="sectionbody">
<div class="paragraph">
<p>When creating a <code>VkImage</code> or <code>VkBuffer</code> to make them protected is as simple as setting <code>VK_IMAGE_CREATE_PROTECTED_BIT</code> and <code>VK_BUFFER_CREATE_PROTECTED_BIT</code> respectively.</p>
</div>
<div class="paragraph">
<p>When binding memory to the protected resource, the <code>VkDeviceMemory</code> must have been allocated from a <code>VkMemoryType</code> with the <code>VK_MEMORY_PROPERTY_PROTECTED_BIT</code> bit.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_protected_swapchain">4. Protected swapchain</h2>
<div class="sectionbody">
<div class="paragraph">
<p>When creating a swapchain the <code>VK_SWAPCHAIN_CREATE_PROTECTED_BIT_KHR</code> bit is used to make a protected swapchain.</p>
</div>
<div class="paragraph">
<p>All <code>VkImage</code> from <code>vkGetSwapchainImagesKHR</code> using a protected swapchain are the same as if the image was created with <code>VK_IMAGE_CREATE_PROTECTED_BIT</code>.</p>
</div>
<div class="paragraph">
<p>Sometimes it is unknown whether swapchains can be created with the <code>VK_SWAPCHAIN_CREATE_PROTECTED_BIT_KHR</code> flag set. The <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_surface_protected_capabilities.html">VK_KHR_surface_protected_capabilities</a> extension is exposed on platforms where this might be unknown.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_protected_command_buffer">5. Protected command buffer</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Using the protected <code>VkQueue</code>, an application can also use <code>VK_COMMAND_POOL_CREATE_PROTECTED_BIT</code> when creating a <code>VkCommandPool</code></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkCommandPoolCreateInfo info = {};
info.flags            = VK_COMMAND_POOL_CREATE_PROTECTED_BIT;
info.queueFamilyIndex = queueFamilyFound; // protected queue
vkCreateCommandPool(deviceHandle, &amp;info, nullptr, &amp;protectedCommandPool));</code></pre>
</div>
</div>
<div class="paragraph">
<p>All command buffers allocated from the protected command pool become &#8220;protected command buffers&#8221;</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkCommandBufferAllocateInfo info = {};
info.commandPool = protectedCommandPool;
vkAllocateCommandBuffers(deviceHandle, &amp;info, &amp;protectedCommandBuffers);</code></pre>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_submitting_protected_work">6. Submitting protected work</h2>
<div class="sectionbody">
<div class="paragraph">
<p>When submitting work to be protected, all the <code>VkCommandBuffer</code> submitted must also be protected.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkProtectedSubmitInfo protectedSubmitInfo = {};
protectedSubmitInfo.protectedSubmit       = true;

VkSubmitInfo submitInfo                  = {};
submitInfo.pNext                         = &amp;protectedSubmitInfo;
submitInfo.pCommandBuffers               = protectedCommandBuffers;

vkQueueSubmit(protectedQueue, 1, &amp;submitInfo, fence));</code></pre>
</div>
</div>
<div class="paragraph">
<p>or using <a href="extensions/VK_KHR_synchronization2.html#VK_KHR_synchronization2">VK_KHR_synchronization2</a></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkSubmitInfo2KHR submitInfo = {}
submitInfo.flags = VK_SUBMIT_PROTECTED_BIT_KHR;

vkQueueSubmit2KHR(protectedQueue, 1, submitInfo, fence);</code></pre>
</div>
</div>
</div>
</div>