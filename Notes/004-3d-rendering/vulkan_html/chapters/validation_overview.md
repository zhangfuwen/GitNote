<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#validation-overview">Vulkan Validation Overview</a>
<ul class="sectlevel1">
<li><a href="#_valid_usage_vu">1. Valid Usage (VU)</a></li>
<li><a href="#_undefined_behavior">2. Undefined Behavior</a></li>
<li><a href="#_valid_usage_id_vuid">3. Valid Usage ID (VUID)</a></li>
<li><a href="#khronos-validation-layer">4. Khronos Validation Layer</a>
<ul class="sectlevel2">
<li><a href="#_getting_validation_layers">4.1. Getting Validation Layers</a></li>
</ul>
</li>
<li><a href="#_breaking_down_a_validation_error_message">5. Breaking Down a Validation Error Message</a>
<ul class="sectlevel2">
<li><a href="#_example_1_implicit_valid_usage">5.1. Example 1 - Implicit Valid Usage</a></li>
<li><a href="#_example_2_explicit_valid_usage">5.2. Example 2 - Explicit Valid Usage</a></li>
</ul>
</li>
<li><a href="#_multiple_vuids">6. Multiple VUIDs</a></li>
<li><a href="#_special_usage_tags">7. Special Usage Tags</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/validation_overview.html
layout: default
---</p>
</div>
<h1 id="validation-overview" class="sect0">Vulkan Validation Overview</h1>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>The purpose of this section is to give a full overview of how Vulkan deals with <em>valid usage</em> of the API.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="sect1">
<h2 id="_valid_usage_vu">1. Valid Usage (VU)</h2>
<div class="sectionbody">
<div class="paragraph">
<p>A <strong>VU</strong> is explicitly <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#fundamentals-validusage">defined in the Vulkan Spec</a> as:</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>set of conditions that <strong>must</strong> be met in order to achieve well-defined run-time behavior in an application.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>One of the main advantages of Vulkan, as an explicit API, is that the implementation (driver) doesn&#8217;t waste time checking for valid input. In OpenGL, the implementation would have to always check for valid usage which added noticeable overhead. There is no <a href="https://www.khronos.org/opengl/wiki/OpenGL_Error">glGetError</a> equivalent in Vulkan.</p>
</div>
<div class="paragraph">
<p>The valid usages will be listed in the spec after every function and structure. For example, if a VUID checks for an invalid <code>VkImage</code> at <code>VkBindImageMemory</code> then the valid usage in the spec is found under <code>VkBindImageMemory</code>. This is because the Validation Layers will only know about all the information at <code>VkBindImageMemory</code> during the execution of the application.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_undefined_behavior">2. Undefined Behavior</h2>
<div class="sectionbody">
<div class="paragraph">
<p>When an application supplies invalid input, according to the valid usages in the spec, the result is <em>undefined behavior</em>. In this state, Vulkan makes no guarantees as <a href="https://raphlinus.github.io/programming/rust/2018/08/17/undefined-behavior.html">anything is possible with undefined behavior</a>.</p>
</div>
<div class="paragraph">
<p><strong>VERY IMPORTANT</strong>: While undefined behavior might seem to work on one implementation, there is a good chance it will fail on another.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_valid_usage_id_vuid">3. Valid Usage ID (VUID)</h2>
<div class="sectionbody">
<div class="paragraph">
<p>A <code>VUID</code> is an unique ID given to each valid usage. This allows a way to point to a valid usage in the spec easily.</p>
</div>
<div class="paragraph">
<p>Using <code>VUID-vkBindImageMemory-memoryOffset-01046</code> as an example, it is as simple as adding the VUID to an anchor in the HMTL version of the spec (<a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBindImageMemory-memoryOffset-01046">vkspec.html#VUID-vkBindImageMemory-memoryOffset-01046</a>) and it will jump right to the VUID.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="khronos-validation-layer">4. Khronos Validation Layer</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Since Vulkan doesn&#8217;t do any error checking, it is <strong>very important</strong>, when developing, to enable the <a href="https://github.com/KhronosGroup/Vulkan-ValidationLayers">Validation Layers</a> right away to help catch invalid behavior. Applications should also never ship the Validation Layers with their application as they noticeably reduce performance and are designed for the development phase.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>The Khronos Validation Layer used to consist of multiple layers but now has been unified to a single <code>VK_LAYER_KHRONOS_validition</code> layer. <a href="https://www.lunarg.com/wp-content/uploads/2019/04/UberLayer_V3.pdf">More details explained in LunarG&#8217;s whitepaper</a>.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="sect2">
<h3 id="_getting_validation_layers">4.1. Getting Validation Layers</h3>
<div class="paragraph">
<p>The Validation Layers are constantly being updated and improved so it is always possible to grab the source and <a href="https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/master/BUILD.md">build it yourself</a>. In case you want a prebuit version there are various options for all supported platforms:</p>
</div>
<div class="ulist">
<ul>
<li>
<p><strong>Android</strong> - Binaries are <a href="https://github.com/KhronosGroup/Vulkan-ValidationLayers/releases">released on GitHub</a> with most up to date version. The NDK will also comes with the Validation Layers built and <a href="https://developer.android.com/ndk/guides/graphics/validation-layer">information on how to use them</a>.</p>
</li>
<li>
<p><strong>Linux</strong> - The <a href="https://vulkan.lunarg.com/sdk/home">Vulkan SDK</a> comes with the Validation Layers built and instructions on how to use them on <a href="https://vulkan.lunarg.com/doc/sdk/latest/linux/validation_layers.html">Linux</a>.</p>
</li>
<li>
<p><strong>MacOS</strong> - The <a href="https://vulkan.lunarg.com/sdk/home">Vulkan SDK</a> comes with the Validation Layers built and instructions on how to use them on <a href="https://vulkan.lunarg.com/doc/sdk/latest/mac/validation_layers.html">MacOS</a>.</p>
</li>
<li>
<p><strong>Windows</strong> - The <a href="https://vulkan.lunarg.com/sdk/home">Vulkan SDK</a> comes with the Validation Layers built and instructions on how to use them on <a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/validation_layers.html">Windows</a>.</p>
</li>
</ul>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_breaking_down_a_validation_error_message">5. Breaking Down a Validation Error Message</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The Validation Layers attempt to supply as much useful information as possible when an error occurs. The following examples are to help show how to get the most information out of the Validation Layers</p>
</div>
<div class="sect2">
<h3 id="_example_1_implicit_valid_usage">5.1. Example 1 - Implicit Valid Usage</h3>
<div class="paragraph">
<p>This example shows a case where an <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#fundamentals-implicit-validity">implicit VU</a> is triggered. There will not be a number at the end of the VUID.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code>Validation Error: [ VUID-vkBindBufferMemory-memory-parameter ] Object 0: handle =
0x20c8650, type = VK_OBJECT_TYPE_INSTANCE; | MessageID = 0xe9199965 | Invalid
VkDeviceMemory Object 0x60000000006. The Vulkan spec states: memory must be a valid
VkDeviceMemory handle (https://www.khronos.org/registry/vulkan/specs/1.1-extensions/
html/vkspec.html#VUID-vkBindBufferMemory-memory-parameter)</code></pre>
</div>
</div>
<div class="ulist">
<ul>
<li>
<p>The first thing to notice is the VUID is listed first in the message (<code>VUID-vkBindBufferMemory-memory-parameter</code>)</p>
<div class="ulist">
<ul>
<li>
<p>There is also a link at the end of the message to the VUID in the spec</p>
</li>
</ul>
</div>
</li>
<li>
<p><code>The Vulkan spec states:</code> is the quoted VUID from the spec.</p>
</li>
<li>
<p>The <code>VK_OBJECT_TYPE_INSTANCE</code> is the <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#_debugging">VkObjectType</a></p>
</li>
<li>
<p><code>Invalid VkDeviceMemory Object 0x60000000006</code> is the <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#fundamentals-objectmodel-overview">Dispatchable Handle</a> to help show which <code>VkDeviceMemory</code> handle was the cause of the error.</p>
</li>
</ul>
</div>
</div>
<div class="sect2">
<h3 id="_example_2_explicit_valid_usage">5.2. Example 2 - Explicit Valid Usage</h3>
<div class="paragraph">
<p>This example shows an error where some <code>VkImage</code> is trying to be bound to 2 different <code>VkDeviceMemory</code> objects</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code>Validation Error: [ VUID-vkBindImageMemory-image-01044 ] Object 0: handle =
0x90000000009, name = myTextureMemory, type = VK_OBJECT_TYPE_DEVICE_MEMORY; Object 1:
handle = 0x70000000007, type = VK_OBJECT_TYPE_IMAGE; Object 2: handle = 0x90000000006,
name = myIconMemory, type = VK_OBJECT_TYPE_DEVICE_MEMORY; | MessageID = 0x6f3eac96 |
In vkBindImageMemory(), attempting to bind VkDeviceMemory 0x90000000009[myTextureMemory]
to VkImage 0x70000000007[] which has already been bound to VkDeviceMemory
0x90000000006[myIconMemory]. The Vulkan spec states: image must not already be
backed by a memory object (https://www.khronos.org/registry/vulkan/specs/1.1-extensions/
html/vkspec.html#VUID-vkBindImageMemory-image-01044)</code></pre>
</div>
</div>
<div class="ulist">
<ul>
<li>
<p>Example 2 is about the same as Example 1 with the exception that the <code>name</code> that was attached to the object (<code>name = myTextureMemory</code>). This was done using the <a href="https://www.lunarg.com/new-tutorial-for-vulkan-debug-utilities-extension/">VK_EXT_debug_util</a> extension (<a href="https://github.com/KhronosGroup/Vulkan-Samples/tree/master/samples/extensions/debug_utils">Sample of how to use the extension</a>). Note that the old way of using <a href="https://www.saschawillems.de/blog/2016/05/28/tutorial-on-using-vulkans-vk_ext_debug_marker-with-renderdoc/">VK_EXT_debug_report</a> might be needed on legacy devices that don&#8217;t support <code>VK_EXT_debug_util</code>.</p>
</li>
<li>
<p>There were 3 objects involved in causing this error.</p>
<div class="ulist">
<ul>
<li>
<p>Object 0 is a <code>VkDeviceMemory</code> named <code>myTextureMemory</code></p>
</li>
<li>
<p>Object 1 is a <code>VkImage</code> with no name</p>
</li>
<li>
<p>Object 2 is a <code>VkDeviceMemory</code> named <code>myIconMemory</code></p>
</li>
</ul>
</div>
</li>
<li>
<p>With the names it is easy to see &#8220;In <code>vkBindImageMemory()</code>, the <code>myTextureMemory</code> memory was attempting to bind to an image already been bound to the <code>myIconMemory</code> memory&#8221;.</p>
</li>
</ul>
</div>
<div class="paragraph">
<p>Each error message contains a uniform logging pattern. This allows information to be easily found in any error. The pattern is as followed:</p>
</div>
<div class="ulist">
<ul>
<li>
<p>Log status (ex. <code>Error:</code>, <code>Warning:</code>, etc)</p>
</li>
<li>
<p>The VUID</p>
</li>
<li>
<p>Array of objects involved</p>
<div class="ulist">
<ul>
<li>
<p>Index of array</p>
</li>
<li>
<p>Dispatch Handle value</p>
</li>
<li>
<p>Optional name</p>
</li>
<li>
<p>Object Type</p>
</li>
</ul>
</div>
</li>
<li>
<p>Function or struct error occurred in</p>
</li>
<li>
<p>Message the layer has created to help describe the issue</p>
</li>
<li>
<p>The full Valid Usage from the spec</p>
</li>
<li>
<p>Link to the Valid Usage</p>
</li>
</ul>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_multiple_vuids">6. Multiple VUIDs</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>The following is not ideal and is being looked into how to make it simpler</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>Currently, the spec is designed to only show the VUIDs depending on the <a href="vulkan_spec.html#vulkan-spec-variations">version and extensions the spec was built with</a>. Simply put, additions of extensions and versions may alter the VU language enough (from new API items added) that a separate VUID is created.</p>
</div>
<div class="paragraph">
<p>An example of this from the <a href="https://github.com/KhronosGroup/Vulkan-Docs">Vulkan-Docs</a> where the <a href="vulkan_spec.html#vulkan-spec">spec in generated from</a></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-c" data-lang="c">  * [[VUID-VkPipelineLayoutCreateInfo-pSetLayouts-00287]]
    ...</code></pre>
</div>
</div>
<div class="paragraph">
<p>What this creates is two very similar VUIDs</p>
</div>
<div class="paragraph">
<p>In this example, both VUIDs are very similar and the only difference is the fact <code>VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT</code> is referenced in one and not this other. This is because the enum was added with the addition of <code>VK_EXT_descriptor_indexing</code> which is now part of Vulkan 1.2.</p>
</div>
<div class="paragraph">
<p>This means the 2 valid <a href="vulkan_spec.html#html-full">html links to the spec</a> would look like</p>
</div>
<div class="ulist">
<ul>
<li>
<p><code>1.1/html/vkspec.html#VUID-VkPipelineLayoutCreateInfo-pSetLayouts-00287</code></p>
</li>
<li>
<p><code>1.2/html/vkspec.html#VUID-VkPipelineLayoutCreateInfo-descriptorType-03016</code></p>
</li>
</ul>
</div>
<div class="paragraph">
<p>The Validation Layer uses the device properties of the application in order to decide which one to display. So in this case, if you are running on a Vulkan 1.2 implementation or a device that supports <code>VK_EXT_descriptor_indexing</code> it will display the VUID <code>03016</code>.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_special_usage_tags">7. Special Usage Tags</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The <a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/best_practices.html">Best Practices layer</a> will produce warnings when an application tries to use any extension with <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#extendingvulkan-compatibility-specialuse">special usage tags</a>. An example of such an extension is <a href="extensions/translation_layer_extensions.html#vk_ext_transform_feedback">VK_EXT_transform_feedback</a> which is only designed for emulation layers. If an application&#8217;s intended usage corresponds to one of the special use cases, the following approach will allow you to ignore the warnings.</p>
</div>
<div class="paragraph">
<p>Ignoring Special Usage Warnings with <code>VK_EXT_debug_report</code></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkBool32 DebugReportCallbackEXT(/* ... */ const char* pMessage /* ... */)
{
    // If pMessage contains "specialuse-extension", then exit
    if(strstr(pMessage, "specialuse-extension") != NULL) {
        return VK_FALSE;
    };

    // Handle remaining validation messages
}</code></pre>
</div>
</div>
<div class="paragraph">
<p>Ignoring Special Usage Warnings with <code>VK_EXT_debug_utils</code></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkBool32 DebugUtilsMessengerCallbackEXT(/* ... */ const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData /* ... */)
{
    // If pMessageIdName contains "specialuse-extension", then exit
    if(strstr(pCallbackData-&gt;pMessageIdName, "specialuse-extension") != NULL) {
        return VK_FALSE;
    };

    // Handle remaining validation messages
}</code></pre>
</div>
</div>
</div>
</div>