<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#enabling-features">Enabling Features</a>
<ul class="sectlevel1">
<li><a href="#_category_of_features">1. Category of Features</a></li>
<li><a href="#_how_to_enable_the_features">2. How to Enable the Features</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink: /Notes/004-3d-rendering/vulkan/chapters/enabling_features.html
---</p>
</div>
<h1 id="enabling-features" class="sect0">Enabling Features</h1>
<div class="paragraph">
<p>This section goes over the logistics for enabling features.</p>
</div>
<div class="sect1">
<h2 id="_category_of_features">1. Category of Features</h2>
<div class="sectionbody">
<div class="paragraph">
<p>All features in Vulkan can be categorized/found in 3 sections</p>
</div>
<div class="olist arabic">
<ol class="arabic">
<li>
<p>Core 1.0 Features</p>
<div class="ulist">
<ul>
<li>
<p>These are the set of features that were available from the initial 1.0 release of Vulkan. The list of features can be found in <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VkPhysicalDeviceFeatures">VkPhysicalDeviceFeatures</a></p>
</li>
</ul>
</div>
</li>
<li>
<p>Future Core Version Features</p>
<div class="ulist">
<ul>
<li>
<p>With Vulkan 1.1+ some new features were added to the core version of Vulkan. To keep the size of <code>VkPhysicalDeviceFeatures</code> backward compatible, new structs were created to hold the grouping of features.</p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VkPhysicalDeviceVulkan11Features">VkPhysicalDeviceVulkan11Features</a></p>
</li>
<li>
<p><a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VkPhysicalDeviceVulkan12Features">VkPhysicalDeviceVulkan12Features</a></p>
</li>
</ul>
</div>
</li>
<li>
<p>Extension Features</p>
<div class="ulist">
<ul>
<li>
<p>Sometimes extensions contain features in order to enable certain aspects of the extension. These are easily found as they are all labeled as <code>VkPhysicalDevice<a id="ExtensionName"></a>Features</code></p>
</li>
</ul>
</div>
</li>
</ol>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_how_to_enable_the_features">2. How to Enable the Features</h2>
<div class="sectionbody">
<div class="paragraph">
<p>All features must be enabled at <code>VkDevice</code> creation time inside the <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VkDeviceCreateInfo">VkDeviceCreateInfo</a> struct.</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Don&#8217;t forget to query first with <code>vkGetPhysicalDeviceFeatures</code> or <code>vkGetPhysicalDeviceFeatures2</code></p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>For the Core 1.0 Features, this is as simple as setting <code>VkDeviceCreateInfo::pEnabledFeatures</code> with the features desired to be turned on.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkPhysicalDeviceFeatures features = {};
vkGetPhysicalDeviceFeatures(physical_device, &amp;features);

// Logic if feature is not supported
if (features.robustBufferAccess == VK_FALSE) {
}

VkDeviceCreateInfo info = {};
info.pEnabledFeatures = &amp;features;</code></pre>
</div>
</div>
<div class="paragraph">
<p>For <strong>all features</strong>, including the Core 1.0 Features, use <code>VkPhysicalDeviceFeatures2</code> to pass into <code>VkDeviceCreateInfo.pNext</code></p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkPhysicalDeviceShaderDrawParametersFeatures ext_feature = {};

VkPhysicalDeviceFeatures2 physical_features2 = {};
physical_features2.pNext = &amp;ext_feature;

vkGetPhysicalDeviceFeatures2(physical_device, &amp;physical_features2);

// Logic if feature is not supported
if (ext_feature.shaderDrawParameters == VK_FALSE) {
}

VkDeviceCreateInfo info = {};
info.pNext = &amp;physical_features2;</code></pre>
</div>
</div>
<div class="paragraph">
<p>The same works for the &#8220;Future Core Version Features&#8221; too.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkPhysicalDeviceVulkan11Features features11 = {};

VkPhysicalDeviceFeatures2 physical_features2 = {};
physical_features2.pNext = &amp;features11;

vkGetPhysicalDeviceFeatures2(physical_device, &amp;physical_features2);

// Logic if feature is not supported
if (features11.shaderDrawParameters == VK_FALSE) {
}

VkDeviceCreateInfo info = {};
info.pNext = &amp;physical_features2;</code></pre>
</div>
</div>
</div>
</div>