<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#querying-extensions-features">Querying Properties, Extensions, Features, Limits, and Formats</a>
<ul class="sectlevel1">
<li><a href="#_properties">1. Properties</a></li>
<li><a href="#_extensions">2. Extensions</a></li>
<li><a href="#_features">3. Features</a></li>
<li><a href="#_limits">4. Limits</a></li>
<li><a href="#_formats">5. Formats</a></li>
<li><a href="#_tools">6. Tools</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/querying_extensions_features.html
layout: default
---</p>
</div>
<h1 id="querying-extensions-features" class="sect0">Querying Properties, Extensions, Features, Limits, and Formats</h1>
<div class="paragraph">
<p>One of Vulkan&#8217;s main features is that is can be used to develop on multiple platforms and devices. To make this possible, an application is responsible for querying the information from each physical device and then basing decisions on this information.</p>
</div>
<div class="paragraph">
<p>The items that can be queried from a physical device</p>
</div>
<div class="ulist">
<ul>
<li>
<p>Properties</p>
</li>
<li>
<p>Features</p>
</li>
<li>
<p>Extensions</p>
</li>
<li>
<p>Limits</p>
</li>
<li>
<p>Formats</p>
</li>
</ul>
</div>
<div class="sect1">
<h2 id="_properties">1. Properties</h2>
<div class="sectionbody">
<div class="paragraph">
<p>There are many other components in Vulkan that are labeled as properties. The term &#8220;properties&#8221; is an umbrella term for any read-only data that can be queried.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_extensions">2. Extensions</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Check out the <a href="enabling_extensions.html#enabling-extensions">Enabling Extensions</a> chapter for more information.</p>
</div>
<div class="paragraph">
<p>There is a <a href="https://www.khronos.org/registry/vulkan/#repo-docs">Registry</a> with all available extensions.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>There are many times when a set of new functionality is desired in Vulkan that doesn&#8217;t currently exist. Extensions have the ability to add new functionality. Extensions may define new Vulkan functions, enums, structs, or feature bits. While all of these extended items are found by default in the Vulkan Headers, it is <strong>undefined behavior</strong> to use extended Vulkan if the <a href="enabling_extensions.html#enabling-extensions">extensions are not enabled</a>.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_features">3. Features</h2>
<div class="sectionbody">
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Checkout the <a href="enabling_features.html#enabling-features">Enabling Features</a> chapter for more information.</p>
</div>
</td>
</tr>
</table>
</div>
<div class="paragraph">
<p>Features describe functionality which is not supported on all implementations. Features can be <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#vkGetPhysicalDeviceFeatures">queried</a> and then enabled when creating the <code>VkDevice</code>. Besides the <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#features">list of all features</a>, some <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#features-requirements">features are mandatory</a> due to newer Vulkan versions or use of extensions.</p>
</div>
<div class="paragraph">
<p>A common technique is for an extension to expose a new struct that can be passed through <code>pNext</code> that adds more features to be queried.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_limits">4. Limits</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Limits are implementation-dependent minimums, maximums, and other device characteristics that an application may need to be aware of. Besides the <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#limits">list of all limits</a>, some limits also have <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#limits-minmax">minimum/maximum required values</a> guaranteed from a Vulkan implementation.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_formats">5. Formats</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Vulkan provides many <code>VkFormat</code> that have multiple <code>VkFormatFeatureFlags</code> each holding a various <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkFormatFeatureFlagBits.html">VkFormatFeatureFlagBits</a> bitmasks that can be queried.</p>
</div>
<div class="paragraph">
<p>Checkout the <a href="formats.html#feature-support">Format chapter</a> for more information.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_tools">6. Tools</h2>
<div class="sectionbody">
<div class="paragraph">
<p>There are a few tools to help with getting all the information in a quick and in a human readable format.</p>
</div>
<div class="paragraph">
<p><code>vulkaninfo</code> is a command line utility for Windows, Linux, and macOS that enables you to see all the available items listed above about your GPU. Refer to the <a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/vulkaninfo.html">Vulkaninfo documentation</a> in the Vulkan SDK.</p>
</div>
<div class="paragraph">
<p>The <a href="https://play.google.com/store/apps/details?id=de.saschawillems.vulkancapsviewer&amp;hl=en_US">Vulkan Hardware Capability Viewer</a> app developed by Sascha Willems, is an Android app to display all details for devices that support Vulkan.</p>
</div>
</div>
</div>