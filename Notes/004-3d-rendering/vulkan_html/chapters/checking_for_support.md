<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#checking-for-support">Checking For Vulkan Support</a>
<ul class="sectlevel1">
<li><a href="#_platform_support">1. Platform Support</a>
<ul class="sectlevel2">
<li><a href="#_android">1.1. Android</a></li>
<li><a href="#_bsd_unix">1.2. BSD Unix</a></li>
<li><a href="#_ios">1.3. iOS</a></li>
<li><a href="#_linux">1.4. Linux</a></li>
<li><a href="#_macos">1.5. MacOS</a></li>
<li><a href="#_windows">1.6. Windows</a></li>
</ul>
</li>
<li><a href="#_device_support">2. Device Support</a>
<ul class="sectlevel2">
<li><a href="#_hardware_implementation">2.1. Hardware Implementation</a></li>
<li><a href="#_null_driver">2.2. Null Driver</a></li>
<li><a href="#_software_implementation">2.3. Software Implementation</a></li>
</ul>
</li>
<li><a href="#_ways_of_checking_for_vulkan">3. Ways of Checking for Vulkan</a>
<ul class="sectlevel2">
<li><a href="#_via_vulkan_installation_analyzer">3.1. VIA (Vulkan Installation Analyzer)</a></li>
<li><a href="#_hello_create_instance">3.2. Hello Create Instance</a></li>
</ul>
</li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink: /Notes/004-3d-rendering/vulkan/chapters/checking_for_support.html
---</p>
</div>
<h1 id="checking-for-support" class="sect0">Checking For Vulkan Support</h1>
<div class="paragraph">
<p>Vulkan requires both a <a href="loader.html#loader">Vulkan Loader</a> and a Vulkan Driver (also referred to as a <em>Vulkan Implementation</em>). The driver is in charge of translating Vulkan API calls into a valid implementation of Vulkan. The most common case is a GPU hardware vendor releasing a driver that is used to run Vulkan on a physical GPU. It should be noted that it is possible to have an entire implementation of Vulkan software based, though the performance impact would be very noticeable.</p>
</div>
<div class="paragraph">
<p>When checking for Vulkan Support it is important to distinguish the difference between <em>platform support</em> and <em>device support</em>.</p>
</div>
<div class="sect1">
<h2 id="_platform_support">1. Platform Support</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The first thing to check is if your <a href="platforms.html#platforms">platform</a> even supports Vulkan. Each platform uses a different mechanism to manage how the <a href="loader.html#loader">Vulkan Loader</a> is implemented. The loader is then in charge of determining if a Vulkan Driver is exposed correctly.</p>
</div>
<div class="sect2">
<h3 id="_android">1.1. Android</h3>
<div class="paragraph">
<p>A simple way of grabbing info on Vulkan is to run the <a href="https://play.google.com/store/apps/details?id=de.saschawillems.vulkancapsviewer&amp;hl=en_US">Vulkan Hardware Capability Viewer</a> app developed by Sascha Willems. This app will not only show if Vulkan is supported, but also all the capabilities the device offers.</p>
</div>
</div>
<div class="sect2">
<h3 id="_bsd_unix">1.2. BSD Unix</h3>
<div class="paragraph">
<p>Grab the <a href="https://vulkan.lunarg.com/sdk/home#linux">Vulkan SDK</a>. Build Vulkan SDK using the command <code>./vulkansdk.sh</code> and then run the <a href="https://vulkan.lunarg.com/doc/sdk/latest/linux/vulkaninfo.html">vulkaninfo</a> executable to easily check for Vulkan support as well as all the capabilities the device offers.</p>
</div>
</div>
<div class="sect2">
<h3 id="_ios">1.3. iOS</h3>
<div class="paragraph">
<p>A simple way of grabbing info on Vulkan is to run the iOS port of the <a href="https://apps.apple.com/us/app/vulkan-capabilities-viewer/id1552796816">Vulkan Hardware Capability Viewer</a> provided by LunarG. This app will not only show if Vulkan is supported, but also all the capabilities the device offers.</p>
</div>
</div>
<div class="sect2">
<h3 id="_linux">1.4. Linux</h3>
<div class="paragraph">
<p>Grab the <a href="https://vulkan.lunarg.com/sdk/home#linux">Vulkan SDK</a> and run the <a href="https://vulkan.lunarg.com/doc/sdk/latest/linux/vulkaninfo.html">vulkaninfo</a> executable to easily check for Vulkan support as well as all the capabilities the device offers.</p>
</div>
</div>
<div class="sect2">
<h3 id="_macos">1.5. MacOS</h3>
<div class="paragraph">
<p>Grab the <a href="https://vulkan.lunarg.com/sdk/home#mac">Vulkan SDK</a> and run the <a href="https://vulkan.lunarg.com/doc/sdk/latest/mac/vulkaninfo.html">vulkaninfo</a> executable to easily check for Vulkan support as well as all the capabilities the device offers.</p>
</div>
</div>
<div class="sect2">
<h3 id="_windows">1.6. Windows</h3>
<div class="paragraph">
<p>Grab the <a href="https://vulkan.lunarg.com/sdk/home#windows">Vulkan SDK</a> and run the <a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/vulkaninfo.html">vulkaninfo.exe</a> executable to easily check for Vulkan support as well as all the capabilities the device offers.</p>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_device_support">2. Device Support</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Just because the platform supports Vulkan does not mean there is device support. For device support, one will need to make sure a Vulkan Driver is available that fully implements Vulkan. There are a few different variations of a Vulkan Driver.</p>
</div>
<div class="sect2">
<h3 id="_hardware_implementation">2.1. Hardware Implementation</h3>
<div class="paragraph">
<p>A driver targeting a physical piece of GPU hardware is the most common case for a Vulkan implementation. It is important to understand that while a certain GPU might have the physical capabilities of running Vulkan, it still requires a driver to control it. The driver is in charge of getting the Vulkan calls mapped to the hardware in the most efficient way possible.</p>
</div>
<div class="paragraph">
<p>Drivers, like any software, are updated and this means there can be many variations of drivers for the same physical device and platform. There is a <a href="https://vulkan.gpuinfo.org/">Vulkan Database</a>, developed and maintained by Sascha Willems, which is the largest collection of recorded Vulkan implementation details</p>
</div>
<div class="admonitionblock note">
<table>
<tr>
<td class="icon">
<div class="title">Note</div>
</td>
<td class="content">
<div class="paragraph">
<p>Just because a physical device or platform isn&#8217;t in the Vulkan Database doesn&#8217;t mean it couldn&#8217;t exist.</p>
</div>
</td>
</tr>
</table>
</div>
</div>
<div class="sect2">
<h3 id="_null_driver">2.2. Null Driver</h3>
<div class="paragraph">
<p>The term &#8220;null driver&#8221; is given to any driver that accepts Vulkan API calls, but does not do anything with them. This is common for testing interactions with the driver without needing any working implementation backing it. Many uses cases such as creating <a href="vulkan_cts.html#vulkan-cts">CTS tests</a> for new features, <a href="https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/master/docs/creating_tests.md#running-tests-on-devsim-and-mockicd">testing the Validation Layers</a>, and more rely on the idea of a null driver.</p>
</div>
<div class="paragraph">
<p>Khronos provides the <a href="https://github.com/KhronosGroup/Vulkan-Tools/tree/master/icd">Mock ICD</a> as one implementation of a null driver that works on various platforms.</p>
</div>
</div>
<div class="sect2">
<h3 id="_software_implementation">2.3. Software Implementation</h3>
<div class="paragraph">
<p>It is possible to create a Vulkan implementation that only runs on the CPU. This is useful if there is a need to test Vulkan that is hardware independent, but unlike the null driver, also outputs a valid result.</p>
</div>
<div class="paragraph">
<p><a href="https://github.com/google/swiftshader">SwiftShader</a> is an example of CPU-based implementation.</p>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_ways_of_checking_for_vulkan">3. Ways of Checking for Vulkan</h2>
<div class="sectionbody">
<div class="sect2">
<h3 id="_via_vulkan_installation_analyzer">3.1. VIA (Vulkan Installation Analyzer)</h3>
<div class="paragraph">
<p>Included in the <a href="https://vulkan.lunarg.com/sdk/home">Vulkan SDK</a> is a utility to check the Vulkan installation on your computer. It is supported on Windows, Linux, and macOS. VIA can:</p>
</div>
<div class="ulist">
<ul>
<li>
<p>Determine the state of Vulkan components on your system</p>
</li>
<li>
<p>Validate that your Vulkan Loader and drivers are installed properly</p>
</li>
<li>
<p>Capture your system state in a form that can be used as an attachment when submitting bugs</p>
</li>
</ul>
</div>
<div class="paragraph">
<p>View the <a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/via.html">SDK documentation on VIA</a> for more information.</p>
</div>
</div>
<div class="sect2">
<h3 id="_hello_create_instance">3.2. Hello Create Instance</h3>
<div class="paragraph">
<p>A simple way to check for Vulkan support cross platform is to create a simple &#8220;Hello World&#8221; Vulkan application. The <code>vkCreateInstance</code> function is used to create a Vulkan Instance and is also the shortest way to write a valid Vulkan application.</p>
</div>
<div class="paragraph">
<p>The Vulkan SDK provides a minimal <a href="https://vulkan.lunarg.com/doc/view/latest/windows/tutorial/html/01-init_instance.html">vkCreateInstance</a> example <code>01-init_instance.cpp</code> that can be used.</p>
</div>
</div>
</div>
</div>