<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#loader">Loader</a>
<ul class="sectlevel1">
<li><a href="#_linking_against_the_loader">1. Linking Against the Loader</a></li>
<li><a href="#_platform_variations">2. Platform Variations</a>
<ul class="sectlevel2">
<li><a href="#_android">2.1. Android</a></li>
<li><a href="#_linux">2.2. Linux</a></li>
<li><a href="#_macos">2.3. MacOS</a></li>
<li><a href="#_windows">2.4. Windows</a></li>
</ul>
</li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink: /Notes/004-3d-rendering/vulkan/chapters/loader.html
---</p>
</div>
<h1 id="loader" class="sect0">Loader</h1>
<div class="paragraph">
<p>The loader is responsible for mapping an application to Vulkan layers and Vulkan installable client drivers (ICD).</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/loader_overview.png" alt="loader_overview.png">
</div>
</div>
<div class="paragraph">
<p>Anyone can create their own Vulkan Loader, as long as they follow the <a href="https://github.com/KhronosGroup/Vulkan-Loader/blob/master/loader/LoaderAndLayerInterface.md">Loader Interface</a>. One can build the <a href="https://github.com/KhronosGroup/Vulkan-Loader/blob/master/BUILD.md">reference loader</a> as well or grab a built version from the <a href="https://vulkan.lunarg.com/sdk/home">Vulkan SDK</a> for selected platforms.</p>
</div>
<div class="sect1">
<h2 id="_linking_against_the_loader">1. Linking Against the Loader</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The <a href="https://github.com/KhronosGroup/Vulkan-Headers">Vulkan headers</a> only provide the Vulkan function prototypes. When building a Vulkan application you have to link it to the loader or you will get errors about undefined references to the Vulkan functions. There are two ways of linking the loader, <a href="https://github.com/KhronosGroup/Vulkan-Loader/blob/master/loader/LoaderAndLayerInterface.md#directly-linking-to-the-loader">directly</a> and <a href="https://github.com/KhronosGroup/Vulkan-Loader/blob/master/loader/LoaderAndLayerInterface.md#indirectly-linking-to-the-loader">indirectly</a>, which should not be confused with &#8220;static and dynamic linking&#8221;.</p>
</div>
<div class="ulist">
<ul>
<li>
<p><a href="https://github.com/KhronosGroup/Vulkan-Loader/blob/master/loader/LoaderAndLayerInterface.md#directly-linking-to-the-loader">Directly linking</a> at compile time</p>
<div class="ulist">
<ul>
<li>
<p>This requires having a built Vulkan Loader (either as a static or dynamic library) that your build system can find.</p>
</li>
<li>
<p>Build systems (Visual Studio, CMake, etc) have documentation on how to link to the library. Try searching &#8220;(InsertBuildSystem) link to external library&#8221; online.</p>
</li>
</ul>
</div>
</li>
<li>
<p><a href="https://github.com/KhronosGroup/Vulkan-Loader/blob/master/loader/LoaderAndLayerInterface.md#indirectly-linking-to-the-loader">Indirectly linking</a> at runtime</p>
<div class="ulist">
<ul>
<li>
<p>Using dynamic symbol lookup (via system calls such as <code>dlsym</code> and <code>dlopen</code>) an application can initialize its own dispatch table. This allows an application to fail gracefully if the loader cannot be found. It also provides the fastest mechanism for the application to call Vulkan functions.</p>
</li>
<li>
<p><a href="https://github.com/zeux/volk/">Volk</a> is an open source implementation of a meta-loader to help simplify this process.</p>
</li>
</ul>
</div>
</li>
</ul>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_platform_variations">2. Platform Variations</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Each platform can set its own rules on how to enforce the Vulkan Loader.</p>
</div>
<div class="sect2">
<h3 id="_android">2.1. Android</h3>
<div class="paragraph">
<p>Android devices supporting Vulkan provide a <a href="https://source.android.com/devices/graphics/implement-vulkan#vulkan_loader">Vulkan loader</a> already built into the OS.</p>
</div>
<div class="paragraph">
<p>A <a href="https://developer.android.com/ndk/guides/graphics/getting-started#using">vulkan_wrapper.c/h</a> file is provided in the Android NDK for indirectly linking. This is needed, in part, because the Vulkan Loader can be different across different vendors and OEM devices.</p>
</div>
</div>
<div class="sect2">
<h3 id="_linux">2.2. Linux</h3>
<div class="paragraph">
<p>The <a href="https://vulkan.lunarg.com/sdk/home">Vulkan SDK</a> provides a pre-built loader for Linux.</p>
</div>
<div class="paragraph">
<p>The <a href="https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started.html">Getting Started</a> page in the Vulkan SDK explains how the loader is found on Linux.</p>
</div>
</div>
<div class="sect2">
<h3 id="_macos">2.3. MacOS</h3>
<div class="paragraph">
<p>The <a href="https://vulkan.lunarg.com/sdk/home">Vulkan SDK</a> provides a pre-built loader for MacOS</p>
</div>
<div class="paragraph">
<p>The <a href="https://vulkan.lunarg.com/doc/sdk/latest/mac/getting_started.html">Getting Started</a> page in the Vulkan SDK explains how the loader is found on MacOS.</p>
</div>
</div>
<div class="sect2">
<h3 id="_windows">2.4. Windows</h3>
<div class="paragraph">
<p>The <a href="https://vulkan.lunarg.com/sdk/home">Vulkan SDK</a> provides a pre-built loader for Windows.</p>
</div>
<div class="paragraph">
<p>The <a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/getting_started.html">Getting Started</a> page in the Vulkan SDK explains how the loader is found on Windows.</p>
</div>
</div>
</div>
</div>